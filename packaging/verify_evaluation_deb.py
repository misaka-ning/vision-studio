#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise dataset evaluation through the real Qt application/EvaluationWorker.

This tool never runs Python model inference. It calls the installed application's
--smoke-evaluation CLI, then audits its immutable reports/candidate caches and
exports. A prepared CPU runtime, optional existing CUDA runtime and an explicitly
supplied, actually annotated YOLO dataset are inputs. No downloads, drivers,
runtime installation or writes to the input dataset/application are performed.

--app-root + --binary support a development checkout. --deb can extract a private
read-only application or bind an existing extraction to exact DEB code hashes.
Synthetic fixture tests remain separate from this public/pretrained-model matrix.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

import verify_gpu_deb


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise RuntimeError(f"Missing/oversized evidence JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Evidence is not a JSON object: {path}")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run(arguments, environment, directory, log_prefix, timeout=600):
    stdout = directory / (log_prefix + ".stdout.log")
    stderr = directory / (log_prefix + ".stderr.log")
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen([str(x) for x in arguments], env=environment, cwd=directory,
                                   stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
            raise RuntimeError(f"Evaluation process timed out; raw logs retained in {directory}")
    detail = {"arguments": [str(x) for x in arguments], "exit_code": code,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "stdout": str(stdout), "stderr": str(stderr),
              "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr)}
    if code:
        message = stderr.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Process exited {code}: {message}; logs {directory}")
    return detail


def isolated_environment(state, app, python, qt_home, gpu_runtime):
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "VISION_STUDIO_HOME": str(app), "VISION_STUDIO_PYTHON": str(python),
           "VISION_STUDIO_PT_WORKER": str(app / "scripts/pt_worker.py"),
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "QT_QPA_PLATFORM": "offscreen", "QT_QUICK_BACKEND": "software",
           "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu", "YOLO_AUTOINSTALL": "false",
           "YOLOv5_AUTOINSTALL": "false", "MPLBACKEND": "Agg"}
    for key, relative in {"HOME": "home", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "xdg-data",
        "XDG_CACHE_HOME": "cache", "XDG_STATE_HOME": "state", "XDG_RUNTIME_DIR": "xdg-runtime",
        "VISION_STUDIO_DATA_DIR": "data", "YOLO_CONFIG_DIR": "ultralytics", "YOLOV5_CONFIG_DIR": "yolov5",
        "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch", "TMPDIR": "tmp", "CUDA_CACHE_PATH": "cuda-cache"}.items():
        path = state / relative
        path.mkdir(parents=True, mode=0o700)
        env[key] = str(path)
    if gpu_runtime:
        env["VISION_STUDIO_GPU_RUNTIME_DIR"] = str(gpu_runtime)
    else:
        # An absent isolated directory must not discover the operator's GPU state.
        env["VISION_STUDIO_GPU_RUNTIME_DIR"] = str(state / "not-prepared-gpu")
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if key in os.environ:
            env[key] = os.environ[key]
    if qt_home:
        env.update(VISION_STUDIO_QT_HOME=str(qt_home), LD_LIBRARY_PATH=str(qt_home / "lib"),
                   QT_PLUGIN_PATH=str(qt_home / "plugins"),
                   QT_QPA_PLATFORM_PLUGIN_PATH=str(qt_home / "plugins/platforms"),
                   QTWEBENGINEPROCESS_PATH=str(qt_home / "libexec/QtWebEngineProcess"),
                   QTWEBENGINE_RESOURCES_PATH=str(qt_home / "resources"),
                   QTWEBENGINE_LOCALES_PATH=str(qt_home / "translations/qtwebengine_locales"))
    return env


def bind_package(deb, app):
    """Bind all changed evaluation code plus inference/helpers to the final DEB."""
    fields, version = verify_gpu_deb.metadata(deb)
    paths = {"build/bin/vision-studio", "scripts/evaluation_worker.py", "scripts/evaluation_data.py",
             "scripts/evaluation_metrics.py", "scripts/evaluation_reporting.py", "scripts/evaluation_snapshot.py", "scripts/evaluation_import.py",
             "scripts/pt_worker.py", "scripts/model_convert.py",
             "run-installed.sh", "models/yolov5n.onnx", "models/yolov8n.pt"}
    matched = {}
    process = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                relative = name.removeprefix("opt/VisionStudio/")
                if not name.startswith("opt/VisionStudio/") or relative not in paths:
                    continue
                if not member.isfile() or not (app / relative).is_file():
                    raise RuntimeError(f"DEB/application code missing: {relative}")
                value = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        value.update(block)
                if value.hexdigest() != digest(app / relative):
                    raise RuntimeError(f"Application does not match final DEB: {relative}")
                matched[relative] = value.hexdigest()
        if process.wait(timeout=30):
            raise RuntimeError("dpkg-deb failed while binding the application")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if set(matched) != paths:
        raise RuntimeError(f"Required final package code was not bound: {sorted(paths - set(matched))}")
    return {"version": version, "deb_version": fields["Version"], "deb_sha256": digest(deb), "code_sha256": matched}


def finite_metric(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and
                             math.isfinite(value) and 0 <= value <= 1)


def require_readonly(app):
    for directory, _, files in os.walk(app, followlinks=False):
        for path in (Path(directory), *(Path(directory) / name for name in files)):
            if not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise RuntimeError("Final package evaluation requires a read-only extracted tree; "
                                   "omit --app-root to create a private read-only extraction")


def audit_case(smoke, state, model, device, model_id, zero_case=False):
    if smoke.get("success") is not True:
        raise RuntimeError("Application did not report successful real evaluation")
    directory = Path(smoke.get("report_directory", "")).resolve()
    if state.resolve() not in directory.parents:
        raise RuntimeError("Evaluation report escaped its isolated QA state")
    report = read_json(directory / "report.json")
    if report.get("status") != "complete" or report.get("full_split") is not True:
        raise RuntimeError("Matrix case must complete the entire supplied dataset split")
    if report.get("completed_images", 0) < 2 or report.get("completed_images") != report.get("planned_images"):
        raise RuntimeError("Matrix case did not evaluate the full multi-image set")
    request, config, metrics = report["request"], report["config"], report["metrics"]
    if Path(request["model_path"]).resolve() != model.resolve() or request["model_sha256"] != digest(model):
        raise RuntimeError("Evaluation model path/SHA differs from the intended actual model")
    if request["model_class_id"] != model_id or request["dataset_class_id"] != model_id:
        raise RuntimeError("Original model/dataset class IDs were changed or remapped")
    if config.get("device") != device or config.get("resolved_device") != device:
        raise RuntimeError("Explicit compute device was not actually used")
    if config.get("model_class_count", 0) <= model_id:
        raise RuntimeError("Actual model class metadata is missing or out of range")
    for key in ("mAP50", "mAP50_95", "precision", "recall", "F1", "no_ball_false_positive_rate"):
        if not finite_metric(metrics.get(key)):
            raise RuntimeError(f"Invalid measured metric {key}")
    rows = report.get("rows", [])
    if len(rows) != 91 or abs(rows[0]["confidence"] - .05) > 1e-9 or abs(rows[-1]["confidence"] - .95) > 1e-9:
        raise RuntimeError("The actual evaluation did not generate all 91 threshold points")
    cache = directory / "predictions.jsonl"
    if report["prediction_sha256"] != digest(cache):
        raise RuntimeError("Raw candidate cache SHA differs from the completed report")
    predictions = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    if len(predictions) != report["completed_images"]:
        raise RuntimeError("Raw candidate count does not match completed images")
    for index, row in enumerate(predictions):
        if row["image"] != request["sample_images"][index] or row["device"] != device:
            raise RuntimeError("Candidate cache index/actual device differs from the frozen task")
        backend = row.get("backend", "")
        if not isinstance(row.get("inference_ms"), (int, float)) or row["inference_ms"] <= 0:
            raise RuntimeError("Actual forward timing is absent from the candidate cache")
        if device == "cpu" and "CPU" not in backend:
            raise RuntimeError("Candidate cache did not establish the actual CPU backend")
        if device == "cuda" and ("CUDA" not in backend or not row.get("device_name")):
            raise RuntimeError("Candidate cache did not establish a real CUDA backend")
        for value in row["predictions"]:
            if value["class_id"] != model_id:
                raise RuntimeError("Cached output did not preserve/filter the original class ID")
    if zero_case:
        if metrics.get("predicted_boxes") != 0:
            raise RuntimeError("High-threshold zero-result case unexpectedly kept predictions")
    elif metrics.get("gt_boxes", 0) <= 0 or sum(len(row["predictions"]) for row in predictions) <= 0:
        raise RuntimeError("Positive public-GT case has no GT or actual model candidates")
    artifacts = {}
    for name in ("thresholds.csv", "per-image.csv", "errors.csv", "REPORT.md", "threshold-performance.png",
                 "threshold-performance.svg", "negative-false-positive-rate.png", "negative-false-positive-rate.svg"):
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Actual report export missing: {name}")
        if path.suffix == ".png" and path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Invalid exported report PNG: {name}")
        artifacts[name] = digest(path)
    source_files = {}
    for sample in report["manifest"]["samples"]:
        for key in ("image", "label"):
            if digest(sample[key]) != sample[key + "_sha256"]:
                raise RuntimeError("Real dataset input changed after evaluation")
            source_files[sample[key]] = sample[key + "_sha256"]
    exports = smoke.get("exports")
    if not isinstance(exports, list) or len(exports) != 3:
        raise RuntimeError("Actual workbench selected-image PNG/JSON/CSV export is incomplete")
    export_paths = {}
    for entry in exports:
        path = Path(entry["path"]).resolve()
        if state.resolve() not in path.parents or not path.is_file() or digest(path) != entry["sha256"]:
            raise RuntimeError("Actual selected-image export path/hash is invalid")
        export_paths[path.suffix] = path
    if set(export_paths) != {".png", ".json", ".csv"}:
        raise RuntimeError("Actual selected-image export formats differ from PNG/JSON/CSV")
    exported = read_json(export_paths[".json"])
    selected = predictions[0]
    evaluation = exported.get("evaluation", {})
    if exported.get("source") != selected["image"] or evaluation.get("has_ground_truth") is not True or \
            evaluation.get("matching_available") is not True or Path(evaluation.get("report_directory", "")).resolve() != directory:
        raise RuntimeError("Selected-image export is not tied to the same real completed evaluation")
    threshold = evaluation["analysis_confidence"]
    expected = [p for p in selected["predictions"] if p["class_id"] == model_id and p["confidence"] >= threshold]
    if len(exported.get("predictions", [])) != len(expected) or \
            any(p["class_id"] != model_id for p in exported["predictions"]):
        raise RuntimeError("Exported predictions do not match the analyzed cached threshold/original IDs")
    expected_gt = [g for g in selected["gt"] if g["class_id"] == model_id]
    if len(exported.get("ground_truth", [])) != len(expected_gt):
        raise RuntimeError("Exported GT does not match the frozen dataset class")
    with export_paths[".csv"].open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(expected) or any(int(p["class_id"]) != model_id for p in csv_rows):
        raise RuntimeError("CSV and JSON exports disagree about the analyzed detections")
    return {"success": True, "report_directory": str(directory), "report_sha256": digest(directory / "report.json"),
            "candidate_sha256": digest(cache), "metrics": metrics, "rows": len(rows), "config": config,
            "dataset_fingerprint": report["dataset_fingerprint"], "model_sha256": request["model_sha256"],
            "completed_images": report["completed_images"], "artifacts_sha256": artifacts,
            "source_files_verified": len(source_files), "selected_exports": exports,
            "raw_candidates": predictions}


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def compare(left, right, score_tolerance=.005, iou_minimum=.95):
    maximum_score_error, minimum_iou = 0., 1.
    for a, b in zip(left["raw_candidates"], right["raw_candidates"]):
        first = [p for p in a["predictions"] if p["confidence"] >= .25]
        second = [p for p in b["predictions"] if p["confidence"] >= .25]
        if a["image"] != b["image"] or len(first) != len(second):
            raise RuntimeError("Working-point image order/prediction counts differ between equivalent backends")
        remaining = list(second)
        for prediction in first:
            if not remaining:
                raise RuntimeError("Missing equivalent-backend prediction")
            target = max(remaining, key=lambda p: box_iou(prediction["box"], p["box"]))
            remaining.remove(target)
            overlap, error = box_iou(prediction["box"], target["box"]), abs(prediction["confidence"] - target["confidence"])
            if overlap < iou_minimum or error > score_tolerance:
                raise RuntimeError(f"Equivalent-backend prediction mismatch: IoU={overlap}, score error={error}")
            minimum_iou, maximum_score_error = min(minimum_iou, overlap), max(maximum_score_error, error)
    metric_differences = {}
    for key in ("mAP50", "mAP50_95", "precision", "recall", "F1", "no_ball_false_positive_rate"):
        a, b = left["metrics"][key], right["metrics"][key]
        if (a is None) != (b is None):
            raise RuntimeError(f"Equivalent-backend metric availability differs: {key}")
        delta = 0 if a is None else abs(a - b)
        metric_differences[key] = delta
        if delta > .02:
            raise RuntimeError(f"Equivalent-backend {key} differs by {delta} (limit 0.02)")
    return {"success": True, "minimum_working_box_iou": minimum_iou,
            "maximum_working_score_error": maximum_score_error, "metric_absolute_differences": metric_differences,
            "tolerances": {"box_iou_minimum": iou_minimum, "score_absolute_maximum": score_tolerance,
                           "metric_absolute_maximum": .02},
            "note": "Same weights only; different YOLO architectures are not expected to have equal accuracy."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deb", type=Path)
    parser.add_argument("--app-root", type=Path)
    parser.add_argument("--binary", type=Path, help="Development executable; final DEB QA uses its launcher")
    parser.add_argument("--python", type=Path, help="Development CPU Python; packaged default is runtime/bin/python")
    parser.add_argument("--qt-home", type=Path, help="Development Qt SDK, never modifies the SDK")
    parser.add_argument("--gpu-runtime-dir", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true", help="Explicit CPU-only development subset, not full final GPU QA")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-provenance", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    project_cmake = Path(__file__).resolve().parents[1] / "CMakeLists.txt"
    default_version = verify_gpu_deb.release_metadata.cmake_application_version(project_cmake.read_text(encoding="utf-8"))
    parser.add_argument("--expected-version", default=default_version)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    protected = [path.resolve() for path in (args.dataset, args.deb, args.binary, args.python) if path]
    destination = args.report.resolve()
    if destination in protected or (args.gpu_runtime_dir and
            (destination == args.gpu_runtime_dir.resolve() or args.gpu_runtime_dir.resolve() in destination.parents)):
        parser.error("--report must not overwrite an input or write inside the prepared GPU runtime")
    report = {"schema_version": 1, "success": False, "version": args.expected_version,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "deb_sha256": None,
              "scope": "cpu_only_development" if args.cpu_only else "cpu_cuda_matrix", "checks": {},
              "cases": {}, "comparisons": {}, "errors": [],
              "dataset_kind": "supplied_real_ground_truth", "synthetic_model_tests": "separate CTest worker suite",
              "limitations": ["COCO8 is a tiny pipeline integration set, not a deployment benchmark or football accuracy claim.",
                              "CUDA comparisons apply to identical weights and record explicit numeric tolerances."]}
    workspace = None
    try:
        if os.geteuid() == 0:
            raise RuntimeError("Run evaluation QA as an ordinary user")
        if not args.app_root and not args.deb:
            raise RuntimeError("Provide --app-root or --deb")
        if not args.cpu_only and not args.gpu_runtime_dir:
            raise RuntimeError("Full matrix requires an existing --gpu-runtime-dir; use --cpu-only explicitly for development")
        if args.work_dir:
            workspace = args.work_dir.resolve()
            workspace.mkdir(parents=True, exist_ok=False)
        else:
            workspace = Path(tempfile.mkdtemp(prefix="vision-studio-evaluation-qa-"))
        report["workdir"] = str(workspace)
        if args.app_root:
            app = args.app_root.resolve()
        else:
            deb = args.deb.resolve()
            verify_gpu_deb.check_archive(deb)
            subprocess.run(["dpkg-deb", "--extract", str(deb), str(workspace / "extracted")], check=True, timeout=300)
            app = workspace / "extracted/opt/VisionStudio"
            verify_gpu_deb.readonly_application(app)
        binary = args.binary.resolve() if args.binary else app / "run-installed.sh"
        # Keep the venv entry point: resolving its symlink to /usr/bin/python
        # would silently discard the packaged environment's site-packages.
        python = args.python.expanduser().absolute() if args.python else app / "runtime/bin/python"
        dataset = args.dataset.resolve()
        if not app.is_dir() or not binary.is_file() or not os.access(binary, os.X_OK) or not python.is_file() or not dataset.is_file():
            raise RuntimeError("Application executable, CPU Python or annotated dataset is missing")
        if args.deb:
            if args.binary or args.python or args.qt_home:
                raise RuntimeError("Final DEB QA must use the package's own launcher, Python and Qt")
            binding = bind_package(args.deb.resolve(), app)
            if binding["version"] != args.expected_version:
                raise RuntimeError("Final DEB application version differs from expected version")
            report.update(binding)
            require_readonly(app)
            report["package_bound"] = True
            report["read_only_application"] = True
        else:
            report["package_bound"] = False
            report["read_only_application"] = False
        report["binary_sha256"] = digest(app / "build/bin/vision-studio" if not args.binary else binary)
        report["dataset"] = {"yaml": str(dataset), "yaml_sha256": digest(dataset), "split": args.split}
        if args.dataset_provenance:
            report["dataset"]["provenance"] = read_json(args.dataset_provenance.resolve())
        qt = args.qt_home.resolve() if args.qt_home else None
        gpu = args.gpu_runtime_dir.resolve() if args.gpu_runtime_dir else None
        conversion = workspace / "torchscript-export"
        conversion.mkdir()
        env = isolated_environment(conversion, app, python, qt, gpu)
        torchscript = conversion / "yolov8n.torchscript"
        report["torchscript_export"] = run([python, app / "scripts/model_convert.py", "--source", app / "models/yolov8n.pt",
            "--format", "torchscript", "--output", torchscript, "--image-size", "640"], env, conversion, "conversion")
        if not torchscript.is_file():
            raise RuntimeError("Actual packaged converter did not create TorchScript")
        report["torchscript_export"].update(source_sha256=digest(app / "models/yolov8n.pt"), output_sha256=digest(torchscript))
        cases = [("onnx_cpu", app / "models/yolov5n.onnx", "yolov5", "cpu", 0, False),
                 ("pt_cpu", app / "models/yolov8n.pt", "yolov8", "cpu", 0, False),
                 ("torchscript_cpu", torchscript, "yolov8", "cpu", 0, False),
                 ("zero_result_id32_cpu", app / "models/yolov8n.pt", "yolov8", "cpu", 32, True)]
        if not args.cpu_only:
            cases += [("onnx_cuda", app / "models/yolov5n.onnx", "yolov5", "cuda", 0, False),
                      ("pt_cuda", app / "models/yolov8n.pt", "yolov8", "cuda", 0, False)]
        for name, model, task, device, class_id, zero in cases:
            state = workspace / name
            state.mkdir()
            env = isolated_environment(state, app, python, qt, gpu)
            if device == "cuda" and task == "yolov5":
                env["VISION_STUDIO_ORT_PROFILE_PREFIX"] = str(state / "ort-evaluation")
            request = {"dataset_yaml": str(dataset), "split": args.split, "model_path": str(model), "task": task,
                "device": device, "device_index": args.gpu_index, "input_size": 640, "model_class_id": class_id,
                "dataset_class_id": class_id, "confidence": 1.0 if zero else .25, "nms_iou": .45,
                "color_mode": "color", "scan_start": .05, "scan_end": .95, "scan_step": .01, "match_iou": .5}
            request_file = state / "request.json"
            atomic_json(request_file, request)
            output = state / "ui-smoke"
            try:
                detail = run([binary, "--smoke-evaluation", output, "--evaluation-request", request_file], env, state, "application")
                smoke = read_json(output / "evaluation-smoke-report.json")
                if smoke.get("version") != args.expected_version:
                    raise RuntimeError("Application reported a different version")
                result = audit_case(smoke, state, model, device, class_id, zero)
                result["command"] = detail
                result["smoke_report"] = str(output / "evaluation-smoke-report.json")
                if name == "onnx_cuda":
                    traces = list(state.glob("ort-evaluation*.json"))
                    events = [event for trace in traces for event in json.loads(trace.read_text())
                              if event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider"]
                    if not events:
                        raise RuntimeError("No actual CUDA kernel events in the C++ ONNX Runtime profile")
                    result["cuda_node_events"] = len(events)
                    result["cuda_profiles"] = [{"file": str(p), "sha256": digest(p)} for p in traces]
                report["cases"][name] = result
                report["checks"][name] = True
            except Exception as error:
                report["cases"][name] = {"success": False, "error": str(error), "workdir": str(state)}
                report["checks"][name] = False
                report["errors"].append(f"{name}: {error}")
            atomic_json(args.report, report)
        for name, left, right in (("pt_torchscript_cpu", "pt_cpu", "torchscript_cpu"),
                                   ("onnx_cpu_cuda", "onnx_cpu", "onnx_cuda"), ("pt_cpu_cuda", "pt_cpu", "pt_cuda")):
            if args.cpu_only and "cuda" in name:
                continue
            try:
                if not report["checks"].get(left) or not report["checks"].get(right):
                    raise RuntimeError("Equivalent-backend comparison lacks successful cases")
                report["comparisons"][name] = compare(report["cases"][left], report["cases"][right])
                report["checks"][name] = True
            except Exception as error:
                report["comparisons"][name] = {"success": False, "error": str(error)}
                report["checks"][name] = False
                report["errors"].append(f"{name}: {error}")
    except Exception as error:
        report["errors"].append(str(error))
    report["success"] = bool(report["checks"]) and all(report["checks"].values()) and not report["errors"]
    report["eligible_for_release"] = bool(report["success"] and report.get("package_bound") and
                                          report.get("read_only_application") and not args.cpu_only)
    atomic_json(args.report, report)
    print(json.dumps({"success": report["success"], "report": str(args.report.resolve()),
                      "workdir": str(workspace), "checks": report["checks"], "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
