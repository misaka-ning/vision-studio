#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify final-package dataset session persistence across real Qt processes.

For each public YAML/directory source, process one performs actual ONNX CPU
evaluation and writes dataset-session.json through normal dataset import.
Only after it exits does a second --smoke-dataset-session process restore that
same isolated application data. No test writes a presumed successful session.
The successful package matrix supplies a read-only extraction and unchanged
public COCO8 inputs. No private dataset, runtime setup or host installation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "packaging"))
import verify_evaluation_deb as qa

VERSION = "2.6.0"
RESTORE_CHECKS = {
    "session_loaded", "source_entry_preserved", "manifest_source_matches",
    "split_matches", "metadata_matches", "queue_complete_in_manifest_order",
    "first_source_and_pixels_loaded", "no_detection_start", "no_predictions",
    "history_unchanged", "no_evaluation_run", "screenshot_written",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def snapshot(directory):
    return {str(path.relative_to(directory)): qa.digest(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def run_process(arguments, environment, state, prefix, timeout=600):
    stdout, stderr = state / (prefix + ".stdout.log"), state / (prefix + ".stderr.log")
    began = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen([str(value) for value in arguments], cwd=state, env=environment,
                                   stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
            raise RuntimeError(f"Session acceptance timed out; original logs retained in {state}")
    require(code == 0, f"Session acceptance process exited {code}; original logs retained in {state}")
    return {"arguments": [str(value) for value in arguments], "pid": process.pid, "exit_code": code,
            "began_at_utc": began, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - clock, 3),
            "stdout": str(stdout), "stderr": str(stderr),
            "stdout_sha256": qa.digest(stdout), "stderr_sha256": qa.digest(stderr)}


def decode_png(path):
    from PIL import Image
    require(path.is_file() and not path.is_symlink(), f"Missing regular screenshot: {path}")
    with Image.open(path) as image:
        require(image.format == "PNG", "Expected actual PNG screenshot")
        image.load()
        require(image.width > 0 and image.height > 0, "Empty screenshot")
        return {"width": image.width, "height": image.height, "mode": image.mode,
                "sha256": qa.digest(path), "bytes": path.stat().st_size,
                "full_pixel_decode": True}


def audit_restore_report(raw, saved, manifest, state, output, history_sha, evaluations_before):
    """Read-only assertions, also used when final reports are aggregated."""
    require(raw.get("schema_version") == 1 and raw.get("version") == VERSION and
            raw.get("success") is True and not raw.get("error"), "Actual restore smoke did not pass")
    checks = raw.get("checks", {})
    require(set(checks) == RESTORE_CHECKS and all(value is True for value in checks.values()),
            "Actual restore smoke is missing one of the 12 required checks")
    session_path = state / "data/dataset-session.json"
    require(raw.get("session_file") == str(session_path) and
            raw.get("initial_session_json") == saved == raw.get("session_json") == qa.read_json(session_path),
            "A startup did not restore the exact session created by the first process")
    require(raw.get("session_sha256") == qa.digest(session_path), "Saved session hash differs")
    samples = manifest["samples"]
    expected_paths = [sample["image"] for sample in samples]
    require(raw.get("queue_paths") == expected_paths and raw.get("queue_count") == len(samples) > 0,
            "Restored queue lost or reordered actual dataset samples")
    actual_manifest = raw.get("manifest", {})
    for name in ("dataset_source", "data_yaml", "split", "metadata_path", "fingerprint"):
        require(actual_manifest.get(name) == manifest.get(name), f"Restored manifest changed: {name}")
    require(actual_manifest.get("sample_count") == len(samples), "Restored manifest sample count differs")
    expected_truth = [{"index": index, "image": sample["image"],
                       "ground_truth_boxes": len(sample["gt"])} for index, sample in enumerate(samples)]
    require(raw.get("queue_ground_truth") == expected_truth, "Restored queue ground truth differs")
    first = raw.get("first_image", {})
    require(first.get("source") == expected_paths[0] and first.get("non_null") is True and
            first.get("width") == samples[0]["width"] and first.get("height") == samples[0]["height"] and
            first.get("ground_truth_boxes") == sum(gt["class_id"] == 0 for gt in samples[0]["gt"]) and
            first.get("prediction_count") == 0 and first.get("inference_ms") == 0,
            "Restoration did not display the actual first source/GT with no predictions")
    require(raw.get("start_requested_count") == 0 and raw.get("history_unchanged") is True and
            raw.get("history_before_count") == raw.get("history_after_count") and
            raw.get("history_file_sha256_before") == history_sha == raw.get("history_file_sha256_after") and
            qa.digest(state / "data/history.json") == history_sha, "Startup executed inference or altered history")
    require(raw.get("evaluation_directories_before") == raw.get("evaluation_directories_after") and
            snapshot(state / "data/evaluations") == evaluations_before,
            "Startup created or modified an evaluation report/cache")
    require(not (state / "data/results").exists(), "Restore created automatic image-result output")
    return decode_png(output / "dataset-session-restored.png")


def audit_completed_case(row):
    """Independently re-open retained raw outputs, not just a success boolean."""
    state = Path(row["state_directory"])
    output = state / "restored-smoke"
    first_path = state / "evaluation-smoke/evaluation-smoke-report.json"
    second_path = output / "dataset-session-smoke-report.json"
    first = qa.read_json(first_path)
    second = qa.read_json(second_path)
    require(qa.digest(first_path) == row["evaluation_smoke_sha256"] and
            qa.digest(second_path) == row["restore_smoke_sha256"], "Retained session smoke report hash changed")
    require(first.get("success") is True and first.get("version") == VERSION,
            "Session source process did not complete actual evaluation")
    actual_path = Path(first["report_directory"]) / "report.json"
    actual = qa.read_json(actual_path)
    require(qa.digest(actual_path) == row["evaluation_report_sha256"] and
            actual.get("completed_images") == actual.get("planned_images") == row["completed_images"] and
            actual.get("status") == "complete", "Original actual evaluation report changed")
    require(actual["request"]["dataset_class_id"] == actual["request"]["model_class_id"] == 0,
            "Public session acceptance must preserve original dataset/model class zero")
    require(row.get("source_kind") in ("yaml", "directory") and
            actual["manifest"]["dataset_source"] == row["requested_source"] and
            row["saved_session"]["source_path"] == row["requested_source"],
            "The two process case lost its original YAML/directory source")
    for command_key, switch in (("evaluation_command", "--smoke-evaluation"),
                                ("restore_command", "--smoke-dataset-session")):
        command = row[command_key]
        require(command.get("exit_code") == 0 and switch in command.get("arguments", []),
                "Actual distinct source/restore process invocation is missing")
        for prefix in ("stdout", "stderr"):
            require(qa.digest(Path(command[prefix])) == command[prefix + "_sha256"], "Original process log changed")
    first_command, second_command = row["evaluation_command"], row["restore_command"]
    require(isinstance(first_command.get("pid"), int) and isinstance(second_command.get("pid"), int) and
            first_command["pid"] > 0 and second_command["pid"] > 0 and first_command["pid"] != second_command["pid"] and
            datetime.fromisoformat(first_command["ended_at_utc"]) <= datetime.fromisoformat(second_command["began_at_utc"]),
            "Restoration must run in a separate process after the saving process exits")
    return audit_restore_report(second, row["saved_session"], actual["manifest"], state,
                                output, row["history_sha256_before_restore"], row["evaluations_before_restore"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deb", type=Path, required=True)
    parser.add_argument("--matrix-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-provenance", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    args.report = args.report.absolute()
    if args.report.exists() or args.report.is_symlink():
        parser.error("Report exists; retain every earlier success or failure and select a new path")
    workspace = Path(tempfile.mkdtemp(prefix="vision-studio-dataset-session-260-"))
    result = {"schema_version": 1, "version": VERSION, "success": False, "errors": [],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "workdir": str(workspace),
              "scope": "Two real final-package Qt processes per YAML/directory public source: actual ONNX CPU evaluation saves the independent JSON, then startup restores it without inference or report/history mutation.",
              "checks": {}, "cases": {},
              "limitations": ["Four public COCO8 Val images verify persistence and application integration, not football accuracy.",
                              "Train/test split and reviewed single-class metadata restoration use separate native UI regressions.",
                              "No private data, host installation, model training or runtime modification."]}
    try:
        matrix = qa.read_json(args.matrix_report.resolve())
        require(matrix.get("success") is True and matrix.get("eligible_for_release") is True and
                matrix.get("version") == VERSION, "The complete same-version final package matrix must pass first")
        app = Path(matrix["workdir"]) / "extracted/opt/VisionStudio"
        binding = qa.bind_package(args.deb.resolve(), app)
        require(binding["version"] == VERSION and binding["deb_sha256"] == matrix["deb_sha256"],
                "Matrix extraction and final DEB binding differ")
        qa.require_readonly(app)
        result.update(binding)
        result.update(package_bound=True, read_only_application=True,
                      binary_sha256=qa.digest(app / "build/bin/vision-studio"),
                      matrix_report_sha256=qa.digest(args.matrix_report.resolve()))
        result["checks"]["final_deb_binding_and_readonly"] = True
        public_yaml, provenance = args.dataset.resolve(), args.dataset_provenance.resolve()
        require(matrix["dataset"]["yaml"] == str(public_yaml) and
                matrix["dataset"]["yaml_sha256"] == qa.digest(public_yaml), "Public matrix YAML differs")
        public_report = qa.read_json(Path(matrix["cases"]["onnx_cpu"]["report_directory"]) / "report.json")
        public_manifest = public_report["manifest"]
        require(public_manifest["data_yaml"] == str(public_yaml) and public_manifest["split"] == "val" and
                len(public_manifest["samples"]) == 4, "Expected the actual four public COCO8 Val images")
        require(qa.read_json(provenance).get("dataset") == "COCO8", "Missing public COCO8 provenance")
        public_files = {str(public_yaml): qa.digest(public_yaml), str(provenance): qa.digest(provenance)}
        for sample in public_manifest["samples"]:
            for name in ("image", "label"):
                path = Path(sample[name]).resolve()
                require(path.is_relative_to(Path(public_manifest["root"]).resolve()) and
                        qa.digest(path) == sample[name + "_sha256"], "Public input differs from matrix snapshot")
                public_files[str(path)] = qa.digest(path)
        result["public_input_sha256"] = public_files
        model = app / "models/yolov5n.onnx"
        for kind, source in (("yaml", public_yaml), ("directory", public_yaml.parent)):
            state = workspace / kind
            state.mkdir()
            env = qa.isolated_environment(state, app, app / "runtime/bin/python", None, None)
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            request = {"dataset_yaml": str(source), "split": "val", "model_path": str(model),
                       "task": "yolov5", "device": "cpu", "device_index": 0, "input_size": 640,
                       "model_class_id": 0, "dataset_class_id": 0, "confidence": .25, "nms_iou": .45,
                       "color_mode": "color", "scan_start": .05, "scan_end": .95,
                       "scan_step": .01, "match_iou": .5}
            request_file = state / "request.json"
            qa.atomic_json(request_file, request)
            row = {"source_kind": kind, "requested_source": str(source), "state_directory": str(state)}
            result["cases"][kind] = row
            row["evaluation_command"] = run_process([app / "run-installed.sh", "--smoke-evaluation", state / "evaluation-smoke",
                                                "--evaluation-request", request_file], env, state, "evaluation")
            smoke_path = state / "evaluation-smoke/evaluation-smoke-report.json"
            smoke = qa.read_json(smoke_path)
            require(smoke.get("version") == VERSION, "First process reports another application version")
            audited = qa.audit_case(smoke, state, model, "cpu", 0)
            actual_path = Path(audited["report_directory"]) / "report.json"
            actual = qa.read_json(actual_path)
            require(actual["manifest"]["dataset_source"] == str(source) and
                    actual["request"]["sample_images"] == [s["image"] for s in public_manifest["samples"]],
                    "Evaluation did not preserve the requested original source and public queue order")
            session_file = state / "data/dataset-session.json"
            saved = qa.read_json(session_file)
            require(saved == {"schema_version": 1, "source_path": str(source), "split": "val", "metadata_path": ""},
                    "First process did not save the independent session JSON through dataset import")
            require(stat.S_IMODE(session_file.stat().st_mode) == 0o600,
                    "Saved session JSON must be owner-readable/writable only")
            row.update(saved_session=saved, session_sha256_before_restore=qa.digest(session_file),
                       evaluation_smoke_sha256=qa.digest(smoke_path), evaluation_report_sha256=qa.digest(actual_path),
                       completed_images=actual["completed_images"],
                       history_sha256_before_restore=qa.digest(state / "data/history.json"),
                       evaluations_before_restore=snapshot(state / "data/evaluations"))
            output = state / "restored-smoke"
            row["restore_command"] = run_process([app / "run-installed.sh", "--smoke-dataset-session", output],
                                             env, state, "restore", timeout=90)
            raw_path = output / "dataset-session-smoke-report.json"
            row["restore_smoke_sha256"] = qa.digest(raw_path)
            row["screenshot"] = audit_completed_case(row)
            require(qa.digest(session_file) == row["session_sha256_before_restore"],
                    "Session JSON changed on restoration")
            row["success"] = True
            result["checks"][kind + "_two_process_restore"] = True
        require({path: qa.digest(Path(path)) for path in public_files} == public_files,
                "Public input changed during cross-process acceptance")
        qa.require_readonly(app)
        require({name: qa.digest(app / name) for name in binding["code_sha256"]} == binding["code_sha256"],
                "Final application code changed during session acceptance")
        result["checks"].update(public_inputs_unchanged=True, packaged_application_unchanged=True,
                                restore_has_no_inference_history_or_report_writes=True)
        result["success"] = len(result["checks"]) == 6 and all(result["checks"].values())
    except Exception as error:
        result["errors"].append(str(error))
    qa.atomic_json(args.report, result)
    print(json.dumps({"success": result["success"], "report": str(args.report), "workdir": str(workspace),
                      "checks": result["checks"], "errors": result["errors"]}, ensure_ascii=False))
    return 0 if result["success"] and not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
