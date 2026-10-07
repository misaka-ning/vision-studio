#!/usr/bin/env python3
"""Verify real CUDA and CPU inference through a DEB's installed launcher.

Runs as an ordinary user, extracts a private copy (never installs with dpkg),
makes the application read-only and isolates all application HOME/XDG state.
The optional, already-prepared GPU environment is read from --runtime-dir;
the package's CPU Python/Netron runtime is never replaced or modified.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time


PINS = {"schema": 1, "torch_version": "2.9.1+cu128", "torch_cuda": "12.8", "ort_version": "1.23.2"}
LOG_LIMIT = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path):
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise RuntimeError(f"Missing or oversized JSON report: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path.name}")
    return value


def metadata(deb):
    result = subprocess.run(["dpkg-deb", "--field", str(deb)], capture_output=True,
                            text=True, timeout=60, check=True)
    fields = {}
    for line in result.stdout.splitlines():
        if line and not line[0].isspace() and ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    if fields.get("Package") != "vision-studio" or fields.get("Architecture") != "amd64":
        raise RuntimeError("Expected a vision-studio amd64 DEB")
    match = re.fullmatch(r"(?:\d+:)?(\d+\.\d+\.\d+)(?:-[^\s]+)?", fields.get("Version", ""))
    if not match:
        raise RuntimeError("DEB has no valid application semantic version")
    return fields, match.group(1)


def check_archive(deb):
    """Reject escaping links/paths before extracting or changing permissions."""
    process = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    entries = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.removeprefix("./").rstrip("/")
                if name in ("", "."):
                    continue
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or name in entries:
                    raise RuntimeError(f"Unsafe/duplicate DEB entry: {member.name}")
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise RuntimeError(f"Unsupported DEB entry: {member.name}")
                if member.mode & 0o7000:
                    raise RuntimeError(f"Unexpected privileged permission bits: {name}")
                entries[name] = member
                if member.issym() or member.islnk():
                    target = member.linkname
                    if target.startswith("/"):
                        if not member.issym() or target != "/usr/bin/python3.10":
                            raise RuntimeError(f"Unexpected external DEB link: {name}")
                    else:
                        resolved = posixpath.normpath(posixpath.join(
                            str(path.parent) if member.issym() else "", target))
                        if resolved == ".." or resolved.startswith("../"):
                            raise RuntimeError(f"Escaping DEB link: {name}")
        for name in entries:
            for parent in PurePosixPath(name).parents:
                entry = entries.get(str(parent))
                if entry and not entry.isdir():
                    raise RuntimeError(f"DEB entry traverses a link/non-directory: {name}")
        for member in entries.values():
            if member.islnk():
                target = entries.get(posixpath.normpath(member.linkname))
                if not target or not target.isfile():
                    raise RuntimeError(f"DEB hard link does not target a regular archived file: {member.name}")
        if process.wait(timeout=30) != 0:
            raise RuntimeError("dpkg-deb failed to read the archive")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def readonly_application(app):
    if not app.is_dir() or app.is_symlink():
        raise RuntimeError("Extracted application directory is missing or a link")
    for directory, _, files in os.walk(app, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        path = Path(directory)
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    try:
        (app / ".qa-write-probe").write_text("write test", encoding="utf-8")
    except PermissionError:
        return
    raise RuntimeError("Extracted application is writable; read-only package QA cannot proceed")


def isolated_environment(state, runtime):
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "VISION_STUDIO_GPU_RUNTIME_DIR": str(runtime), "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "QT_QPA_PLATFORM": "offscreen",
           # This affects the WebEngine renderer only; inference must use CUDA.
           "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu", "QT_QUICK_BACKEND": "software"}
    directories = {"HOME": "home", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
                   "XDG_CACHE_HOME": "cache", "XDG_STATE_HOME": "state", "XDG_RUNTIME_DIR": "runtime",
                   "YOLO_CONFIG_DIR": "ultralytics", "YOLOV5_CONFIG_DIR": "yolov5",
                   "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch", "TMPDIR": "tmp",
                   "CUDA_CACHE_PATH": "cuda-cache"}
    for key, directory in directories.items():
        path = state / directory
        path.mkdir(parents=True, mode=0o700)
        env[key] = str(path)
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def log_contents(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size > LOG_LIMIT:
            stream.seek(-LOG_LIMIT, os.SEEK_END)
        text = stream.read().decode("utf-8", errors="replace")
    return {"bytes": size, "truncated": size > LOG_LIMIT, "text": text}


def valid_integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_device(report, mode, index, backend):
    if report.get("requested_device") != mode or report.get("actual_device") != mode:
        raise RuntimeError(f"Expected requested and actual device {mode}; got "
                           f"{report.get('requested_device')}/{report.get('actual_device')}")
    expected_index = index if mode == "cuda" else -1
    actual_index = report.get("device_index")
    if not isinstance(actual_index, int) or isinstance(actual_index, bool) or actual_index != expected_index:
        raise RuntimeError(f"Wrong actual device index: {actual_index}")
    if not isinstance(report.get("device_name"), str) or not report["device_name"].strip():
        raise RuntimeError("Actual device name is missing")
    if report.get("device_notice") != "":
        raise RuntimeError("Explicit device QA unexpectedly reported a fallback/notice")
    actual_backend = report.get("backend", "")
    if not isinstance(actual_backend, str) or backend not in actual_backend or mode.upper() not in actual_backend:
        raise RuntimeError(f"Unexpected inference backend: {actual_backend}")


def validate_exports(output, smoke, version, mode, index, backend, expected_model):
    exported = []
    for path in sorted(output.glob("*.json")):
        if path.name == "smoke-report.json":
            continue
        value = json_object(path)
        if value.get("application") != "Vision Studio":
            continue
        validate_device(value, mode, index, backend)
        if value.get("version") != version or value.get("model") != expected_model.name:
            raise RuntimeError(f"Exported model/application version mismatch: {path.name}")
        model_file = value.get("model_file")
        if not isinstance(model_file, str) or Path(model_file).resolve() != expected_model.resolve():
            raise RuntimeError(f"Inference did not load the model from the extracted DEB: {path.name}")
        predictions = value.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise RuntimeError(f"Exported predictions are missing: {path.name}")
        if not valid_integer(value.get("width"), 1) or not valid_integer(value.get("height"), 1):
            raise RuntimeError(f"Exported image size is invalid: {path.name}")
        if len(predictions) != smoke["predictions"]:
            raise RuntimeError(f"Exported prediction count differs from smoke report: {path.name}")
        for field in ("backend", "device_name", "device_index"):
            if value.get(field) != smoke.get(field):
                raise RuntimeError(f"Exported {field} differs from smoke report: {path.name}")
        png, csv = path.with_suffix(".png"), path.with_suffix(".csv")
        if not png.is_file() or not csv.is_file() or not csv.stat().st_size:
            raise RuntimeError(f"Exported PNG/CSV is missing: {path.name}")
        with png.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"Invalid exported PNG: {png.name}")
        exported.append({"file": path.name, "sha256": sha256(path), "report": value,
                         "png_sha256": sha256(png), "csv_sha256": sha256(csv)})
    if not exported:
        raise RuntimeError("No real exported result JSON/PNG/CSV was found")
    return exported


def smoke_case(app, workspace, name, mode, index, flag, backend, model, version, runtime):
    state = workspace / name
    state.mkdir(mode=0o700)
    output = state / "results"
    environment = isolated_environment(state, runtime)
    arguments = [str(app / "run-installed.sh"), "--device", mode, "--gpu-index", str(index), flag, str(output)]
    evidence = {"success": False, "arguments": arguments, "error": "", "exit_code": None,
                "timed_out": False, "report": {}, "exports": []}
    started = time.monotonic()
    stdout, stderr = state / "stdout.log", state / "stderr.log"
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(arguments, env=environment, cwd=state, stdout=out, stderr=err,
                                       start_new_session=True)
            try:
                evidence["exit_code"] = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                evidence["timed_out"] = True
                # Kill only the process group created for this isolated run.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
                evidence["exit_code"] = process.returncode
                raise RuntimeError("Packaged inference smoke timed out after 240 seconds")
        if evidence["exit_code"] != 0:
            raise RuntimeError(f"Installed launcher exited {evidence['exit_code']}")
        report = json_object(output / "smoke-report.json")
        evidence["report"] = report
        if report.get("success") is not True or not valid_integer(report.get("predictions"), 1):
            raise RuntimeError("Smoke did not report successful real predictions")
        latency = report.get("inference_ms")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("Smoke has no valid measured inference time")
        validate_device(report, mode, index, backend)
        evidence["exports"] = validate_exports(output, report, version, mode, index, backend, app / "models" / model)
        evidence["success"] = True
    except Exception as error:
        evidence["error"] = str(error)
        # Preserve a failed application's structured report as well as logs.
        if not evidence["report"] and (output / "smoke-report.json").is_file():
            try:
                evidence["report"] = json_object(output / "smoke-report.json")
            except Exception:
                pass
    evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
    evidence["stdout"] = log_contents(stdout) if stdout.exists() else {"text": "", "bytes": 0, "truncated": False}
    evidence["stderr"] = log_contents(stderr) if stderr.exists() else {"text": "", "bytes": 0, "truncated": False}
    return evidence


def cleanup(workspace, identity):
    if workspace.is_symlink() or (workspace.stat().st_dev, workspace.stat().st_ino) != identity:
        raise RuntimeError("Temporary QA directory identity changed; refusing cleanup")
    for directory, _, files in os.walk(workspace, followlinks=False):
        Path(directory).chmod(0o700)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    shutil.rmtree(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deb", type=Path)
    parser.add_argument("--runtime-dir", type=Path, required=True, help="Existing prepared user GPU environment")
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA device index (default 0)")
    parser.add_argument("--report", type=Path, required=True, help="Write JSON QA evidence including failure logs")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep this run's extraction and complete logs")
    args = parser.parse_args()
    deb, runtime, destination = args.deb.resolve(), args.runtime_dir.resolve(), args.report.resolve()
    # Neither input package nor the existing GPU environment is writable QA state.
    if destination == deb or destination == runtime or runtime in destination.parents:
        parser.error("--report must be outside the input DEB and GPU runtime")
    report = {"schema": 1, "success": False, "version": "", "deb_version": "", "deb": str(deb),
              "deb_sha256": "", "runtime_dir": str(runtime), "gpu_index": args.gpu_index,
              "checks": {"packaged_onnx_cuda": False, "packaged_pt_cuda": False, "cpu_regression": False},
              "evidence": {}, "errors": []}
    workspace = None
    identity = None
    try:
        if os.geteuid() == 0:
            raise RuntimeError("GPU DEB QA must run as an ordinary non-root user")
        if not 0 <= args.gpu_index <= 63:
            raise RuntimeError("GPU index must be between 0 and 63")
        if not deb.is_file() or not shutil.which("dpkg-deb"):
            raise RuntimeError("Input DEB or dpkg-deb is missing")
        fields, version = metadata(deb)
        report.update(version=version, deb_version=fields["Version"], deb_sha256=sha256(deb))
        ready = json_object(runtime / "ready.json")
        if type(ready.get("schema")) is not int or any(ready.get(key) != value for key, value in PINS.items()):
            raise RuntimeError("GPU runtime ready.json does not match release pins/schema")
        if not os.access(runtime / "bin/python", os.X_OK) or not any(
                runtime.glob("lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2")):
            raise RuntimeError("Prepared GPU runtime has incomplete Python/ONNX paths")
        report["runtime_pins"] = PINS
        check_archive(deb)
        workspace = Path(tempfile.mkdtemp(prefix="vision-studio-gpu-deb-qa-"))
        identity = (workspace.stat().st_dev, workspace.stat().st_ino)
        extracted = workspace / "root"
        subprocess.run(["dpkg-deb", "--extract", str(deb), str(extracted)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        app = extracted / "opt/VisionStudio"
        if not (app / "run-installed.sh").is_file():
            raise RuntimeError("DEB has no installed launcher")
        readonly_application(app)
        for name, mode, flag, backend, model in (
            ("onnx_cuda", "cuda", "--smoke", "ONNX Runtime", "yolov5n.onnx"),
            ("pt_cuda", "cuda", "--smoke-pt", "PyTorch", "yolov8n.pt"),
            ("onnx_cpu", "cpu", "--smoke", "OpenCV DNN", "yolov5n.onnx"),
            ("pt_cpu", "cpu", "--smoke-pt", "PyTorch", "yolov8n.pt"),
        ):
            result = smoke_case(app, workspace, name, mode, args.gpu_index, flag, backend, model, version, runtime)
            report["evidence"][name] = result
            if not result["success"]:
                report["errors"].append(f"{name}: {result['error']}")
        report["checks"] = {
            "packaged_onnx_cuda": report["evidence"]["onnx_cuda"]["success"],
            "packaged_pt_cuda": report["evidence"]["pt_cuda"]["success"],
            "cpu_regression": all(report["evidence"][name]["success"] for name in ("onnx_cpu", "pt_cpu")),
        }
    except subprocess.CalledProcessError as error:
        report["errors"].append(str(error))
        report["command_failure"] = {
            "exit_code": error.returncode,
            "stdout": (error.stdout or b"").decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else error.stdout,
            "stderr": (error.stderr or b"").decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else error.stderr,
        }
    except Exception as error:
        report["errors"].append(str(error))
    finally:
        if workspace:
            if args.keep_workdir:
                report["workdir"] = str(workspace)
            else:
                try:
                    cleanup(workspace, identity)
                except Exception as error:
                    report["errors"].append(f"QA cleanup failed: {error}")
                    report["workdir"] = str(workspace)
    report["success"] = all(report["checks"].values()) and not report["errors"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so interrupted verification does not leave a partial QA report.
    handle, name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print(json.dumps({"success": report["success"], "version": report["version"],
                      "checks": report["checks"], "errors": report["errors"], "report": str(destination)},
                     ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
