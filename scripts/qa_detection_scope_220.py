#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run and retain real 2.2.0 detection regression evidence.

This is a CTest evidence runner, not a substitute for package acceptance. It
never installs software, prepares a GPU runtime, downloads models, or writes
to existing application data. A fresh output directory is required. Reports
describe only the selected tests actually executed and reject skipped tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


DEFAULT_TESTS = (
    "vision-scope-filter-ui-tests",
    "vision-core-tests",
    "vision-inference-delivery-tests",
    "vision-recording-tests",
    "vision-pt-tests",
    "vision-ui-tests",
    "vision-batch-responsiveness-tests",
)


def inspect_output(text: str, junit_text: str | None) -> tuple[list[dict], bool, int, bool]:
    """Keep internal Qt/Python skips visible when CTest reports a passed suite."""
    totals = [dict(zip(("passed", "failed", "skipped", "blacklisted"), map(int, match)))
              for match in re.findall(
                  r"Totals:\s+(\d+) passed,\s+(\d+) failed,\s+(\d+) skipped,\s+(\d+) blacklisted", text)]
    skipped = bool(re.search(r"(?:SKIP\s*:|\*\*\*Skipped|Not Run)", text))
    skipped = skipped or any(row["skipped"] or row["blacklisted"] for row in totals)
    # unittest exits 0 for OK (skipped=N), so CTest's suite-level JUnit may
    # contain no <skipped>. Its verbose case lines can also be the only
    # available evidence when a larger test command omits the final summary.
    skipped = skipped or bool(re.search(
        r"\bskipped\s*=\s*[1-9]\d*\b|\.\.\.\s+skipped\b|\b[1-9]\d*\s+skipped\b",
        text, flags=re.IGNORECASE))
    failures = any(row["failed"] for row in totals)
    cases = []
    if junit_text is not None:
        root = ET.fromstring(junit_text)
        cases = list(root.iter("testcase"))
        skipped = skipped or any(case.find("skipped") is not None for case in cases)
        failures = failures or any(case.find("failure") is not None or case.find("error") is not None
                                   for case in cases)
    return totals, skipped, len(cases), failures


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(command: list[str], *, environment: dict[str, str], timeout: int) -> tuple[int, str, float]:
    started = time.monotonic()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output, round(time.monotonic() - started, 3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return 124, output + "\nQA runner: command timed out.\n", round(time.monotonic() - started, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for raw logs, JUnit and measured report; never overwritten")
    parser.add_argument("--test", action="append", dest="tests",
                        help="Exact CTest test name; repeat to replace the default regression selection")
    parser.add_argument("--with-gpu", action="store_true",
                        help="Explicitly run real CUDA tests using an already prepared local runtime")
    parser.add_argument("--gpu-runtime", type=Path)
    parser.add_argument("--python", type=Path,
                        help="Existing CPU runtime interpreter; does not create or modify an environment")
    parser.add_argument("--binary", type=Path,
                        help="Application binary to fingerprint (default <build-dir>/bin/vision-studio)")
    parser.add_argument("--timeout", type=int, default=900, help="Per-suite wall-clock safety limit")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not (build / "CTestTestfile.cmake").is_file():
        parser.error("--build-dir must be an existing configured testing build")
    if output.exists():
        parser.error("Output directory already exists; choose a fresh path to preserve prior evidence")
    if args.with_gpu and (args.gpu_runtime is None or
                          not (args.gpu_runtime / "ready.json").is_file()):
        parser.error("--with-gpu requires --gpu-runtime with an existing ready.json; no setup is attempted")
    if args.python is not None and not args.python.is_file():
        parser.error("--python does not name an existing interpreter")
    tests = list(dict.fromkeys(args.tests or DEFAULT_TESTS))
    if args.with_gpu and "vision-gpu-tests" not in tests:
        tests.append("vision-gpu-tests")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in tests):
        parser.error("Unexpected test name")
    output.mkdir(parents=True)
    environment = dict(os.environ)
    for name, relative in {
        "VISION_STUDIO_DATA_DIR": "app-data", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data", "XDG_CACHE_HOME": "cache", "XDG_STATE_HOME": "state",
        "YOLO_CONFIG_DIR": "ultralytics", "YOLOV5_CONFIG_DIR": "yolov5", "MPLCONFIGDIR": "matplotlib",
    }.items():
        directory = output / "isolated" / relative
        directory.mkdir(parents=True, mode=0o700)
        environment[name] = str(directory)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    artifacts = output / "artifacts"
    artifacts.mkdir()
    environment["VISION_STUDIO_QA_ARTIFACTS"] = str(artifacts)
    environment["VISION_STUDIO_GPU_TESTS"] = "1" if args.with_gpu else "0"
    environment["VISION_STUDIO_GPU_RUNTIME_DIR"] = str(
        args.gpu_runtime.resolve() if args.with_gpu else output / "isolated/absent-gpu-runtime")
    if args.python is not None:
        # venv/bin/python commonly points to the system interpreter. Preserve
        # the invoked venv path so Python can find the adjacent pyvenv.cfg.
        environment["VISION_STUDIO_PYTHON"] = str(args.python.absolute())
        environment["VISION_STUDIO_CONVERSION_PYTHON"] = str(args.python.absolute())
    errors: list[str] = []
    rows: list[dict] = []
    try:
        listing = subprocess.run(["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                                 text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=environment, check=True)
        discovered = json.loads(listing.stdout)
        (output / "ctest-discovery.json").write_text(listing.stdout, encoding="utf-8")
        names = {test["name"] for test in discovered["tests"]}
        missing = sorted(set(tests) - names)
        if missing:
            raise RuntimeError("Required selected tests are not configured: " + ", ".join(missing))
        for name in tests:
            command = ["ctest", "--test-dir", str(build), "--verbose", "--no-tests=error",
                       "--tests-regex", "^" + re.escape(name) + "$", "--output-junit",
                       str(output / (name + ".xml"))]
            print("Running " + name, flush=True)
            code, text, seconds = execute(command, environment=environment, timeout=args.timeout)
            log = output / (name + ".log")
            log.write_text(text, encoding="utf-8")
            junit = output / (name + ".xml")
            totals, skipped, case_count, failures = inspect_output(
                text, junit.read_text(encoding="utf-8") if junit.is_file() else None)
            passed = code == 0 and not skipped and not failures and case_count == 1
            feature_evidence = []
            for line in text.splitlines():
                for prefix in ("detection_scope_evidence:", "CLASS_FILTER_EVIDENCE", "ORT_PROFILE_EVIDENCE"):
                    if prefix in line:
                        value = json.loads(line.split(prefix, 1)[1].strip())
                        feature_evidence.append({"kind": prefix.rstrip(":"), "value": value})
            row = {"name": name, "command": command, "exit_code": code,
                   "elapsed_seconds": seconds, "success": passed, "skipped": skipped,
                   "qt_totals": totals, "log": log.name, "log_sha256": sha256(log),
                   "junit": junit.name if junit.is_file() else None,
                   "feature_evidence": feature_evidence}
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("name", "success", "elapsed_seconds")}), flush=True)
            if not passed:
                errors.append(name + ": command failure, skip, timeout, or missing JUnit test result")
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError, ET.ParseError) as error:
        errors.append(str(error))
    binary = (args.binary or build / "bin/vision-studio").resolve()
    cmake = (project / "CMakeLists.txt").read_text(encoding="utf-8")
    version = re.search(r'set\(VISION_STUDIO_APP_VERSION\s+"([^"]+)"', cmake)
    report = {
        "schema_version": 1, "version": version.group(1) if version else None,
        "success": not errors and len(rows) == len(tests), "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual selected CTest regression execution; not DEB, install, source or release acceptance",
        "selected_tests": tests, "real_cuda_requested": args.with_gpu,
        "binary": str(binary), "binary_sha256": sha256(binary) if binary.is_file() else None,
        "build_directory": str(build), "runs": rows,
        "artifacts_directory": str(artifacts),
        "coverage_note": "Feature coverage is defined by the actual assertions in the selected suites. "
                         "A successful suite is not a measurement of performance or proof of unexecuted cases.",
    }
    (output / "detection-regression-qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "errors": errors,
                      "report": str(output / "detection-regression-qa.json")}, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
