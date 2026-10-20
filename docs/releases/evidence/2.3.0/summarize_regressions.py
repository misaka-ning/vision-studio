#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate actual 2.3.0 regression logs without running or inventing tests.

The baseline is 17 CTest suites, six affected suites are replaced by their full
production replay, and two final direct Qt runs replace the old UI suite and
add workbench integration. All original failures remain archived. A passing
summary requires every effective suite, process exit status, and parsed result
to pass with zero skips. Final package acceptance is a separate gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

VERSION = "2.3.0"
BASE_SUITES = (
    "vision-core-tests", "vision-inference-delivery-tests", "vision-recording-tests",
    "vision-pt-tests", "vision-gpu-tests", "vision-ui-tests", "vision-batch-responsiveness-tests",
    "vision-scope-filter-ui-tests", "vision-modelviewer-tests", "vision-conversion-ui-tests",
    "vision-conversion-inference-tests", "vision-model-conversion-tests", "vision-netron-server-tests",
    "vision-gpu-runtime-tests", "vision-evaluation-worker-tests", "vision-evaluation-panel-tests",
    "vision-evaluation-python-tests",
)
REPLAY_SUITES = {
    "vision-evaluation-worker-tests", "vision-evaluation-panel-tests", "vision-batch-responsiveness-tests",
    "vision-scope-filter-ui-tests", "vision-conversion-inference-tests", "vision-gpu-tests",
}
FINAL_QT = {"vision-ui-tests": "ui-tests.log", "vision-evaluation-integration-tests": "integration.log"}
FOREIGN = {"docs/足球模型微调.md", "scripts/finetune_ball.py", "scripts/evaluate_ball.py", "scripts/run_finetune_ball.sh"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".cache"}
QT_TOTALS = re.compile(r"^Totals:\s+(\d+) passed,\s+(\d+) failed,\s+(\d+) skipped,\s+(\d+) blacklisted,\s+(\d+)ms\s*$", re.MULTILINE)
PY_TOTALS = re.compile(r"^Ran (\d+) tests? in ([0-9.]+)s\s*$", re.MULTILINE)


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), f"Missing regular JSON evidence: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(path, parent):
    return Path(path).resolve().relative_to(Path(parent).resolve()).as_posix()


def regular_inside(root, name):
    require(isinstance(name, str), "Evidence filename must be a string")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe evidence path: {name}")
    full = Path(root) / path
    require(full.is_file() and not full.is_symlink(), f"Missing regular evidence: {full}")
    require(full.resolve().is_relative_to(Path(root).resolve()), f"Evidence path escapes its run: {full}")
    return full


def normalize_ctest(text):
    # Verbose CTest prefixes child output with e.g. "17: "; direct Qt output
    # has no prefix. Remove only that transport prefix, never edit assertions.
    return re.sub(r"^\s*\d+: ?", "", text, flags=re.MULTILINE)


def parse_actual_totals(text, name):
    text = normalize_ctest(text)
    qt = list(QT_TOTALS.finditer(text))
    python = list(PY_TOTALS.finditer(text))
    require(bool(qt) != bool(python), f"{name}: require one unambiguous Qt or Python suite summary")
    require(not re.search(r"^(?:FAIL!|SKIP\s*:\s|QFATAL\s*:\s|XPASS\s*:\s|XFAIL\s*:\s|FAILED\s*\(|ERROR:\s)", text, re.MULTILINE),
            f"{name}: failure, skipped, fatal or expected-failure assertion appears in effective log")
    if qt:
        require(len(qt) == 1, f"{name}: expected exactly one complete Qt suite")
        passed, failed, skipped, blacklisted, milliseconds = map(int, qt[0].groups())
        starts = re.findall(r"^\*+ Start testing of ([^\n]+?) \*+\s*$", text, re.MULTILINE)
        finishes = re.findall(r"^\*+ Finished testing of ([^\n]+?) \*+\s*$", text, re.MULTILINE)
        require(len(starts) == len(finishes) == 1 and starts == finishes, f"{name}: missing/mismatched complete Qt start/finish markers")
        require(passed > 0 and not (failed or skipped or blacklisted), f"{name}: Qt totals are not zero-failure/skip")
        return {"qt_totals": [{"passed": passed, "failed": failed, "skipped": skipped,
                                "blacklisted": blacklisted}], "qt_suite": starts[0], "python_tests": 0,
                "result_elapsed_seconds": milliseconds / 1000.}
    require(len(python) == 1 and int(python[0].group(1)) > 0, f"{name}: incomplete or ambiguous Python summary")
    require(len(re.findall(r"^OK\s*$", text, re.MULTILINE)) == 1,
            f"{name}: Python suite did not finish with plain OK (skips/expected failures are not accepted)")
    require(not re.search(r"^OK\s*\([^\n]+\)", text, re.MULTILINE), f"{name}: non-passing Python outcomes")
    return {"qt_totals": [], "python_tests": int(python[0].group(1)),
            "result_elapsed_seconds": float(python[0].group(2))}


def validate_junit(path, name):
    root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    require(len(suites) == 1, f"{name}: expected one CTest JUnit suite")
    suite = suites[0]
    require(int(suite.get("tests", "0")) == 1, f"{name}: CTest did not execute exactly the selected suite")
    for key in ("failures", "errors", "skipped", "disabled"):
        require(int(suite.get(key, "0")) == 0, f"{name}: JUnit {key} is nonzero")
    tests = suite.findall("testcase")
    require(len(tests) == 1 and tests[0].get("name") == name and tests[0].get("status", "run") == "run",
            f"{name}: JUnit testcase mismatch/not-run")
    require(not suite.findall(".//failure") and not suite.findall(".//error") and not suite.findall(".//skipped"),
            f"{name}: JUnit contains a failure/error/skip element")


def indexed_runs(report, expected):
    runs = report.get("runs")
    require(isinstance(runs, list), "Run report lacks runs list")
    result = {run.get("name"): run for run in runs}
    require(len(result) == len(runs) and set(result) == set(expected), "Run metadata contains duplicate, missing or unexpected suites")
    return result


def inspect_ctest(folder, run, evidence):
    name = run["name"]
    require(run.get("exit_code") == 0 and run.get("success") is True and run.get("skipped") is False,
            f"{name}: runner recorded command failure or skipped result")
    log = regular_inside(folder, run.get("log"))
    xml = regular_inside(folder, run.get("junit"))
    require(sha(log) == run.get("log_sha256"), f"{name}: original runner/log SHA256 mismatch")
    totals = parse_actual_totals(log.read_text(encoding="utf-8"), name)
    validate_junit(xml, name)
    require("100% tests passed, 0 tests failed out of 1" in log.read_text(encoding="utf-8"),
            f"{name}: CTest process did not report selected-suite success")
    require(run.get("qt_totals") == totals["qt_totals"], f"{name}: parsed Qt totals disagree with runner metadata")
    archived_log = evidence / folder.name / log.relative_to(folder)
    archived_xml = evidence / folder.name / xml.relative_to(folder)
    require(sha(archived_log) == sha(log) and sha(archived_xml) == sha(xml), f"{name}: archived evidence differs")
    return {"name": name, "success": True, "exit_code": 0, "skipped": False,
            "command": run.get("command"), "source_run": folder.name,
            "log": relative(archived_log, evidence), "log_sha256": sha(log),
            "junit": relative(archived_xml, evidence), "junit_sha256": sha(xml), **totals,
            "feature_evidence": run.get("feature_evidence", [])}


def inspect_direct(folder, run, name, expected_log, build, evidence):
    require(run.get("exit_code") == 0, f"{name}: final direct-binary exit status is not zero")
    require(run.get("name") == name and run.get("log") == expected_log, f"{name}: final harness name/log mismatch")
    log = regular_inside(folder, expected_log)
    require(sha(log) == run.get("log_sha256"), f"{name}: direct harness/log SHA256 mismatch")
    binary = Path(run.get("binary", run.get("executable", "")))
    require(binary.is_absolute() and binary.is_file() and binary.resolve().is_relative_to(build), f"{name}: final harness binary is outside this production build")
    require(sha(binary) == run.get("binary_sha256", run.get("executable_sha256")), f"{name}: direct harness binary changed after execution")
    command = run.get("command")
    require(isinstance(command, list) and command and Path(command[0]).resolve() == binary.resolve(), f"{name}: harness command does not identify the tested binary")
    totals = parse_actual_totals(log.read_text(encoding="utf-8"), name)
    require(totals["qt_totals"], f"{name}: expected native Qt suite")
    archived = evidence / folder.name / expected_log
    require(sha(archived) == sha(log), f"{name}: archived final Qt log differs")
    return {"name": name, "success": True, "exit_code": run["exit_code"], "skipped": False,
            "source_run": folder.name, "execution_kind": "direct Qt binary with captured process exit status",
            "command": command, "binary": str(binary), "binary_sha256": sha(binary),
            "log": relative(archived, evidence), "log_sha256": sha(log), "junit": None, **totals}


def copy_exact(source, destination):
    require(source.is_file() and not source.is_symlink(), f"Not regular raw evidence: {source}")
    checksum = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.is_symlink(), f"Refusing evidence symlink: {destination}")
    if destination.exists():
        require(destination.is_file() and sha(destination) == checksum, f"Refusing to replace different original evidence: {destination}")
    else:
        with source.open("rb") as original, destination.open("xb") as saved:
            shutil.copyfileobj(original, saved, 1024 * 1024)
    require(sha(source) == checksum == sha(destination), f"Evidence changed during archival: {source}")
    return {"sha256": checksum, "bytes": source.stat().st_size}


def archive_originals(qa, evidence, include_temp):
    copied, omitted = {}, []
    for path in sorted(qa.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".log", ".xml", ".json", ".stdout", ".stderr"}:
            continue
        rel = path.relative_to(qa)
        if SKIP_DIRS.intersection(rel.parts):
            continue
        if path.name.startswith("onnx-cuda") and rel.parts[0] != "production-replay":
            omitted.append({"path": rel.as_posix(), "sha256": sha(path),
                            "reason": "Superseded CUDA trace; raw runner report retained, effective production-replay traces archived"})
            continue
        copied[rel.as_posix()] = copy_exact(path, evidence / rel)
    if include_temp:
        paths = set(Path("/tmp").glob("vision230-*.log"))
        paths.update(Path("/tmp") / name for name in (
            "vision-evaluation-python-tests.log", "vision-evaluation-worker-tests-230-initial.log",
            "vision-evaluation-worker-tests-230.log", "vision-windows-release-tool-tests-230.log"))
        for path in sorted(paths):
            if not path.is_file():
                continue
            filename = path.stem + "-" + sha(path)[:12] + path.suffix
            rel = Path("preliminary") / filename
            copied[rel.as_posix()] = copy_exact(path, evidence / rel)
    return copied, omitted


def code_hashes(project):
    paths = set()
    for directory in ("src", "tests", "scripts", "assets"):
        for path in (project / directory).rglob("*"):
            rel = path.relative_to(project)
            if path.is_file() and not path.is_symlink() and not SKIP_DIRS.intersection(rel.parts) and rel.as_posix() not in FOREIGN:
                paths.add(path)
    for name in ("CMakeLists.txt", "resources.qrc", "run.sh", "VisionStudio.desktop", "runtime-info.json"):
        path = project / name
        if path.is_file(): paths.add(path)
    paths.update(p for p in project.glob("requirements*.txt") if p.is_file())
    paths.update(p for p in (project / "packaging").glob("*.py") if p.is_file())
    paths.update(p for p in (project / "packaging/windows").glob("*.py") if p.is_file())
    return {relative(p, project): sha(p) for p in sorted(paths)}


def cuda_profiles(folder, evidence):
    files = sorted((folder / "artifacts").glob("onnx-cuda*.json"))
    require(len(files) == 2, "Expected two actual effective production CUDA traces")
    result = []
    for path in files:
        events = read_json(path)
        require(isinstance(events, list), f"CUDA trace is not an event array: {path}")
        nodes = [event for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider"]
        require(nodes, f"No actual CUDAExecutionProvider nodes in trace: {path}")
        saved = evidence / folder.name / path.relative_to(folder)
        require(sha(saved) == sha(path), "Archived CUDA trace differs")
        result.append({"file": relative(saved, evidence), "sha256": sha(path),
                       "cuda_node_events": len(nodes), "distinct_cuda_nodes": len({x.get("name") for x in nodes})})
    return result


def aggregate(project, qa, evidence, build, include_temp):
    original, omitted = archive_originals(qa, evidence, include_temp)
    baseline = read_json(qa / "regression-01/detection-regression-qa.json")
    replay = read_json(qa / "production-replay/detection-regression-qa.json")
    require(baseline.get("version") == replay.get("version") == VERSION, "Run metadata version differs")
    base = indexed_runs(baseline, BASE_SUITES)
    replacement = indexed_runs(replay, REPLAY_SUITES)
    final_meta = read_json(qa / "final-ui/run-qa.json")
    final_runs = {r.get("name"): r for r in final_meta.get("runs", []) if r.get("name") in FINAL_QT}
    require(set(final_runs) == set(FINAL_QT), "Final direct Qt harness lacks both complete suites")
    production = build / "bin/vision-studio"
    require(production.is_file(), "Final production application is missing")
    production_sha = sha(production)
    require(final_meta.get("application_binary_sha256") == production_sha,
            "Final direct Qt harness is not associated with the current production application")
    rows = []
    for name in BASE_SUITES:
        if name in FINAL_QT:
            rows.append(inspect_direct(qa / "final-ui", final_runs[name], name, FINAL_QT[name], build, evidence))
        elif name in replacement:
            rows.append(inspect_ctest(qa / "production-replay", replacement[name], evidence))
        else:
            rows.append(inspect_ctest(qa / "regression-01", base[name], evidence))
    added = "vision-evaluation-integration-tests"
    rows.append(inspect_direct(qa / "final-ui", final_runs[added], added, FINAL_QT[added], build, evidence))
    require(len(rows) == 18 and len({r["name"] for r in rows}) == 18, "Effective 18-suite regression is incomplete")
    profiles = cuda_profiles(qa / "production-replay", evidence)
    cmake = (project / "CMakeLists.txt").read_text(encoding="utf-8")
    require(re.search(r"project\(VisionStudio\s+VERSION\s+2\.3\.0\b", cmake) and
            re.search(r'set\(VISION_STUDIO_APP_VERSION\s+"2\.3\.0"\)', cmake), "Current CMake version is not 2.3.0")
    hashes = code_hashes(project)
    test_binaries = {}
    for row in rows:
        for path in (build / row["name"], build / "bin" / row["name"]):
            if path.is_file():
                test_binaries[row["name"]] = sha(path)
                break
    hardware = sorted({entry.get("value", {}).get("actual_cuda_device") for entry in replacement["vision-gpu-tests"].get("feature_evidence", [])
                       if entry.get("value", {}).get("actual_cuda_device")})
    require(hardware, "Actual GPU suite did not record its CUDA device")
    passed_qt = sum(total["passed"] for row in rows for total in row["qt_totals"])
    passed_python = sum(row["python_tests"] for row in rows)
    require(passed_qt > 0 and passed_python > 0, "Missing effective Qt or Python regression results")
    require(sha(production) == production_sha and code_hashes(project) == hashes, "Source or production binary changed during aggregation")
    return {"schema_version": 1, "version": VERSION, "success": True, "errors": [],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Actual effective 18-suite regression: 17-suite baseline, six complete affected-suite production replays, and final full direct Qt UI/integration runs. Final DEB/install/source acceptance is separate.",
            "effective_suite_count": len(rows), "qt_tests_passed": passed_qt, "python_tests_passed": passed_python,
            "failed": 0, "skipped": 0, "blacklisted": 0, "suites": rows,
            "production_binary_sha256": production_sha, "production_binary": str(production),
            "baseline_binary_sha256": baseline.get("binary_sha256"), "replay_binary_sha256": replay.get("binary_sha256"),
            "source_code_input_sha256": hashes, "current_test_binary_sha256": test_binaries,
            "source_hash_scope": "Current final source/static inputs at aggregation; earlier baseline/replay execution identities remain in their original reports. Backend code remained fixed after production replay; final UI suites cover later UI changes.",
            "cuda_profiles": profiles, "hardware": " / ".join(hardware),
            "original_evidence_files": original, "omitted_superseded_cuda_traces": omitted,
            "superseded_runs": [
                {"path": "regression-01/detection-regression-qa.json", "reason": "Initial real UI layout assertion failed. Original failed log/XML/report retained; only the final complete UI run replaces that result. Six selected suites have complete production replays."},
                {"path": "integration-01/raw.log", "reason": "Initial dataset file-picker automation did not wait for the filesystem model; failure retained. Later complete integration runs replace it."},
                {"path": "production-ui-final/legacy-ui.log", "reason": "Sandbox loopback/Netron access failure is not a pass. Original failure retained; final UI executes with the required local-loopback permissions."},
                {"path": "preliminary", "reason": "Earlier diagnostics, configure/build logs and incomplete/subset trials are retained as history, never counted toward effective success."},
            ],
            "limitations": ["No Windows 2.3.0 build or installation acceptance", "No physical camera acceptance",
                            "Synthetic/derived labels validate evaluation mechanics, not general model accuracy",
                            "Local stop/timing observations are not performance guarantees",
                            "Not a single one-shot CTest pass: effective suites are explicitly selected from retained actual runs"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = Path(__file__).resolve().parents[4]
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--qa-root", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--include-temp-logs", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    qa = (args.qa_root or project / f"output/qa-{VERSION}").resolve()
    build = (args.build or project / f"output/build/{VERSION}-release").resolve()
    evidence = (args.evidence or project / f"docs/releases/evidence/{VERSION}").resolve()
    report_path = (args.report or evidence / "ctest-qa.json").absolute()
    if report_path.exists() or report_path.is_symlink():
        parser.error("Report already exists; refusing to overwrite evidence")
    if not qa.is_dir() or not build.is_dir() or evidence == qa or evidence.is_relative_to(qa):
        parser.error("QA/build directories must exist and evidence must be outside the raw QA tree")
    report = {"schema_version": 1, "version": VERSION, "success": False, "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv]}
    try:
        report = aggregate(project, qa, evidence, build, args.include_temp_logs)
        report["command"] = [sys.executable, *sys.argv]
        report["summarizer_sha256"] = sha(Path(__file__).resolve())
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, ET.ParseError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"success": report["success"], "errors": report["errors"], "report": str(report_path),
                      "suites": report.get("effective_suite_count"), "qt_passed": report.get("qt_tests_passed"),
                      "python_passed": report.get("python_tests_passed")}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
