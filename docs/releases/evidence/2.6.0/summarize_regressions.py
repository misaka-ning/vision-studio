#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify and archive one actual complete 2.6.0 CTest runner execution.

Snapshots record production/static inputs before tests. Aggregation checks the
same current source, application and native test binaries, all selected raw
logs, JUnit, actual result totals and CUDA profiles. It never executes inference
or manufactures a passing test. Final package/source acceptance is separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

VERSION = "2.6.0"
SUITES = (
    "vision-core-tests", "vision-inference-delivery-tests", "vision-recording-tests",
    "vision-pt-tests", "vision-gpu-tests", "vision-ui-tests", "vision-batch-responsiveness-tests",
    "vision-scope-filter-ui-tests", "vision-modelviewer-tests", "vision-conversion-ui-tests",
    "vision-conversion-inference-tests", "vision-model-conversion-tests", "vision-netron-server-tests",
    "vision-gpu-runtime-tests", "vision-evaluation-worker-tests", "vision-evaluation-panel-tests",
    "vision-evaluation-python-tests", "vision-evaluation-integration-tests",
    "vision-dataset-session-tests", "vision-dataset-session-ui-tests",
)
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


def code_hashes(project):
    paths = set()
    for directory in ("src", "tests", "scripts", "assets", "vendor", "models"):
        for path in (project / directory).rglob("*"):
            rel = path.relative_to(project)
            if path.is_file() and not path.is_symlink() and not SKIP_DIRS.intersection(rel.parts) and rel.as_posix() not in FOREIGN:
                paths.add(path)
    for name in ("CMakeLists.txt", "resources.qrc", "run.sh", "VisionStudio.desktop", "runtime-info.json"):
        path = project / name
        if path.is_file():
            paths.add(path)
    paths.update(p for p in project.glob("requirements*.txt") if p.is_file())
    for directory in ("packaging", "packaging/windows"):
        paths.update(p for p in (project / directory).glob("*") if p.is_file() and p.suffix in {".py", ".sh"})
    return {relative(p, project): sha(p) for p in sorted(paths)}


def snapshot(project, binary):
    require(binary.is_file() and not binary.is_symlink(), "Production application is not a regular binary")
    build = binary.parent.parent
    require(binary.parent.name == "bin" and (build / "CMakeCache.txt").is_file(), "Require a configured production build/bin application")
    cmake = (project / "CMakeLists.txt").read_text(encoding="utf-8")
    require(re.search(r"project\(VisionStudio\s+VERSION\s+2\.6\.0\b", cmake) and
            re.search(r'set\(VISION_STUDIO_APP_VERSION\s+"2\.6\.0"\)', cmake), "CMake version is not 2.6.0")
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    require(re.search(r"(?m)^CMAKE_BUILD_TYPE:STRING=Release$", cache) and
            re.search(r"(?m)^VISION_RESOURCE_FALLBACK:[^=]+=/opt/VisionStudio$", cache), "Build is not production Release /opt/VisionStudio")
    configured_source = re.search(r"(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache)
    require(configured_source is not None and Path(configured_source.group(1)).resolve() == project, "Production build was configured from another source tree")
    native = {}
    for name in SUITES:
        matches = [path for path in (build / name, build / "bin" / name) if path.is_file()]
        if matches:
            require(len(matches) == 1, "Ambiguous native test binary: " + name)
            native[name] = {"path": str(matches[0]), "sha256": sha(matches[0])}
    require(native, "No native regression test binaries exist")
    result = {"schema_version": 1, "version": VERSION,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "production_binary": str(binary), "production_binary_sha256": sha(binary),
              "build_directory": str(build), "cmake_cache_sha256": sha(build / "CMakeCache.txt"),
              "source_code_input_sha256": code_hashes(project),
              "current_test_binary_sha256": native,
              "source_hash_scope": "src/tests/scripts/assets/vendor/models, CMake/resources/launchers/runtime-info, root dependency locks and packaging Python/shell inputs; caches and four unrelated user files excluded. Documentation/evidence changes do not alter production input identity."}
    require(sha(binary) == result["production_binary_sha256"] and code_hashes(project) == result["source_code_input_sha256"], "Production inputs changed during snapshot")
    return result


def archive_originals(qa, evidence):
    copied = {}
    for path in sorted(qa.iterdir()):
        if path.is_file() and path.suffix in {".log", ".xml", ".json", ".stdout", ".stderr"}:
            rel = Path(qa.name) / path.name
            copied[rel.as_posix()] = copy_exact(path, evidence / rel)
    for path in sorted((qa / "artifacts").glob("*.json")):
        rel = Path(qa.name) / "artifacts" / path.name
        copied[rel.as_posix()] = copy_exact(path, evidence / rel)
    return copied


def cuda_profiles(folder, evidence):
    files = sorted((folder / "artifacts").glob("onnx-cuda*.json"))
    require(len(files) == 2, "Expected two actual CUDA traces from the complete GPU suite")
    result = []
    for path in files:
        events = read_json(path)
        require(isinstance(events, list), f"CUDA trace is not an event array: {path}")
        nodes = [event for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider"]
        require(nodes, f"No actual CUDAExecutionProvider nodes: {path}")
        saved = evidence / folder.name / path.relative_to(folder)
        require(sha(saved) == sha(path), "Archived CUDA trace differs")
        result.append({"file": relative(saved, evidence), "sha256": sha(path),
                       "cuda_node_events": len(nodes), "distinct_cuda_nodes": len({x.get("name") for x in nodes})})
    return result


def aggregate(project, qa, evidence, binary, before_path):
    before = read_json(before_path)
    require(before.get("version") == VERSION, "Before-run snapshot belongs to another version")
    current = snapshot(project, binary)
    fields = ("production_binary", "production_binary_sha256", "build_directory", "cmake_cache_sha256", "source_code_input_sha256", "current_test_binary_sha256")
    require(all(before.get(name) == current[name] for name in fields), "Production inputs or binaries changed since before-run snapshot")
    raw = read_json(qa / "detection-regression-qa.json")
    require(raw.get("version") == VERSION and raw.get("success") is True and raw.get("errors") == [] and
            raw.get("real_cuda_requested") is True, "Complete actual runner did not pass with real CUDA")
    require(set(raw.get("selected_tests", [])) == set(SUITES) and len(raw.get("selected_tests", [])) == len(SUITES), "Runner did not select exactly the complete 20 suites")
    discovery = read_json(qa / "ctest-discovery.json")
    discovered = [test["name"] for test in discovery.get("tests", [])]
    require(len(discovered) == len(set(discovered)) == len(SUITES) and set(discovered) == set(SUITES),
            "Fresh CTest discovery and complete 20-suite selection disagree")
    require(raw.get("binary_sha256") == current["production_binary_sha256"] and
            Path(raw.get("binary", "")).resolve() == binary and
            Path(raw.get("build_directory", "")).resolve() == Path(current["build_directory"]), "Runner does not bind current production application/build")
    began = datetime.fromisoformat(before["generated_at_utc"])
    ended = datetime.fromisoformat(raw["generated_at_utc"])
    require(began <= ended, "Before-run snapshot was created after runner finished")
    runs = indexed_runs(raw, SUITES)
    original = archive_originals(qa, evidence)
    before_saved = evidence / "production-inputs-before.json"
    original[before_saved.name] = copy_exact(before_path, before_saved)
    rows = [inspect_ctest(qa, runs[name], evidence) for name in SUITES]
    profiles = cuda_profiles(qa, evidence)
    hardware = sorted({entry.get("value", {}).get("actual_cuda_device") for entry in runs["vision-gpu-tests"].get("feature_evidence", [])
                       if entry.get("value", {}).get("actual_cuda_device")})
    require(hardware, "Actual GPU suite did not record its CUDA device")
    for name, row in runs.items():
        command = row.get("command")
        require(isinstance(command, list) and "--test-dir" in command and
                Path(command[command.index("--test-dir") + 1]).resolve() == Path(current["build_directory"]), "Suite command used another build: " + name)
    passed_qt = sum(total["passed"] for row in rows for total in row["qt_totals"])
    passed_python = sum(row["python_tests"] for row in rows)
    require(passed_qt > 0 and passed_python > 0, "Missing Qt or Python regression results")
    final = snapshot(project, binary)
    require(all(final[name] == current[name] for name in fields), "Production inputs changed during aggregation")
    return {"schema_version": 1, "version": VERSION, "success": True, "errors": [],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "One complete actual 20-suite CTest runner execution with real CUDA. Raw logs, JUnit and totals independently verified; final DEB/install/source acceptance is separate.",
            "effective_suite_count": len(rows), "qt_tests_passed": passed_qt, "python_tests_passed": passed_python,
            "failed": 0, "skipped": 0, "blacklisted": 0, "suites": rows,
            **{name: current[name] for name in fields}, "source_hash_scope": current["source_hash_scope"],
            "before_run_snapshot": before_saved.name, "before_run_snapshot_sha256": sha(before_saved),
            "cuda_profiles": profiles, "hardware": " / ".join(hardware), "original_evidence_files": original,
            "limitations": ["No Windows 2.6.0 build or installation acceptance", "No physical camera acceptance",
                            "Synthetic/derived labels validate evaluation mechanics, not general model accuracy",
                            "Local stop/timing observations are not performance guarantees"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[4]
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--before-snapshot", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    binary = (args.binary or project / f"output/build/{VERSION}-release/bin/vision-studio").resolve()
    qa = (args.run_dir or project / f"output/v{VERSION}/qa-full").resolve()
    evidence = (args.evidence or project / f"docs/releases/evidence/{VERSION}").resolve()
    before = (args.before_snapshot or qa.parent / "production-inputs-before.json").resolve()
    report_path = (args.report or (before if args.snapshot_only else evidence / "ctest-qa.json")).absolute()
    if report_path.exists() or report_path.is_symlink():
        parser.error("Report already exists; refusing to overwrite evidence")
    if not args.snapshot_only and (not qa.is_dir() or evidence == qa or evidence.is_relative_to(qa)):
        parser.error("Raw runner directory must exist and evidence must be outside it")
    report = {"schema_version": 1, "version": VERSION, "success": False, "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        report = snapshot(project, binary) if args.snapshot_only else aggregate(project, qa, evidence, binary, before)
        report["command"] = [sys.executable, *sys.argv]
        report["summarizer_sha256"] = sha(Path(__file__).resolve())
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, IndexError, ET.ParseError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    success = not report.get("errors") and (args.snapshot_only or report.get("success") is True)
    print(json.dumps({"success": success, "report": str(report_path), "errors": report.get("errors", []),
                      "suites": report.get("effective_suite_count"), "qt_passed": report.get("qt_tests_passed"),
                      "python_passed": report.get("python_tests_passed")}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
