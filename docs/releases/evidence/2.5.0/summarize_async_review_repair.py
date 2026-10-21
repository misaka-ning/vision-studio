#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Audit 17 original successes plus one CPU scope rerun after insertion-only waits.

This tool performs no builds, tests or inference. The complete original run must
retain 14 passing / 24 failing scope cases and exactly 17 other passing suites.
A later CPU-only runner must pass exactly the same 38 scope cases. Only the
approved 35 inserted lines in tests/scope_filter_ui_tests.cpp and that native
test binary may change. Original raw logs, JUnit and failed results are retained.
This is explicitly not one all-passing complete 18-suite execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import summarize_regressions as base

VERSION = "2.5.0"
SOURCE = "tests/scope_filter_ui_tests.cpp"
REPAIRED_SUITE = "vision-scope-filter-ui-tests"
CLASS = "ScopeFilterUiTests"
ORIGINAL_COMMIT = "e1c2f12d308c3435323a9586cd354a60cd7cfbc2"
ORIGINAL_SOURCE_SHA256 = "4e79907500dd3cc1e1f3c6c2002b2a180b1428e0964d3fdff4b9d52f9ace401d"
# Reviewed patch 3ca8133cd4583ecaccb9e940f180d018737eb6ec: asynchronous
# review waits and additional real-pixel assertions, with no removed old lines.
REPAIRED_SOURCE_SHA256 = "d21fed31ea28b0f9c0915cbcea5bb1fbb97843457fb510d7b830f0582fff5d5c"
ADDED_LINES = 35
IDENTITIES = ("production_binary", "production_binary_sha256", "build_directory", "cmake_cache_sha256")
FIELDS = (*IDENTITIES, "source_code_input_sha256", "current_test_binary_sha256")
CASE = rf"{CLASS}::[A-Za-z_]\w*\([^\n)]*\)"
RECORD = re.compile(rf"^(PASS|FAIL!)[ \t]*:[ \t]*({CASE})(?:[ \t]+([^\n]*))?[ \t]*$", re.MULTILINE)
FAILURE_LINES = {
    "fixedRangeAndProgress": {180}, "filtersAndScope": {221, 225},
    "validCacheOutsideRangeAndChangedParameters": {329},
    "stoppedRangePreservesCompletedResultsAndExport": {385},
    "submittedSourceIdentityIsNotReplacedByLaterDiskVersion": {521},
    "equivalentFiltersAndChangedSourceFile": {537},
}


def require(value, message):
    base.require(value, message)


def timestamp(value, label):
    result = datetime.fromisoformat(value["generated_at_utc"])
    require(result.tzinfo is not None, f"{label}: timezone-aware timestamp required")
    return result


def changed_entries(before, after, label):
    require(isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after),
            f"{label}: added or missing snapshot keys are forbidden")
    return [key for key in before if before[key] != after[key]]


def validate_transition(before, after, current):
    require(before.get("version") == after.get("version") == current.get("version") == VERSION,
            "All production snapshots must belong to 2.5.0")
    require(before.get("summarizer_sha256") == after.get("summarizer_sha256") == base.sha(Path(base.__file__).resolve()),
            "Shared snapshot generator changed across the input snapshots")
    require(all(before.get(key) == after.get(key) for key in IDENTITIES),
            "Production application, build location or CMake cache changed")
    source_changes = changed_entries(before.get("source_code_input_sha256"), after.get("source_code_input_sha256"), "Source inputs")
    native_changes = changed_entries(before.get("current_test_binary_sha256"), after.get("current_test_binary_sha256"), "Native binaries")
    require(source_changes == [SOURCE], f"Only the reviewed scope test source may change: {source_changes!r}")
    require(native_changes == [REPAIRED_SUITE], f"Only the scope test binary may change: {native_changes!r}")
    old = before["current_test_binary_sha256"][REPAIRED_SUITE]
    new = after["current_test_binary_sha256"][REPAIRED_SUITE]
    require(isinstance(old, dict) and isinstance(new, dict) and set(old) == set(new) == {"path", "sha256"} and
            old["path"] == new["path"] and old["sha256"] != new["sha256"],
            "Scope binary must change hash at the identical configured binary path")
    require(all(after.get(key) == current[key] for key in FIELDS),
            "Current production/test inputs no longer match the after-repair snapshot")
    return {"allowed_source_changes": source_changes, "allowed_native_binary_changes": native_changes,
        "source_transition": {SOURCE: {"before_sha256": before["source_code_input_sha256"][SOURCE],
                                      "after_sha256": after["source_code_input_sha256"][SOURCE]}},
        "native_binary_transition": {REPAIRED_SUITE: {"path": new["path"], "before_sha256": old["sha256"], "after_sha256": new["sha256"]}},
        "production_application_unchanged": True, "cmake_cache_unchanged": True,
        "other_source_inputs_unchanged": True, "other_native_test_binaries_unchanged": True,
        "snapshotted_runtime_metadata_and_dependency_locks_unchanged": True,
        "current_inputs_match_after_snapshot": True}


def write_exact_bytes(path, data):
    require(not path.is_symlink(), f"Refusing evidence symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"Refusing to rewrite different original evidence: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    require(path.read_bytes() == data, f"Evidence bytes changed while saving: {path}")
    return {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def validate_insert_only_source(project, before, after, evidence):
    old = subprocess.check_output(["git", "-C", str(project), "show", f"{ORIGINAL_COMMIT}:{SOURCE}"])
    path = project / SOURCE
    require(path.is_file() and not path.is_symlink(), "Current scope source must be a regular file")
    new = path.read_bytes()
    old_hash, new_hash = hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()
    require(old_hash == ORIGINAL_SOURCE_SHA256 == before["source_code_input_sha256"][SOURCE],
            "Original scope source differs from the fixed Git revision or before snapshot")
    require(new_hash == REPAIRED_SOURCE_SHA256 == after["source_code_input_sha256"][SOURCE],
            "Repaired scope source differs from the reviewed 35-line insertion patch or after snapshot")
    old_lines, new_lines = old.decode("utf-8").splitlines(keepends=True), new.decode("utf-8").splitlines(keepends=True)
    operations = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()
    require(all(op[0] in {"equal", "insert"} for op in operations), "Repair removed or replaced an original source line")
    inserted = [line for tag, _, _, j1, j2 in operations if tag == "insert" for line in new_lines[j1:j2]]
    require(len(inserted) == ADDED_LINES and len(new_lines) - len(old_lines) == ADDED_LINES,
            "Repair is not exactly 35 inserted lines")
    retained = [line for tag, _, _, j1, j2 in operations if tag == "equal" for line in new_lines[j1:j2]]
    require(retained == old_lines, "Original line sequence is not retained in order")
    require(not re.search(r"\b(?:QSKIP|QEXPECT_FAIL)\b", "".join(inserted)), "Inserted skip/expected-failure logic is forbidden")
    require(all(not line.lstrip().startswith(("#", "onnx.", "h.", "torch.", "np.")) for line in inserted),
            "Inserted fixture/preprocessor/model-generator changes are forbidden")
    # The exact reviewed after-source hash plus retained old lines restricts the
    # insertions to the reviewed wait helper and extra assertions, not an
    # arbitrary fixture rewrite or a newly inserted assertion bypass.
    diff = "".join(difflib.unified_diff(old_lines, new_lines,
        fromfile=f"a/{SOURCE}@{ORIGINAL_COMMIT}", tofile=f"b/{SOURCE}@reviewed-async-waits"))
    original_gzip = gzip.compress(old, mtime=0)
    require(gzip.decompress(original_gzip) == old, "Original source gzip round-trip failed")
    folder = evidence / "async-review-repair"
    saved_old = write_exact_bytes(folder / "original-scope_filter_ui_tests.cpp.gz", original_gzip)
    saved_diff = write_exact_bytes(folder / "scope_filter_ui_tests.diff", diff.encode("utf-8"))
    for row in (saved_old, saved_diff): row["file"] = base.relative(folder / row["file"], evidence)
    return {"original_git_commit": ORIGINAL_COMMIT, "source": SOURCE,
        "original_source_sha256": old_hash, "repaired_source_sha256": new_hash,
        "inserted_lines": len(inserted), "removed_lines": 0, "replaced_lines": 0,
        "sequence_matcher_operations": [list(op) for op in operations],
        "all_original_lines_retained_in_order": True, "all_original_assertion_lines_retained": True,
        "approved_wait_and_extra_assertion_patch_only": True, "model_fixture_changes_permitted": False,
        "skip_or_expected_failure_insertions": False, "original_source_gzip": saved_old, "reviewed_diff": saved_diff,
        "inserted_text_sha256": hashlib.sha256("".join(inserted).encode()).hexdigest(),
        "manual_review_boundary": "The exact reviewed patch hash and insertion-only diff retain every original line. Hash checks do not substitute for reviewer assessment of the approved wait helper."}, old_lines


def validate_runner(report, expected, current, label, *, passing, real_cuda):
    require(report.get("version") == VERSION and report.get("real_cuda_requested") is real_cuda,
            f"{label}: wrong version or CUDA-request setting")
    selected = report.get("selected_tests", [])
    require(len(selected) == len(expected) and set(selected) == set(expected), f"{label}: unexpected selected suites")
    require(report.get("binary_sha256") == current["production_binary_sha256"] and
            Path(report.get("binary", "")).resolve() == Path(current["production_binary"]) and
            Path(report.get("build_directory", "")).resolve() == Path(current["build_directory"]),
            f"{label}: runner does not bind the unchanged production application/build")
    if passing:
        require(report.get("success") is True and report.get("errors") == [], f"{label}: scope rerun did not pass")
    else:
        require(report.get("success") is False and report.get("errors") ==
                [REPAIRED_SUITE + ": command failure, skip, timeout, or missing JUnit test result"],
                f"{label}: original runner must retain exactly the known scope failure")
    runs = base.indexed_runs(report, expected)
    for name, run in runs.items():
        command = run.get("command")
        require(isinstance(command, list) and "--test-dir" in command and "--tests-regex" in command,
                f"{label}/{name}: missing exact CTest command")
        require(Path(command[command.index("--test-dir") + 1]).resolve() == Path(current["build_directory"]) and
                command[command.index("--tests-regex") + 1] == "^" + re.escape(name) + "$",
                f"{label}/{name}: command used another build or suite selection")
    return runs


def parse_original_scope(text, label, old_source_lines):
    text = base.normalize_ctest(text)
    summaries = list(base.QT_TOTALS.finditer(text))
    require(len(summaries) == 1, f"{label}: ambiguous/missing original Qt totals")
    passed, failed, skipped, blacklisted, milliseconds = map(int, summaries[0].groups())
    require((passed, failed, skipped, blacklisted) == (14, 24, 0, 0),
            f"{label}: original scope must retain exactly 14 pass / 24 fail / 0 skip / 0 blacklist")
    require(not re.search(r"^(?:SKIP\s*:|QFATAL\s*:|XPASS\s*:|XFAIL\s*:)", text, re.MULTILINE),
            f"{label}: original scope has skip/fatal/expected-failure output")
    for marker in ("Start", "Finished"):
        require(re.findall(rf"^\*+ {marker} testing of ([^\n]+?) \*+\s*$", text, re.MULTILINE) == [CLASS],
                f"{label}: incomplete/wrong original Qt suite")
    records = list(RECORD.finditer(text))
    require(len(records) == passed + failed == 38 and len({m[2] for m in records}) == 38,
            f"{label}: missing/duplicate original case identities")
    require(len(re.findall(r"^FAIL!\s*:", text, re.MULTILINE)) == failed and
            len(re.findall(r"^PASS\s*:", text, re.MULTILINE)) == passed,
            f"{label}: original log contains foreign or unparsed test results")
    failures, passes, locations = [], [], []
    for record in records:
        kind, case = record[1], record[2]
        if kind == "PASS": passes.append(case); continue
        failures.append(case)
        following = re.search(r"^(?:PASS|FAIL!|QINFO|QWARN|QDEBUG|QCRITICAL|QFATAL|QSYSTEM)\s*:", text[record.end():], re.MULTILINE)
        end = record.end() + following.start() if following else len(text)
        block = text[record.start():end]
        loc = re.findall(r"^\s*Loc:\s*\[([^\n]+)\((\d+)\)\]\s*$", block, re.MULTILINE)
        require(len(loc) == 1 and loc[0][0].endswith("/" + SOURCE), f"{label}: each original FAIL must come from the scope test source")
        method = case.split("::", 1)[1].split("(", 1)[0]; line = int(loc[0][1])
        require(line in FAILURE_LINES.get(method, set()) and 1 <= line <= len(old_source_lines),
                f"{label}: failure is outside the reviewed asynchronous-review assertion locations: {case}/{line}")
        source_line = old_source_lines[line - 1].strip()
        require(re.search(r"\b(?:QCOMPARE|QVERIFY|QVERIFY2)\s*\(", source_line),
                f"{label}: failure location is not an original retained assertion")
        locations.append({"case": case, "source": SOURCE, "line": line, "original_assertion": source_line})
    return {"qt_totals": [{"passed": passed, "failed": failed, "skipped": skipped, "blacklisted": blacklisted}],
            "case_ids_in_execution_order": [m[2] for m in records], "failed_case_ids": failures,
            "passed_case_ids": passes, "original_failure_locations": locations,
            "result_elapsed_seconds": milliseconds / 1000.}


def inspect_failed_original(folder, run, evidence, old_source_lines):
    require(run.get("name") == REPAIRED_SUITE and run.get("success") is False and
            run.get("exit_code") == 8 and run.get("skipped") is False, "Original scope failure metadata differs")
    log = base.regular_inside(folder, run.get("log")); xml = base.regular_inside(folder, run.get("junit"))
    require(base.sha(log) == run.get("log_sha256"), "Original failed log no longer matches runner SHA256")
    text = log.read_text(encoding="utf-8"); parsed = parse_original_scope(text, "Original raw log", old_source_lines)
    require(run.get("qt_totals") == parsed["qt_totals"] and "0% tests passed, 1 tests failed out of 1" in text,
            "Original failed runner totals/CTest failure footer differ")
    root = ET.fromstring(xml.read_text(encoding="utf-8"))
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    require(len(suites) == 1, "Original failed JUnit must contain one CTest suite")
    suite = suites[0]
    require(int(suite.get("tests", "0")) == 1 and int(suite.get("failures", "0")) == 1 and
            all(int(suite.get(k, "0")) == 0 for k in ("errors", "skipped", "disabled")), "Original JUnit failure totals differ")
    cases = suite.findall("testcase")
    require(len(cases) == 1 and cases[0].get("name") == REPAIRED_SUITE and cases[0].get("status") == "fail" and
            cases[0].find("failure") is not None and not suite.findall(".//error") and not suite.findall(".//skipped"),
            "Original failed JUnit testcase differs")
    require(parse_original_scope(cases[0].findtext("system-out", ""), "Original JUnit output", old_source_lines) == parsed,
            "Original JUnit case/failure details disagree with the retained raw log")
    saved_log, saved_xml = evidence / folder.name / log.relative_to(folder), evidence / folder.name / xml.relative_to(folder)
    require(base.sha(saved_log) == base.sha(log) and base.sha(saved_xml) == base.sha(xml), "Archived original failure differs")
    return {"name": REPAIRED_SUITE, "success": False, "exit_code": 8, "skipped": False,
            "source_run": folder.name, "log": base.relative(saved_log, evidence), "log_sha256": base.sha(log),
            "junit": base.relative(saved_xml, evidence), "junit_sha256": base.sha(xml),
            "replaced_in_effective_results": True,
            "reason": "Original assertions observed the canvas synchronously after selecting a row. The approved insertion-only repair waits for real asynchronous review and retains all original assertions and fixtures.", **parsed}


def aggregate(project, full, rerun, evidence, binary, before_path, after_path):
    require(full != rerun and full.name != rerun.name, "Original/rerun evidence folders must be distinct")
    before, after = base.read_json(before_path), base.read_json(after_path)
    current = base.snapshot(project, binary)
    transition = validate_transition(before, after, current)
    original_report = base.read_json(full / "detection-regression-qa.json")
    rerun_report = base.read_json(rerun / "detection-regression-qa.json")
    old_runs = validate_runner(original_report, base.SUITES, current, "Original complete run", passing=False, real_cuda=True)
    new_runs = validate_runner(rerun_report, [REPAIRED_SUITE], current, "CPU-only scope rerun", passing=True, real_cuda=False)
    require([name for name, row in old_runs.items() if row.get("success") is not True] == [REPAIRED_SUITE],
            "The original complete runner did not pass exactly the other 17 suites")
    require(timestamp(before, "Before") <= timestamp(original_report, "Original runner") <= timestamp(after, "After") <=
            timestamp(rerun_report, "Scope runner") <= timestamp(current, "Current"), "Snapshot/runner chronology is inconsistent")
    require(base.read_json(full / "ctest-discovery.json") == base.read_json(rerun / "ctest-discovery.json"),
            "Configured CTest definitions/runtime environment changed between runners")
    originals = base.archive_originals(full, evidence)
    originals.update(base.archive_originals(rerun, evidence))
    snapshots = {}
    for name, path in (("before", before_path), ("after", after_path)):
        require(path.name not in snapshots.values(), "Snapshot archive filename collision")
        originals[path.name] = base.copy_exact(path, evidence / path.name); snapshots[name] = path.name
    source_proof, old_lines = validate_insert_only_source(project, before, after, evidence)
    failed_original = inspect_failed_original(full, old_runs[REPAIRED_SUITE], evidence, old_lines)
    rows = [base.inspect_ctest(rerun, new_runs[name], evidence) if name == REPAIRED_SUITE
            else base.inspect_ctest(full, old_runs[name], evidence) for name in base.SUITES]
    new_log = base.regular_inside(rerun, new_runs[REPAIRED_SUITE]["log"])
    text = base.normalize_ctest(new_log.read_text(encoding="utf-8"))
    passing_cases = [m[2] for m in RECORD.finditer(text) if m[1] == "PASS"]
    require(passing_cases == failed_original["case_ids_in_execution_order"] and len(passing_cases) == len(set(passing_cases)) == 38,
            "Repaired scope must PASS all original 38 case IDs in the identical order, without omission/addition/renaming")
    require(rows[list(base.SUITES).index(REPAIRED_SUITE)]["qt_totals"] ==
            [{"passed": 38, "failed": 0, "skipped": 0, "blacklisted": 0}], "CPU scope rerun did not pass exactly 38 cases")
    profiles = base.cuda_profiles(full, evidence)
    require(not list((rerun / "artifacts").glob("onnx-cuda*.json")), "CPU-only scope rerun unexpectedly includes CUDA suite traces")
    hardware = sorted({entry.get("value", {}).get("actual_cuda_device") for entry in old_runs["vision-gpu-tests"].get("feature_evidence", [])
                       if entry.get("value", {}).get("actual_cuda_device")})
    require(hardware, "Original successful GPU suite lacks actual CUDA hardware evidence")
    passed_qt = sum(total["passed"] for row in rows for total in row["qt_totals"])
    passed_python = sum(row["python_tests"] for row in rows)
    require(passed_qt > 0 and passed_python > 0, "Missing actual Qt/Python results")
    final = base.snapshot(project, binary)
    require(all(final[key] == current[key] for key in FIELDS), "Production/test inputs changed during aggregation")
    transition.update(insertion_only_source_proof=source_proof,
        original_complete_runner=base.relative(evidence / full.name / "detection-regression-qa.json", evidence),
        original_complete_runner_sha256=base.sha(full / "detection-regression-qa.json"),
        repaired_runner=base.relative(evidence / rerun.name / "detection-regression-qa.json", evidence),
        repaired_runner_sha256=base.sha(rerun / "detection-regression-qa.json"),
        original_complete_runner_real_cuda_requested=True, repaired_runner_real_cuda_requested=False,
        repaired_runner_selected_suites=[REPAIRED_SUITE], configured_ctest_environment_unchanged=True,
        repaired_case_ids=passing_cases, repaired_case_count=38, original_passed_case_count=14, original_failed_case_count=24,
        effective_origin_counts={full.name: len(base.SUITES) - 1, rerun.name: 1})
    return {"schema_version": 1, "version": VERSION, "success": True, "errors": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Effective 18 suites: 17 original successes plus one actual CPU-only scope rerun after the reviewed 35-line insertion-only asynchronous-review wait repair. Original scope retained 14 passes and 24 failures; all same 38 cases passed on rerun. This is not a single all-passing 18-suite execution. Production application, CMake cache, other source/native inputs and locks remained byte-identical. Final DEB/install/source acceptance is separate.",
        "single_complete_run_passed": False, "async_review_test_only_repair": True, "fixture_only_repair": True,
        "effective_suite_count": len(rows), "qt_tests_passed": passed_qt, "python_tests_passed": passed_python,
        "failed": 0, "skipped": 0, "blacklisted": 0, "suites": rows,
        **{key: current[key] for key in FIELDS}, "source_hash_scope": current["source_hash_scope"],
        "before_run_snapshot": snapshots["before"], "before_run_snapshot_sha256": base.sha(before_path),
        "after_fixture_snapshot": snapshots["after"], "after_fixture_snapshot_sha256": base.sha(after_path),
        "fixture_repair_transition": transition, "superseded_original_failure": failed_original,
        "additional_verification": [], "original_run_failed_suites": [REPAIRED_SUITE],
        "cuda_profiles": profiles, "hardware": " / ".join(hardware), "original_evidence_files": originals,
        "limitations": ["Not one complete all-passing run: effective totals count the original 17 successes and only the repaired 38-case CPU scope rerun, each suite once.",
            "Only the scope test source/binary changed; source proof retains every old line and permits exactly the reviewed 35 inserted wait/helper/extra-assertion lines, with no model-fixture or assertion removals.",
            "This aggregation executes no builds, tests, inference or package acceptance.",
            "Snapshot identity covers declared runtime metadata/locks, not independent hashing of external runtime trees.",
            "Final corresponding source must include the corrected scope test and be reverified separately.",
            "No Windows 2.5.0 or physical camera acceptance; synthetic/derived labels are not general model-accuracy measurements."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[4]
    parser.add_argument("--project", type=Path, default=root)
    parser.add_argument("--full-run-dir", type=Path)
    parser.add_argument("--rerun-dir", type=Path)
    parser.add_argument("--before-snapshot", type=Path)
    parser.add_argument("--after-snapshot", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    full = (args.full_run_dir or project / f"output/v{VERSION}/qa-full").resolve()
    rerun = (args.rerun_dir or project / f"output/v{VERSION}/qa-scope-async-rerun").resolve()
    before = (args.before_snapshot or project / f"output/v{VERSION}/production-inputs-before.json").resolve()
    after = (args.after_snapshot or project / f"output/v{VERSION}/production-inputs-after-async.json").resolve()
    binary = (args.binary or project / f"output/build/{VERSION}-release/bin/vision-studio").resolve()
    evidence = (args.evidence or project / f"docs/releases/evidence/{VERSION}").resolve()
    report_path = (args.report or evidence / "ctest-qa.json").absolute()
    if report_path.exists() or report_path.is_symlink(): parser.error("Report exists; refusing to overwrite evidence")
    for folder in (full, rerun):
        if not folder.is_dir() or evidence == folder or evidence.is_relative_to(folder): parser.error("Raw run directories must exist; evidence must be outside them")
    for path in (before, after, full / "detection-regression-qa.json", rerun / "detection-regression-qa.json"):
        if not path.is_file() or path.is_symlink(): parser.error(f"Required final raw evidence absent; no report generated: {path}")
    report = {"schema_version": 1, "version": VERSION, "success": False, "errors": [], "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        report = aggregate(project, full, rerun, evidence, binary, before, after)
        report["command"] = [sys.executable, *sys.argv]
        report["summarizer_sha256"] = base.sha(Path(__file__).resolve())
        report["shared_verifier_sha256"] = base.sha(Path(base.__file__).resolve())
    except (base.EvidenceError, OSError, ValueError, TypeError, KeyError, IndexError, ET.ParseError, subprocess.SubprocessError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    success = report.get("success") is True and not report["errors"]
    print(json.dumps({"success": success, "report": str(report_path), "errors": report["errors"],
        "effective_suites": report.get("effective_suite_count"), "qt_passed": report.get("qt_tests_passed"),
        "python_passed": report.get("python_tests_passed"), "single_complete_run_passed": report.get("single_complete_run_passed")}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
