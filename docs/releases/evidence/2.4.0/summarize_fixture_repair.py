#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Audit the retained 17+1 regression after one conversion-test fixture repair.

This is deliberately not the single-run aggregator. The original complete
18-suite run must retain exactly one failed suite; a second real execution
must select and pass exactly that suite plus the runner-required GPU suite.
The duplicated GPU result is additional evidence, not counted twice. Only the
conversion test source and its native test
binary may change between the two input snapshots. Production application,
all other snapshotted code and tests, CMake cache, runtime metadata and locks
must retain their identities. Original failures are archived, never edited.
No tests, inference, conversion, builds, installs or package checks run here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import summarize_regressions as base

VERSION = "2.4.0"
FIXTURE = "tests/conversion_inference_tests.cpp"
REPAIRED_SUITE = "vision-conversion-inference-tests"
CLASS = "ConversionInferenceTests"
FIELDS = ("production_binary", "production_binary_sha256", "build_directory",
          "cmake_cache_sha256", "source_code_input_sha256", "current_test_binary_sha256")
IDENTITY_FIELDS = ("production_binary", "production_binary_sha256", "build_directory", "cmake_cache_sha256")


def require(value, message):
    base.require(value, message)


def timestamp(report, name):
    value = datetime.fromisoformat(report["generated_at_utc"])
    require(value.tzinfo is not None, f"{name}: UTC-offset timestamp is required")
    return value


def changed_entries(before, after, name):
    require(isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after),
            f"{name}: missing or added snapshot inputs are forbidden")
    return [key for key in before if before[key] != after[key]]


def validate_transition(before, after, current):
    require(before.get("version") == after.get("version") == VERSION,
            "Both snapshots must belong to this application version")
    require(before.get("summarizer_sha256") == after.get("summarizer_sha256") ==
            base.sha(Path(base.__file__).resolve()),
            "The shared snapshot generator changed between before, after and this audit")
    require(all(before.get(key) == after.get(key) for key in IDENTITY_FIELDS),
            "Production application, build location or CMake cache changed during fixture repair")
    code_changes = changed_entries(before.get("source_code_input_sha256"),
                                   after.get("source_code_input_sha256"), "Source identity")
    require(code_changes == [FIXTURE],
            "Allowed source transition is exactly the one conversion test fixture: " + repr(code_changes))
    binary_changes = changed_entries(before.get("current_test_binary_sha256"),
                                     after.get("current_test_binary_sha256"), "Native test identity")
    require(binary_changes == [REPAIRED_SUITE],
            "Allowed native binary transition is exactly the repaired test: " + repr(binary_changes))
    old_binary = before["current_test_binary_sha256"][REPAIRED_SUITE]
    new_binary = after["current_test_binary_sha256"][REPAIRED_SUITE]
    require(isinstance(old_binary, dict) and isinstance(new_binary, dict) and
            set(old_binary) == set(new_binary) == {"path", "sha256"} and
            old_binary["path"] == new_binary["path"] and old_binary["sha256"] != new_binary["sha256"],
            "The repaired test must change SHA256 at the same native binary path")
    require(all(after.get(key) == current[key] for key in FIELDS),
            "Current source or binaries no longer match the after-fixture snapshot")
    return {
        "allowed_source_changes": code_changes,
        "source_transition": {FIXTURE: {"before_sha256": before["source_code_input_sha256"][FIXTURE],
                                        "after_sha256": after["source_code_input_sha256"][FIXTURE]}},
        "allowed_native_binary_changes": binary_changes,
        "native_binary_transition": {REPAIRED_SUITE: {"path": new_binary["path"],
            "before_sha256": old_binary["sha256"], "after_sha256": new_binary["sha256"]}},
        "other_source_inputs_unchanged": True,
        "other_native_test_binaries_unchanged": True,
        "production_application_unchanged": True,
        "cmake_cache_unchanged": True,
        "snapshotted_runtime_metadata_and_dependency_locks_unchanged": True,
        "current_inputs_match_after_snapshot": True,
        "manual_review_boundary": "Path/hash verification restricts the repair to one test fixture; "
            "it does not by itself prove semantic equivalence of edited assertions. "
            "The repository diff must be reviewed to retain all conversion/inference assertions."
    }


def validate_runner(report, expected, current, name, passing):
    require(report.get("version") == VERSION and report.get("real_cuda_requested") is True,
            f"{name}: expected this version with real CUDA requested")
    selected = report.get("selected_tests", [])
    require(len(selected) == len(expected) and set(selected) == set(expected),
            f"{name}: did not select exactly the required suites")
    require(report.get("binary_sha256") == current["production_binary_sha256"] and
            Path(report.get("binary", "")).resolve() == Path(current["production_binary"]) and
            Path(report.get("build_directory", "")).resolve() == Path(current["build_directory"]),
            f"{name}: runner is not bound to the unchanged production application/build")
    if passing:
        require(report.get("success") is True and report.get("errors") == [],
                f"{name}: actual repaired-suite runner did not pass")
    else:
        errors = report.get("errors")
        require(report.get("success") is False and isinstance(errors, list) and len(errors) == 1 and
                errors[0].startswith(REPAIRED_SUITE + ":"),
                f"{name}: original run must retain exactly the known suite failure")
    runs = base.indexed_runs(report, expected)
    for suite, run in runs.items():
        command = run.get("command")
        require(isinstance(command, list) and "--test-dir" in command and "--tests-regex" in command,
                f"{name}/{suite}: missing exact CTest command")
        require(Path(command[command.index("--test-dir") + 1]).resolve() == Path(current["build_directory"]),
                f"{name}/{suite}: command used another build")
        require(command[command.index("--tests-regex") + 1] == "^" + re.escape(suite) + "$",
                f"{name}/{suite}: command did not isolate its named suite")
    return runs


def qt_failed_fixture_text(text, label):
    text = base.normalize_ctest(text)
    summaries = list(base.QT_TOTALS.finditer(text))
    require(len(summaries) == 1, f"{label}: missing/ambiguous original Qt totals")
    passed, failed, skipped, blacklisted, milliseconds = map(int, summaries[0].groups())
    require(passed > 0 and failed > 0 and skipped == blacklisted == 0,
            f"{label}: expected actual fixture failures with no skip/blacklist")
    require(not re.search(r"^(?:SKIP\s*:|QFATAL\s*:|XPASS\s*:|XFAIL\s*:)", text, re.MULTILINE),
            f"{label}: unexpected skip/fatal/expected failure")
    starts = re.findall(r"^\*+ Start testing of ([^\n]+?) \*+\s*$", text, re.MULTILINE)
    finishes = re.findall(r"^\*+ Finished testing of ([^\n]+?) \*+\s*$", text, re.MULTILINE)
    require(starts == finishes == [CLASS], f"{label}: incomplete original test suite")
    failure = re.compile(
        r"^FAIL!\s*:\s*(ConversionInferenceTests::convertedModelsUseTheirOwnTaskClassesChannelsAndPixels\([^\n]*\))"
        r" Compared values are not the same\n"
        r"\s*Actual\s+\(starts\.size\(\)\)\s*:\s*0\n"
        r"\s*Expected\s+\(1\)\s*:\s*1\n"
        r"\s*Loc:\s*\[[^\n]*tests/conversion_inference_tests\.cpp\(\d+\)\]\s*$", re.MULTILINE)
    cases = failure.findall(text)
    all_failed = re.findall(r"^FAIL!\s*:", text, re.MULTILINE)
    require(len(cases) == len(set(cases)) == len(all_failed) == failed,
            f"{label}: original failures were not exclusively the known starts.size()==0 assertion")
    pass_cases = re.findall(r"^PASS\s*:\s*(ConversionInferenceTests::[^\n]+?\([^\n]*\))\s*$", text, re.MULTILINE)
    require(len(pass_cases) == len(set(pass_cases)) == passed,
            f"{label}: original passing case list differs from its actual totals")
    return {"qt_totals": [{"passed": passed, "failed": failed, "skipped": skipped,
                            "blacklisted": blacklisted}], "failed_case_ids": cases,
            "passed_case_ids": pass_cases, "result_elapsed_seconds": milliseconds / 1000.}


def inspect_original_failure(folder, run, evidence):
    require(run.get("name") == REPAIRED_SUITE and run.get("success") is False and
            run.get("exit_code") == 8 and run.get("skipped") is False,
            "Original replaced suite must retain its actual CTest failure")
    log = base.regular_inside(folder, run.get("log"))
    junit = base.regular_inside(folder, run.get("junit"))
    require(base.sha(log) == run.get("log_sha256"), "Original failed log changed from runner metadata")
    parsed = qt_failed_fixture_text(log.read_text(encoding="utf-8"), "Original raw log")
    require(run.get("qt_totals") == parsed["qt_totals"], "Original failed Qt totals differ from runner metadata")
    require("0% tests passed, 1 tests failed out of 1" in log.read_text(encoding="utf-8"),
            "Original CTest failure footer is missing")
    root = ET.fromstring(junit.read_text(encoding="utf-8"))
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    require(len(suites) == 1, "Original failed JUnit must contain one CTest suite")
    suite = suites[0]
    require(int(suite.get("tests", "0")) == 1 and int(suite.get("failures", "0")) == 1 and
            all(int(suite.get(k, "0")) == 0 for k in ("errors", "skipped", "disabled")),
            "Original JUnit must record exactly one failed CTest suite, no skips")
    cases = suite.findall("testcase")
    require(len(cases) == 1 and cases[0].get("name") == REPAIRED_SUITE and
            cases[0].get("status") == "fail" and cases[0].find("failure") is not None and
            not suite.findall(".//error") and not suite.findall(".//skipped"),
            "Original JUnit failure testcase differs")
    require(qt_failed_fixture_text(cases[0].findtext("system-out", ""), "Original JUnit output") == parsed,
            "Original JUnit Qt details differ from the archived failed log")
    archived_log = evidence / folder.name / log.relative_to(folder)
    archived_junit = evidence / folder.name / junit.relative_to(folder)
    require(base.sha(archived_log) == base.sha(log) and base.sha(archived_junit) == base.sha(junit),
            "Archived original failure differs from its raw evidence")
    return {"name": REPAIRED_SUITE, "success": False, "exit_code": run["exit_code"], "skipped": False,
            "source_run": folder.name, "log": base.relative(archived_log, evidence),
            "log_sha256": base.sha(log), "junit": base.relative(archived_junit, evidence),
            "junit_sha256": base.sha(junit), "replaced_in_effective_results": True,
            "reason": "Test fixture relied on removed automatic startup image; every retained failure "
                      "occurred at starts.size()==0. Conversion/inference must be proven by the repaired rerun.",
            **parsed}


def aggregate(project, full, rerun, evidence, binary, before_path, after_path):
    require(full != rerun and full.name != rerun.name, "Original and repaired runner directories must be distinct")
    before, after = base.read_json(before_path), base.read_json(after_path)
    current = base.snapshot(project, binary)
    transition = validate_transition(before, after, current)
    original_report = base.read_json(full / "detection-regression-qa.json")
    rerun_report = base.read_json(rerun / "detection-regression-qa.json")
    old_runs = validate_runner(original_report, base.SUITES, current, "Original complete run", False)
    new_runs = validate_runner(rerun_report, [REPAIRED_SUITE, "vision-gpu-tests"], current, "Fixture/GPU rerun", True)
    failed_names = [name for name, row in old_runs.items() if row.get("success") is not True]
    require(failed_names == [REPAIRED_SUITE], "Original complete run did not pass exactly the other 17 suites")
    require(timestamp(before, "Before snapshot") <= timestamp(original_report, "Original runner") <=
            timestamp(after, "After snapshot") <= timestamp(rerun_report, "Repaired runner") <=
            timestamp(current, "Current snapshot"), "Snapshot/runner chronology is inconsistent")
    discovery_before = base.read_json(full / "ctest-discovery.json")
    discovery_after = base.read_json(rerun / "ctest-discovery.json")
    require(discovery_before == discovery_after,
            "CTest definitions/configured runtime environment changed between full run and fixture rerun")
    originals = base.archive_originals(full, evidence)
    originals.update(base.archive_originals(rerun, evidence))
    snapshot_files = {}
    for label, path in (("before", before_path), ("after", after_path)):
        destination = evidence / path.name
        require(path.name not in snapshot_files.values(), "Snapshot destination filenames collide")
        originals[destination.name] = base.copy_exact(path, destination)
        snapshot_files[label] = destination.name
    failed_original = inspect_original_failure(full, old_runs[REPAIRED_SUITE], evidence)
    rows = []
    for name in base.SUITES:
        folder, row = (rerun, new_runs[name]) if name == REPAIRED_SUITE else (full, old_runs[name])
        rows.append(base.inspect_ctest(folder, row, evidence))
    repaired_log = base.regular_inside(rerun, new_runs[REPAIRED_SUITE]["log"])
    actual_pass_cases = re.findall(r"^PASS\s*:\s*(ConversionInferenceTests::[^\n]+?\([^\n]*\))\s*$",
                                  base.normalize_ctest(repaired_log.read_text(encoding="utf-8")), re.MULTILINE)
    expected_cases = failed_original["failed_case_ids"] + failed_original["passed_case_ids"]
    require(len(actual_pass_cases) == len(set(actual_pass_cases)) == len(expected_cases) and
            set(actual_pass_cases) == set(expected_cases),
            "The repaired suite did not actually PASS every original case, or changed/omitted its case list")
    profiles = base.cuda_profiles(full, evidence)
    additional_gpu = base.inspect_ctest(rerun, new_runs["vision-gpu-tests"], evidence)
    additional_gpu_profiles = base.cuda_profiles(rerun, evidence)
    additional_gpu.update(counted_in_effective_totals=False,
        reason="The generic --with-gpu runner appends the GPU suite; effective GPU coverage uses the original successful run.",
        cuda_profiles=additional_gpu_profiles)
    hardware = sorted({entry.get("value", {}).get("actual_cuda_device")
        for entry in old_runs["vision-gpu-tests"].get("feature_evidence", [])
        if entry.get("value", {}).get("actual_cuda_device")})
    require(hardware, "Original successful GPU suite lacks actual CUDA hardware evidence")
    passed_qt = sum(total["passed"] for row in rows for total in row["qt_totals"])
    passed_python = sum(row["python_tests"] for row in rows)
    require(passed_qt > 0 and passed_python > 0, "Missing effective Qt or Python results")
    final = base.snapshot(project, binary)
    require(all(final[key] == current[key] for key in FIELDS), "Inputs changed while aggregating evidence")
    transition.update(original_complete_runner=base.relative(evidence / full.name / "detection-regression-qa.json", evidence),
        original_complete_runner_sha256=base.sha(full / "detection-regression-qa.json"),
        repaired_runner=base.relative(evidence / rerun.name / "detection-regression-qa.json", evidence),
        repaired_runner_sha256=base.sha(rerun / "detection-regression-qa.json"),
        repaired_runner_selected_suites=[REPAIRED_SUITE, "vision-gpu-tests"],
        configured_ctest_environment_unchanged=True,
        repaired_case_ids=actual_pass_cases,
        repaired_case_count=len(actual_pass_cases),
        original_failed_case_count=len(failed_original["failed_case_ids"]),
        effective_origin_counts={full.name: len(base.SUITES) - 1, rerun.name: 1})
    return {"schema_version": 1, "version": VERSION, "success": True, "errors": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Effective 18 suites: 17 successes from the original complete run plus one actual "
                 "conversion-inference rerun after a test-fixture-only repair. The rerun also passed the runner-required GPU suite, retained as additional evidence without duplicate counting. The original run failed; "
                 "this is not a single all-passing 18-suite run and test inputs did not remain wholly unchanged. "
                 "Production application and other snapshotted inputs retained identical hashes. "
                 "Final DEB/install/corresponding-source acceptance is separate.",
        "single_complete_run_passed": False, "fixture_only_repair": True,
        "effective_suite_count": len(rows), "qt_tests_passed": passed_qt, "python_tests_passed": passed_python,
        "failed": 0, "skipped": 0, "blacklisted": 0, "suites": rows,
        **{key: current[key] for key in FIELDS}, "source_hash_scope": current["source_hash_scope"],
        "before_run_snapshot": snapshot_files["before"], "before_run_snapshot_sha256": base.sha(before_path),
        "after_fixture_snapshot": snapshot_files["after"], "after_fixture_snapshot_sha256": base.sha(after_path),
        "fixture_repair_transition": transition, "superseded_original_failure": failed_original,
        "additional_verification": [additional_gpu],
        "original_run_failed_suites": [REPAIRED_SUITE],
        "cuda_profiles": profiles, "hardware": " / ".join(hardware), "original_evidence_files": originals,
        "limitations": ["The repaired conversion suite and the runner-required GPU suite were rerun after the test-fixture edit. "
                        "Effective totals use the original 17 successes plus only the repaired conversion suite; "
                        "the duplicated GPU success is additional, not counted twice. "
                        "The other successful suites retain unchanged test/source/binary identities.",
                        "The aggregation performs no builds, tests, inference, conversion or package acceptance.",
                        "Snapshot identity covers declared runtime metadata and dependency locks; "
                        "external runtime trees are not independently rehashed by this tool.",
                        "Final corresponding source must include the corrected test fixture and be reverified separately.",
                        "No Windows 2.4.0 build or installation acceptance", "No physical camera acceptance",
                        "Synthetic/derived labels validate evaluation mechanics, not general model accuracy",
                        "Local stop/timing observations are not performance guarantees"]}


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
    rerun = (args.rerun_dir or project / f"output/v{VERSION}/qa-conversion-fixture-rerun").resolve()
    before = (args.before_snapshot or project / f"output/v{VERSION}/production-inputs-before.json").resolve()
    after = (args.after_snapshot or project / f"output/v{VERSION}/production-inputs-after-fixture.json").resolve()
    binary = (args.binary or project / f"output/build/{VERSION}-release/bin/vision-studio").resolve()
    evidence = (args.evidence or project / f"docs/releases/evidence/{VERSION}").resolve()
    report_path = (args.report or evidence / "ctest-qa.json").absolute()
    if report_path.exists() or report_path.is_symlink():
        parser.error("Report exists; refusing to overwrite evidence")
    for folder in (full, rerun):
        if not folder.is_dir() or evidence == folder or evidence.is_relative_to(folder):
            parser.error("Both original runner directories must exist and evidence must be outside them")
    report = {"schema_version": 1, "version": VERSION, "success": False, "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        report = aggregate(project, full, rerun, evidence, binary, before, after)
        report["command"] = [sys.executable, *sys.argv]
        report["summarizer_sha256"] = base.sha(Path(__file__).resolve())
        report["shared_verifier_sha256"] = base.sha(Path(base.__file__).resolve())
    except (base.EvidenceError, OSError, ValueError, TypeError, KeyError, IndexError, ET.ParseError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    success = report.get("success") is True and not report["errors"]
    print(json.dumps({"success": success, "report": str(report_path), "errors": report["errors"],
                      "effective_suites": report.get("effective_suite_count"),
                      "qt_passed": report.get("qt_tests_passed"), "python_passed": report.get("python_tests_passed"),
                      "single_complete_run_passed": report.get("single_complete_run_passed")}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
