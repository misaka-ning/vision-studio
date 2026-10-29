#!/usr/bin/env python3
"""Validate stage 4c's initial 44 suites plus corrected reruns.

Default is read-only preview. --execute archives byte-exact raw originals and
writes the stage acceptance record; it never runs tests, Git mutations or builds.
This stage is not the final 2.10.1 full-run/package/source acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

CHANGED_PATHS = frozenset({
    "src/ui/pages/modeldisplaypage.cpp", "src/ui/pages/modeldisplaypage.h",
    "tests/conversion_inference_tests.cpp", "tests/ui_tests.cpp", "tests/workspace_pages_tests.cpp",
})
INITIAL_FAILURES = frozenset({
    "vision-ui-tests", "vision-conversion-inference-tests", "vision-model-conversion-tests",
})
SUITE_COUNT = 44


class AcceptanceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing regular original: " + str(path))
    return path


def read(path):
    return json.loads(regular(path).read_text(encoding="utf-8"))


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def load_helper(path):
    spec = importlib.util.spec_from_file_location("stage4c_original_regression_parser", regular(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indexed(report):
    rows = report.get("runs")
    selected = report.get("selected_tests")
    require(isinstance(rows, list) and isinstance(selected, list), "Incomplete runner report")
    result = {row.get("name"): row for row in rows}
    require(len(result) == len(rows) == len(selected) and set(result) == set(selected),
            "Duplicate/missing/unexpected runner suites")
    return result


def choose_effective(initial, corrected, before, current):
    require(set(corrected) <= set(initial), "Rerun contains a suite absent from the initial execution")
    require(set(current) == set(before), "Native binary inventory gained or lost members")
    changed = {name for name in current if current[name] != before[name]}
    require(changed <= set(corrected), "Changed native binary was not rerun: " + ", ".join(sorted(changed - set(corrected))))
    result = {}
    for name in initial:
        origin = "corrected" if name in corrected else "initial"
        row = (corrected if origin == "corrected" else initial)[name]
        require(row.get("success") is True and row.get("exit_code") == 0 and row.get("skipped") is False,
                "Effective suite failed/skipped: " + name)
        if origin == "initial" and name in current:
            require(current[name] == before[name], "Unrerun native binary identity changed: " + name)
        result[name] = (origin, row)
    return result, changed


def tree_inputs(root, source, tree):
    """Bind every tracked regular Git member, not a previously asserted boolean."""
    members = {}
    for record in git(root, "ls-tree", "-r", "-z", tree).split(b"\0"):
        if not record:
            continue
        prefix, raw_name = record.split(b"\t", 1)
        mode, kind, blob = prefix.decode("ascii").split()
        name = raw_name.decode("utf-8")
        require(kind == "blob" and mode in {"100644", "100755"}, "Unsupported Git member: " + name)
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe Git member")
        path = regular(source / relative)
        require(path.resolve().is_relative_to(source.resolve()), "Frozen input escapes snapshot: " + name)
        digest = hashlib.sha1()
        digest.update(b"blob " + str(path.stat().st_size).encode("ascii") + b"\0")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                checksum.update(block)
        require(digest.hexdigest() == blob, "Frozen member differs from actual Git tree: " + name)
        members[name] = {"git_blob": blob, "sha256": checksum.hexdigest(), "bytes": path.stat().st_size}
    require(members, "Empty source input snapshot")
    return members


def verify_raw(folder, report, helper):
    """Verify all raw identities now; failed initial suites are retained, not erased."""
    rows = indexed(report)
    for name, row in rows.items():
        for key in ("log", "junit"):
            path = helper.regular_inside(folder, row.get(key))
            require(path.name == name + (".log" if key == "log" else ".xml"), "Unexpected suite raw filename")
        require(sha(folder / row["log"]) == row.get("log_sha256"), "Runner/log SHA mismatch: " + name)
    return rows


def validate_effective(folder, name, row, helper):
    text = (folder / row["log"]).read_text(encoding="utf-8")
    totals = helper.parse_actual_totals(text, name)
    helper.validate_junit(folder / row["junit"], name)
    require("100% tests passed, 0 tests failed out of 1" in text, "Missing actual CTest success: " + name)
    require(totals["qt_totals"] == row.get("qt_totals"), "Raw totals disagree with metadata: " + name)
    return totals


def collect(args):
    root = args.root.resolve()
    area = root / "output/v2.10.1"
    helper_path = area / "qa-tools/summarize_regressions.py"
    helper = load_helper(helper_path)
    initial_dir = area / "stage4c-accepted-regression"
    corrected_dir = area / "stage4c-corrected-regression"
    initial = read(initial_dir / "detection-regression-qa.json")
    corrected = read(corrected_dir / "detection-regression-qa.json")
    initial_rows = verify_raw(initial_dir, initial, helper)
    corrected_rows = verify_raw(corrected_dir, corrected, helper)
    require(initial.get("schema_version") == corrected.get("schema_version") == 1, "Runner schema changed")
    require(initial.get("version") == corrected.get("version") == "2.10.0", "Unexpected stage application version")
    require(len(initial_rows) == SUITE_COUNT, "Initial execution is not all 44 registered suites")
    require(initial.get("success") is False and len(initial.get("errors", [])) == 3,
            "Initial failed execution must remain accurately recorded")
    require({name for name, row in initial_rows.items() if not row.get("success")} == INITIAL_FAILURES,
            "Unexpected initial failure/skip identities")
    require(initial_rows["vision-model-conversion-tests"]["skipped"] is True and
            initial_rows["vision-model-conversion-tests"]["exit_code"] == 0,
            "Missing original successful-process / skipped-fixture distinction")
    require(corrected.get("success") is True and corrected.get("errors") == [], "Corrected rerun is not complete/successful")
    require(initial.get("real_cuda_requested") is True and corrected.get("real_cuda_requested") is True,
            "Both executions must request actual CUDA")
    require(datetime.fromisoformat(initial["generated_at_utc"]) < datetime.fromisoformat(corrected["generated_at_utc"]),
            "Corrected report does not follow initial execution")
    planned = read(area / "stage4c-corrected-test-selection.json")["selected"]
    require(len(planned) == len(set(planned)) == 15 and set(planned) <= set(corrected_rows), "Planned corrected 15 suites not all rerun")
    require(set(corrected_rows) - set(planned) <= {"vision-gpu-tests", "vision-gpu-runtime-tests"},
            "Unexpected additional rerun outside planned suites/GPU runner additions")

    discoveries = {label: read(folder / "ctest-discovery.json") for label, folder in
                   (("initial", initial_dir), ("corrected", corrected_dir))}
    discovered = {label: {row["name"]: row for row in value.get("tests", [])}
                  for label, value in discoveries.items()}
    for label in discoveries:
        require(len(discoveries[label]["tests"]) == len(discovered[label]) == SUITE_COUNT and
                set(discovered[label]) == set(initial_rows), "Fresh discovery is not the same 44 suites: " + label)
    require({key: row["command"] for key, row in discovered["initial"].items()} ==
            {key: row["command"] for key, row in discovered["corrected"].items()}, "CTest commands changed unexpectedly")

    build = root / "output/build/2.10.1-stage4c-accepted"
    binary = regular(build / "bin/vision-studio")
    before_all = read(area / "stage4c-before-final-fixes-binaries.json")
    before = {key: value for key, value in before_all.items() if key != "bin/vision-studio"}
    current = {}
    native_paths = {}
    for name, row in discovered["corrected"].items():
        command = row["command"]
        candidate = Path(command[0])
        if candidate.parent.resolve() == build.resolve() or candidate.parent.resolve() == (build / "bin").resolve():
            require(candidate.name == name, "Native binary/name mismatch")
            native_paths[name] = regular(candidate)
            current[name] = sha(candidate)
    require(len(before) == len(current) == 36, "Native suite inventory premise changed")
    effective, changed = choose_effective(initial_rows, corrected_rows, before, current)
    require(set(changed) <= set(planned), "Changed native binary omitted from explicit corrected selection")
    require(initial["binary_sha256"] == before_all["bin/vision-studio"], "Initial application/before-map mismatch")
    require(corrected["binary_sha256"] == sha(binary), "Corrected application identity mismatch")
    for report in (initial, corrected):
        require(Path(report["binary"]).resolve() == binary.resolve() and
                Path(report["build_directory"]).resolve() == build.resolve(), "Runner used another app/build")
    for name, (_, row) in effective.items():
        command = row.get("command", [])
        require("--test-dir" in command and Path(command[command.index("--test-dir") + 1]).resolve() == build.resolve(),
                "Suite used another configured build: " + name)

    initial_commit = git(root, "rev-parse", args.initial_commit + "^{commit}").decode().strip()
    corrected_commit = git(root, "rev-parse", args.corrected_commit + "^{commit}").decode().strip()
    initial_tree = git(root, "rev-parse", initial_commit + "^{tree}").decode().strip()
    corrected_tree = git(root, "rev-parse", corrected_commit + "^{tree}").decode().strip()
    require(subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", initial_commit, corrected_commit]).returncode == 0,
            "Corrected commit is not a descendant of initial input")
    changed_paths = set(git(root, "diff", "--name-only", "-z", initial_tree, corrected_tree).decode().strip("\0").split("\0"))
    require(changed_paths == CHANGED_PATHS, "Git source correction differs from exact five authorized paths")
    binding = read(area / "stage4c-accepted-source-binding.json")
    require(binding["source_tree"] == initial_tree and binding["binary_sha256"] == initial["binary_sha256"],
            "Original binding is not the initial tested input")
    source_before = area / "stage4c-accepted-source-before-final-fixes"
    source_current = area / "stage4c-accepted-source"
    sources = {"initial": tree_inputs(root, source_before, initial_tree),
               "corrected": tree_inputs(root, source_current, corrected_tree)}
    require(set(sources["initial"]) == set(sources["corrected"]), "Frozen Git source members gained/lost files")
    actual_changes = {name for name in sources["initial"] if
                      sources["initial"][name]["sha256"] != sources["corrected"][name]["sha256"]}
    require(actual_changes == CHANGED_PATHS, "Frozen source bytes differ outside the authorized five paths")
    production = {label: helper.code_hashes(source) for label, source in
                  (("initial", source_before), ("corrected", source_current))}
    require(set(production["initial"]) == set(production["corrected"]), "Production input members changed")
    require({name for name in production["initial"] if production["initial"][name] != production["corrected"][name]} == CHANGED_PATHS,
            "Production hash delta is not the same five files")

    fixture_path = area / "stage4c-inference-only-runtime-fixture.json"
    fixture = read(fixture_path)
    require(fixture["source_inputs_changed"] is False and fixture["probe_onnx_absent"] is True,
            "Inference-only fixture premise changed")
    link = Path(fixture["link"])
    require(link == source_current / "build/release-runtime" and link.is_symlink() and
            link.resolve() == Path(fixture["runtime"]).resolve(), "Runtime fixture link identity changed")
    require(sha(Path(fixture["runtime"]) / "bin/python") == fixture["interpreter_sha256"], "Runtime fixture interpreter changed")

    copies = {}
    def add(path, relative):
        path = regular(path)
        require(relative not in copies, "Duplicate raw evidence destination")
        copies[relative] = {"source": path, "bytes": path.stat().st_size, "sha256": sha(path)}
    for label, folder in (("initial", initial_dir), ("corrected", corrected_dir)):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {".log", ".xml", ".json", ".stdout", ".stderr"}:
                add(path, label + "/" + path.name)
        for path in sorted((folder / "artifacts").glob("*.json")):
            add(path, label + "/artifacts/" + path.name)
    for name in ("stage4c-accepted-configure.log", "stage4c-accepted-build.log", "stage4c-accepted-build-retry.log",
                 "stage4c-corrected-build.log", "stage4c-before-final-fixes-binaries.json",
                 "stage4c-accepted-source-binding.json", "stage4c-corrected-test-selection.json",
                 "stage4c-inference-only-runtime-fixture.json", "stage4c-accepted-runner.log", "stage4c-corrected-runner.log"):
        add(area / name, "inputs/" + name)
    add(Path(__file__).resolve(), "tools/stage4c_acceptance.py")
    add(helper_path, "tools/summarize_regressions.py")
    add(area / "prep/test_stage4c_acceptance.py", "tools/test_stage4c_acceptance.py")
    add(area / "prep/stage4c-acceptance-unit-tests.log", "tools/stage4c-acceptance-unit-tests.log")
    require(not re.search(r"(?m)^(FAILED:|ninja: build stopped)", (area / "stage4c-corrected-build.log").read_text()),
            "Corrected rebuild log reports failure")

    rows = []
    for name, (origin, run) in effective.items():
        folder = initial_dir if origin == "initial" else corrected_dir
        totals = validate_effective(folder, name, run, helper)
        command = discovered[origin][name]["command"]
        native = name in current
        interpreter = regular(Path(command[0]).resolve()) if not native else None
        rows.append({"name": name, "success": True, "source_run": origin,
                     "actual_ctest_command": run["command"], "discovered_test_command": command,
                     "native_binary_sha256": current.get(name),
                     "native_initial_sha256": before.get(name),
                     "unrerun_native_binary_identical": native and origin == "initial" and current[name] == before[name],
                     "current_interpreter_sha256": sha(interpreter) if interpreter else None,
                     "log": origin + "/" + run["log"], "log_sha256": sha(folder / run["log"]),
                     "junit": origin + "/" + run["junit"], "junit_sha256": sha(folder / run["junit"]),
                     **totals})
    gpu_origin, gpu_run = effective["vision-gpu-tests"]
    gpu_folder = initial_dir if gpu_origin == "initial" else corrected_dir
    profiles = []
    for path in sorted((gpu_folder / "artifacts").glob("onnx-cuda*.json")):
        events = read(path)
        require(isinstance(events, list), "CUDA trace is not an event array")
        nodes = [event for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider"]
        require(nodes, "No actual CUDAExecutionProvider node events")
        profiles.append({"path": gpu_origin + "/artifacts/" + path.name, "sha256": sha(path), "cuda_node_events": len(nodes)})
    require(len(profiles) == 2, "Effective GPU suite requires two real CUDA traces")
    hardware = sorted({row.get("value", {}).get("actual_cuda_device") for row in gpu_run.get("feature_evidence", [])
                       if row.get("value", {}).get("actual_cuda_device")})
    require(hardware, "Effective GPU run did not record actual CUDA device")
    initial_failures = [{"name": name, "exit_code": initial_rows[name]["exit_code"],
                         "skipped": initial_rows[name]["skipped"], "raw_log": "initial/" + initial_rows[name]["log"]}
                        for name in sorted(INITIAL_FAILURES)]
    result = {
        "schema_version": 1, "stage": "4c", "version": "2.10.1", "tested_application_version": "2.10.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "success": True, "errors": [],
        "build_success": True, "regression_success": True, "not_final_release_acceptance": True,
        "scope": "Production page/controller wiring, configuration authority, lifecycle, drop and independent smoke boundaries. Effective 44-suite coverage combines the initial execution with corrected reruns; it is not one single all-green 44-suite invocation.",
        "initial_execution": {"suite_count": len(initial_rows), "successful_suites": len(initial_rows) - len(INITIAL_FAILURES),
                              "success": False, "failed_or_fixture_skipped": initial_failures},
        "corrected_execution": {"suite_count": len(corrected_rows), "success": True, "suites": list(corrected_rows)},
        "effective_suite_count": len(rows), "qt_tests_passed": sum(t["passed"] for row in rows for t in row["qt_totals"]),
        "python_tests_passed": sum(row["python_tests"] for row in rows), "failed": 0, "skipped": 0,
        "blacklisted": 0, "suites": rows, "source_tree": corrected_tree,
        "source_binding": {"initial_commit": initial_commit, "initial_tree": initial_tree,
                           "corrected_commit": corrected_commit, "corrected_tree": corrected_tree,
                           "changed_paths": sorted(CHANGED_PATHS),
                           "file_changes": {name: {"before": sources["initial"][name], "after": sources["corrected"][name]} for name in sorted(CHANGED_PATHS)},
                           "initial_tracked_members_verified": len(sources["initial"]),
                           "corrected_tracked_members_verified": len(sources["corrected"]),
                           "manifest": "inputs/source-input-manifests.json", "diff": "inputs/source-correction.diff"},
        "binary_binding": {"initial_application_sha256": initial["binary_sha256"],
                           "corrected_application_sha256": corrected["binary_sha256"],
                           "native_binaries_before": before, "native_binaries_current": current,
                           "changed_native_suites": sorted(changed),
                           "unrerun_native_suite_count": sum(row["unrerun_native_binary_identical"] for row in rows)},
        "production_input_manifests": "inputs/production-input-manifests.json", "cuda_profiles": profiles,
        "hardware": hardware, "raw_evidence": "../releases/evidence/2.10.1/stages/stage4c",
        "initial_failure_explanations": [
            "UI recording list publication preceded asynchronous decoder readiness; the original immediate assertion failed. Corrected test waits for the same Play-enabled condition and retains playback, dimensions, grayscale and byte-exact export assertions.",
            "Conversion/inference test crashed during page teardown. ModelDisplayPage now deletes its owned viewer while presentation members still exist; real converted-output identity checks are retained.",
            "Python conversion process exited zero with one skipped TorchScript-without-ONNX check because the ignored inference-only runtime fixture was absent from the Git export. The actual existing runtime is linked read-only for corrected execution; tracked source and dependencies are unchanged by the fixture link."],
        "limitations": [
            "This is staged effective coverage, not a claim that the failed initial invocation passed or that one corrected invocation ran all 44 suites.",
            "Final 2.10.1 requires its own fresh complete 44-suite execution, final DEB/install/source rebuild and remote publication verification.",
            "Python commands and unchanged tracked inputs are bound across runs; interpreter digests are observed at aggregation, not falsely represented as a historic before-run interpreter snapshot.",
            "No Windows native build/install, physical camera, accuracy improvement or performance guarantee is established by this record."],
        "files": {name: {key: value[key] for key in ("bytes", "sha256")} for name, value in copies.items()},
    }
    manifests = {"source-input-manifests.json": sources, "production-input-manifests.json": production}
    diff = git(root, "diff", "--binary", initial_tree, corrected_tree, "--", *sorted(CHANGED_PATHS))
    # Re-open all originals immediately before an execution is allowed to copy.
    require(all(sha(row["source"]) == row["sha256"] for row in copies.values()), "Original evidence changed during aggregation")
    require(all(sha(native_paths[name]) == value for name, value in current.items()) and sha(binary) == corrected["binary_sha256"],
            "Binaries changed during aggregation")
    return result, copies, manifests, diff, helper


def execute(root, report, copies, manifests, diff, helper):
    target = root / "docs/releases/evidence/2.10.1/stages/stage4c"
    acceptance = root / "docs/architecture/stage4c-acceptance.json"
    require(not acceptance.exists() and not acceptance.is_symlink(), "Acceptance report already exists; refusing replacement")
    for destination in (target, acceptance):
        for path in (destination, *destination.parents):
            if path == root:
                break
            require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Evidence target escapes repository")
    for name, row in copies.items():
        helper.copy_exact(row["source"], target / name)
    generated = {"inputs/" + name: (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                 for name, value in manifests.items()}
    generated["inputs/source-correction.diff"] = diff
    for name, data in generated.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.is_symlink(), "Generated evidence target is a symlink")
        if path.exists(): require(path.read_bytes() == data, "Generated evidence already differs")
        else:
            with path.open("xb") as stream: stream.write(data)
        report["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    require(all(sha(target / name) == row["sha256"] for name, row in report["files"].items()), "Archived original changed")
    acceptance.parent.mkdir(parents=True, exist_ok=True)
    with acceptance.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--initial-commit", default="f24e1aa2f311290cf3a28c3a40f40b45d4343c5b")
    parser.add_argument("--corrected-commit", default="168abdf503c490eaa693735f6e55104a70841ce2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        report, copies, manifests, diff, helper = collect(args)
        if args.execute: execute(args.root.resolve(), report, copies, manifests, diff, helper)
        print(json.dumps({"success": True, "mode": "execute" if args.execute else "read-only-preview",
                          "effective_suites": report["effective_suite_count"],
                          "initial_successful_suites": report["initial_execution"]["successful_suites"],
                          "corrected_suites": report["corrected_execution"]["suite_count"],
                          "qt_tests_passed": report["qt_tests_passed"], "python_tests_passed": report["python_tests_passed"],
                          "changed_native_suites": len(report["binary_binding"]["changed_native_suites"]),
                          "unrerun_native_identical": report["binary_binding"]["unrerun_native_suite_count"],
                          "source_tree": report["source_tree"], "original_files": len(copies),
                          "docs_written": args.execute}, ensure_ascii=False))
        return 0
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError, ET.ParseError) as error:
        print(json.dumps({"success": False, "mode": "execute" if args.execute else "read-only-preview",
                          "error": str(error), "docs_written": "possibly_partial" if args.execute else False}, ensure_ascii=False))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
