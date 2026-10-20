#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate completed 2.3.1 acceptance; never creates a presumed passing check.

Run only after the final DEB, corresponding source archive, applicable package
acceptance reports and summarize_regressions.py output exist. Every assertion
is read-only validation of retained raw evidence. No inference is performed by
this aggregation, and previous failed diagnostics remain in their own folders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import os

VERSION = "2.3.1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), f"Missing/non-regular evidence: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def passed(report, name, deb_sha):
    require(report.get("success") is True and not report.get("errors"), f"{name} did not pass")
    actual = report.get("deb_sha256") or report.get("details", {}).get("deb_sha256")
    require(actual == deb_sha, f"{name} does not bind the final DEB SHA256")
    require(report.get("version", VERSION) == VERSION, f"{name} belongs to another version")


def suite(ctest, evidence, name):
    rows = [row for row in ctest.get("suites", []) if row.get("name") == name]
    require(len(rows) == 1, f"Missing/duplicate effective suite: {name}")
    row = rows[0]
    require(row.get("success") is True and row.get("exit_code") == 0 and not row.get("skipped"),
            f"Effective suite did not pass: {name}")
    path = (evidence / row["log"]).resolve()
    require(path.is_relative_to(evidence.resolve()) and path.is_file() and sha(path) == row["log_sha256"],
            f"Effective raw log changed: {name}")
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"(?m)^\d+:\s?", "", text)
    require(not re.search(r"(?m)^\s*(FAIL!|SKIP\s*:|XFAIL|QFATAL)", text), f"Failure/skip in {name}")
    totals = re.findall(r"Totals:\s*(\d+) passed,\s*(\d+) failed,\s*(\d+) skipped,\s*(\d+) blacklisted", text)
    if row.get("qt_totals"):
        parsed = [{"passed": int(p), "failed": int(f), "skipped": int(s), "blacklisted": int(b)} for p, f, s, b in totals]
        require(parsed == row["qt_totals"] and all(p["passed"] > 0 and p["failed"] == p["skipped"] == p["blacklisted"] == 0 for p in parsed),
                f"Qt raw totals disagree: {name}")
        count = sum(p["passed"] for p in parsed)
    else:
        ran = re.findall(r"Ran (\d+) tests? in", text)
        require(len(ran) == 1 and int(ran[0]) == row.get("python_tests") and
                re.search(r"(?m)^OK\s*$", text) and "skipped=" not in text,
                f"Python raw totals disagree: {name}")
        count = int(ran[0])
    return {"name": name, "log": row["log"], "log_sha256": row["log_sha256"], "actual_passed": count}, text


def named_pass(text, class_name, method):
    return bool(re.search(r"(?m)^\s*PASS\s*:\s*" + re.escape(class_name + "::" + method + "()") + r"\s*$", text))


def aggregate(project, release, evidence):
    deb = release / f"vision-studio_{VERSION}-1_amd64.deb"
    source_archive = release / f"vision-studio-{VERSION}-complete-source.tar.xz"
    require(deb.is_file() and source_archive.is_file(), "Final DEB and complete corresponding source must exist")
    deb_sha = sha(deb)
    raw_names = ("qa-report.json", "install-qa.json", "source-bundle-qa.json", "packaged-gpu-qa.json", "packaged-evaluation-qa.json")
    reports = {name: read(release / name) for name in raw_names}
    for name, report in reports.items():
        passed(report, name, deb_sha)
    ctest = read(evidence / "ctest-qa.json")
    require(ctest.get("version") == VERSION and ctest.get("success") is True and not ctest.get("errors") and
            ctest.get("failed") == ctest.get("skipped") == ctest.get("blacklisted") == 0,
            "Complete effective CTest aggregation has not passed")
    require(ctest.get("effective_suite_count", len(ctest.get("suites", []))) == len(ctest.get("suites", [])),
            "CTest suite count is inconsistent")
    # Derive every count from the signed-by-hash effective raw log; never freeze
    # a target count such as 14/18/23/34 into a presumed acceptance result.
    needed = ("vision-gpu-tests", "vision-recording-tests", "vision-conversion-inference-tests",
              "vision-model-conversion-tests", "vision-evaluation-integration-tests",
              "vision-evaluation-worker-tests", "vision-evaluation-python-tests")
    effective = {name: suite(ctest, evidence, name) for name in needed}
    profiles = ctest.get("cuda_profiles", [])
    require(profiles, "Actual effective CUDA profiles are missing")
    for row in profiles:
        path = (evidence / row["file"]).resolve()
        require(path.is_relative_to(evidence.resolve()) and path.is_file() and sha(path) == row["sha256"], "CUDA profile changed")
        events = json.loads(path.read_text())
        count = sum(event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider" for event in events)
        require(count > 0 and count == row["cuda_node_events"], "Actual CUDA node count differs from retained profile")
    gpu = reports["packaged-gpu-qa.json"]
    for name, mode in (("onnx_cuda", "cuda"), ("pt_cuda", "cuda"), ("onnx_cpu", "cpu"), ("pt_cpu", "cpu")):
        case = gpu.get("evidence", {}).get(name, {})
        raw = case.get("report", {})
        require(case.get("success") is True and case.get("exit_code") == 0 and
                raw.get("success") is True and raw.get("actual_device") == raw.get("requested_device") == mode,
                f"Final packaged actual {name} evidence is incomplete")
    gpu_log = effective["vision-gpu-tests"][1]
    record_log = effective["vision-recording-tests"][1]
    checks = {"onnx_cuda_nodes": True,
        "pt_cuda": named_pass(gpu_log, "GpuBackendTests", "realModernAndLegacyPtColorAndGrayscale"),
        "packaged_onnx_cuda": gpu.get("checks", {}).get("packaged_onnx_cuda") is True,
        "packaged_pt_cuda": gpu.get("checks", {}).get("packaged_pt_cuda") is True,
        "cpu_regression": gpu.get("checks", {}).get("cpu_regression") is True,
        "grayscale_c1_c3": all(named_pass(gpu_log, "GpuBackendTests", method) for method in
            ("torchScriptC1AndC3GrayscaleDetectionAndClassification", "nativeGrayC1C3Classification", "onnxC1AndC3GrayPixelsMatchModelInput")),
        "stereo_recording": named_pass(record_log, "RecordingTests", "gpuRightEyeGrayscaleRecordingPreservesActualDevice")}
    require(all(checks.values()), "Seven required actual GPU acceptance checks are not all true")
    source = reports["source-bundle-qa.json"]
    binary_sha = source.get("details", {}).get("packaged_binary_sha256")
    require(isinstance(binary_sha, str) and len(binary_sha) == 64, "Source binding lacks packaged ELF SHA")
    require(source.get("complete_source_sha256") == sha(source_archive), "Source bundle report binds another archive")
    common = {"schema_version": 1, "version": VERSION, "success": True, "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    details = {"deb_sha256": deb_sha, "deb_bytes": deb.stat().st_size,
        "packaged_binary_sha256": binary_sha, "production_unstripped_binary_sha256": ctest["production_binary_sha256"],
        "ctest_report": "ctest-qa.json", "hardware": ctest.get("hardware"),
        "aggregation_only": "No new inference was executed by this report aggregation."}
    final_gpu = {**common, "checks": checks, "scope": "Actual effective GPU/recording suites, CUDA kernel traces and four final read-only DEB CPU/CUDA cases.",
                 "details": {**details, "packaged_report": "packaged-gpu-qa.json", "cuda_profiles": profiles}}
    qa = reports["qa-report.json"]
    smoke = qa.get("details", {}).get("smoke", {}).get("results", {})
    for name in ("conversion_onnx", "conversion_torchscript"):
        case = smoke.get(name, {})
        require(case.get("success") is True and case.get("exit_code") == 0 and
                case.get("report", {}).get("success") is True and case.get("report", {}).get("actual_inference") is True,
                f"Final read-only packaged {name} did not execute actual conversion and inference")
    conversion = {**common, "scope": "Complete effective native/Python conversion regressions and two actual final read-only package conversion/inference smokes.",
        "details": {**details, "full_conversion_regression": effective["vision-conversion-inference-tests"][0],
            "python_actual_conversion_regression": effective["vision-model-conversion-tests"][0],
            "readonly_final_package_smokes": {name: smoke[name] for name in ("conversion_onnx", "conversion_torchscript")}}}
    evaluation = reports["packaged-evaluation-qa.json"]
    require(evaluation.get("package_bound") is True and evaluation.get("read_only_application") is True and
            evaluation.get("eligible_for_release") is True and evaluation.get("scope") == "cpu_cuda_matrix",
            "Final evaluation QA is development-only, CPU-only or not bound to read-only package")
    actual_cases = ("onnx_cpu", "pt_cpu", "torchscript_cpu", "zero_result_id32_cpu", "onnx_cuda", "pt_cuda")
    comparisons = ("pt_torchscript_cpu", "onnx_cpu_cuda", "pt_cpu_cuda")
    for name in actual_cases:
        case = evaluation.get("cases", {}).get(name, {})
        require(case.get("success") is True and evaluation.get("checks", {}).get(name) is True and
                case.get("completed_images", 0) > 1 and case.get("source_files_verified", 0) > 0 and
                len(case.get("selected_exports", [])) == 3, f"Final actual evaluation case is incomplete: {name}")
        require(case.get("config", {}).get("resolved_device") == ("cuda" if "cuda" in name else "cpu"),
                f"Final evaluation case used wrong actual device: {name}")
    for name in comparisons:
        require(evaluation.get("comparisons", {}).get(name, {}).get("success") is True and
                evaluation.get("checks", {}).get(name) is True, f"Final equivalent-backend comparison failed: {name}")
    require(evaluation["cases"]["onnx_cuda"].get("cuda_node_events", 0) > 0, "Final evaluation did not record actual CUDA kernel events")
    final_evaluation = {**common, "scope": "Complete effective evaluation Qt/worker/Python suites plus six actual final-package pretrained-model cases and three same-weight comparisons.",
        "checks": {name: True for name in (*actual_cases, *comparisons)},
        "details": {**details, "packaged_report": "packaged-evaluation-qa.json",
            "regressions": [effective[name][0] for name in ("vision-evaluation-integration-tests", "vision-evaluation-worker-tests", "vision-evaluation-python-tests")],
            "dataset": evaluation.get("dataset"), "limitations": evaluation.get("limitations"),
            "comparisons": evaluation["comparisons"]}}
    # Validate SHA256SUMS against the exact two main assets before publishing a
    # manifest that also binds the checksum file itself.
    expected_assets = {deb.name: deb, source_archive.name: source_archive}
    sums = {}
    for line in (release / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)
            sums[name.lstrip("*")] = value
    require(set(sums) == set(expected_assets) and all(sums[name] == sha(path) for name, path in expected_assets.items()),
            "Final SHA256SUMS differs from actual immutable main assets")
    assets = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in
              {**expected_assets, "SHA256SUMS": release / "SHA256SUMS"}.items()}
    require(all(row["bytes"] < 2 * 1024**3 for row in assets.values()), "A GitHub asset exceeds the size limit")
    for name, value in (("gpu-qa.json", final_gpu), ("conversion-qa.json", conversion), ("evaluation-qa.json", final_evaluation)):
        write(release / name, value)
    write(evidence / "final-assets.json", {"schema_version": 1, "version": VERSION, "assets": assets})
    shutil.copy2(evidence / "ctest-qa.json", release / "ctest-qa.json")
    for path in sorted(release.iterdir()):
        if path.is_file() and not path.is_symlink() and (path.suffix in (".json", ".log") or path.name in ("SHA256SUMS", "SOURCE-SHA256SUMS")):
            destination = evidence / path.name
            require(not destination.is_symlink(), "Refusing to replace an evidence symlink")
            shutil.copy2(path, destination)
    return {"success": True, "deb_sha256": deb_sha, "gpu_checks": checks,
            "conversion_passed": {name: effective[name][0]["actual_passed"] for name in needed if "conversion" in name},
            "evaluation_passed": {name: effective[name][0]["actual_passed"] for name in needed if "evaluation" in name},
            "manual_assets": assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[4]
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    root = args.project.resolve()
    release = (args.release_dir or root / "output/releases" / VERSION).resolve()
    evidence = (args.evidence_dir or root / "docs/releases/evidence" / VERSION).resolve()
    print(json.dumps(aggregate(root, release, evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
