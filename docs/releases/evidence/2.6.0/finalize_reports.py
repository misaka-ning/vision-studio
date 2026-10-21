#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate completed 2.6.0 acceptance; never creates a presumed passing check.

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
from summarize_regressions import SUITES, code_hashes
from verify_dataset_session_package import audit_completed_case

VERSION = "2.6.0"


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


def data_rows(text, class_name, method):
    return re.findall(r"(?m)^\s*PASS\s*:\s*" + re.escape(class_name + "::" + method) + r"\(([^)]+)\)\s*$", text)


def current_feature_checks(effective, session, deb_sha):
    """Require actual new-version tests and reopen both final-package processes."""
    core = effective["vision-dataset-session-tests"][1]
    ui = effective["vision-dataset-session-ui-tests"][1]
    integration = effective["vision-evaluation-integration-tests"][1]
    methods = {
        "missing_json_creates_nothing": (core, "DatasetSessionTests", "missingFileDoesNotCreateAnything"),
        "non_file_json_preserved": (core, "DatasetSessionTests", "nonFileRecordReportsIoErrorWithoutDeletingIt"),
        "optional_metadata_and_missing_source_record": (core, "DatasetSessionTests", "optionalMetadataAndUnavailableSourceAreNotFilesystemValidation"),
        "oversized_record_preserved": (core, "DatasetSessionTests", "oversizedInputIsRejectedWithoutChangingIt"),
        "invalid_save_preserves_valid_record": (core, "DatasetSessionTests", "invalidSaveDoesNotReplaceValidRecord"),
        "failed_atomic_replace_keeps_original": (core, "DatasetSessionTests", "unwritableParentDoesNotFallBackToTruncatingExistingFile"),
        "concurrent_readers_see_only_complete_records": (core, "DatasetSessionTests", "concurrentReadersObserveCompleteOldOrNewRecords"),
        "failed_import_keeps_previous_success": (ui, "DatasetSessionUiTests", "failedImportKeepsLastSuccessfulRecord"),
        "cancelled_queue_keeps_previous_success": (ui, "DatasetSessionUiTests", "cancelledQueuePopulationKeepsLastSuccessfulRecord"),
        "picker_cancel_suppresses_pending_restore": (ui, "DatasetSessionUiTests", "openingAndCancellingPickerSuppressesPendingRestore"),
        "unique_fp_union_fn_threshold_navigation": (integration, "EvaluationIntegrationTests", "errorReviewNavigatesUniqueImagesAtCurrentThresholdWithoutInference"),
        "zero_errors_and_current_range_remain_empty": (integration, "EvaluationIntegrationTests", "zeroErrorReviewDoesNotInventErrorsOutsideCurrentImageRange"),
        "stopped_completed_prefix_error_review": (integration, "EvaluationIntegrationTests", "stoppingPreservesCompletedPrefixAndReview"),
    }
    checks = {key: named_pass(text, name, method) for key, (text, name, method) in methods.items()}
    expected_rows = {
        "store_three_splits": (core, "DatasetSessionTests", "roundTripPreservesOriginalSourceAndSplit", {"train", "val", "test"}),
        "seven_malformed_records_preserved": (core, "DatasetSessionTests", "invalidJsonIsPreserved",
            {"truncated", "array", "unknown-schema", "wrong-field-type", "relative-source", "bad-split", "relative-metadata"}),
        "six_source_split_metadata_restore_rows": (ui, "DatasetSessionUiTests", "successfulImportRestoresOriginalSourceSplitAndMetadata",
            {f"{source}-{split}" for source in ("directory", "yaml") for split in ("train", "val", "test")}),
        "four_invalid_startup_rows_recover": (ui, "DatasetSessionUiTests", "missingCorruptAndUnavailableSessionsRecoverSafely",
            {"missing-json", "corrupt-json", "unknown-schema", "missing-source"}),
        "manual_selection_wins_both_orders": (ui, "DatasetSessionUiTests", "explicitManualSelectionWinsBeforeDeferredRestore",
            {"manual-then-restore", "restore-then-manual-without-event-loop"}),
    }
    actual_rows = {}
    for key, (text, name, method, expected) in expected_rows.items():
        rows = data_rows(text, name, method)
        checks[key] = len(rows) == len(expected) and set(rows) == expected
        actual_rows[key] = rows
    atomic = re.findall(r"dataset_session_atomic_read_observations\s+(\d+)\s+committed_replacements\s+(\d+)", core)
    require(len(atomic) == 1 and int(atomic[0][0]) > 0 and int(atomic[0][1]) == 120,
            "Missing actual concurrent-reader atomic-save observations")
    restores = re.findall(r"dataset_session_restore_evidence\s+(yaml|directory)\s+(train|val|test)\s+images\s+(\d+)\s+metadata_excluded\s+(\d+)\s+inference_requests\s+(\d+)\s+heartbeat_ticks\s+(\d+)\s+max_gap_ms\s+(\d+)", ui)
    require(len(restores) == 6 and {(r[0], r[1]) for r in restores} ==
            {(source, split) for source in ("yaml", "directory") for split in ("train", "val", "test")} and
            all(int(images) > 0 and int(excluded) == 1 and int(starts) == 0 and int(ticks) > 0 and int(gap) < 1000
                for _, _, images, excluded, starts, ticks, gap in restores),
            "Actual six-case restore/no-inference/heartbeat observations disagree")
    cancelled = re.findall(r"dataset_session_cancel_evidence\s+partial_count\s+(\d+)\s+stop_return_ms\s+(\d+)", ui)
    require(len(cancelled) == 1 and 0 < int(cancelled[0][0]) < 1500 and int(cancelled[0][1]) < 500,
            "Missing actual cancelled-session population observations")
    review = re.findall(r"evaluation_error_review_evidence\s+threshold_0\.7_unique_errors\s+(\d+)\s+threshold_0\.9_unique_errors\s+(\d+)\s+completed_images\s+(\d+)\s+inference_requests\s+(\d+)", integration)
    require(len(review) == 1 and tuple(map(int, review[0])) == (2, 3, 4, 0),
            "Actual four-image threshold-linked unique-error observations disagree")
    passed(session, "packaged-dataset-session-qa.json", deb_sha)
    expected_package_checks = {"final_deb_binding_and_readonly", "yaml_two_process_restore",
        "directory_two_process_restore", "public_inputs_unchanged", "packaged_application_unchanged",
        "restore_has_no_inference_history_or_report_writes"}
    require(session.get("package_bound") is True and session.get("read_only_application") is True and
            set(session.get("checks", {})) == expected_package_checks and
            all(value is True for value in session["checks"].values()) and
            set(session.get("cases", {})) == {"yaml", "directory"},
            "Final package did not complete both distinct-process restore cases")
    package_rows = {}
    for kind, row in session["cases"].items():
        require(row.get("success") is True and row.get("source_kind") == kind,
                "Missing successful packaged source/restore case")
        screenshot = audit_completed_case(row)
        require(screenshot == row["screenshot"], "Packaged restore screenshot changed")
        package_rows[kind] = {"completed_images": row["completed_images"],
                              "evaluation_command": row["evaluation_command"], "restore_command": row["restore_command"],
                              "evaluation_smoke_sha256": row["evaluation_smoke_sha256"],
                              "restore_smoke_sha256": row["restore_smoke_sha256"], "screenshot": screenshot}
        checks["packaged_" + kind + "_separate_process_restore"] = True
    require(all(checks.values()), "New dataset-session/error-review native or packaged assertions did not all pass")
    return checks, {"data_rows": actual_rows, "atomic_save_observation": atomic[0],
                    "restore_observations": restores, "cancelled_population_observation": cancelled[0],
                    "error_review_observation": review[0], "package_cases": package_rows,
                    "methods": {key: {"class": name, "method": method}
                                for key, (_, name, method) in methods.items()}}


def aggregate(project, release, evidence):
    deb = release / f"vision-studio_{VERSION}-1_amd64.deb"
    source_archive = release / f"vision-studio-{VERSION}-complete-source.tar.xz"
    require(deb.is_file() and source_archive.is_file(), "Final DEB and complete corresponding source must exist")
    deb_sha = sha(deb)
    raw_names = ("qa-report.json", "install-qa.json", "source-bundle-qa.json", "packaged-gpu-qa.json", "packaged-evaluation-qa.json",
                 "folder-package-qa.json", "packaged-private-precheck.json", "packaged-dataset-session-qa.json")
    reports = {name: read(release / name) for name in raw_names}
    for name, report in reports.items():
        passed(report, name, deb_sha)
    folder = reports["folder-package-qa.json"]
    expected_folder_checks = ("final_deb_binding_and_readonly", "unchanged_public_val_fixture_and_readonly",
        "real_onnx_cpu_four_images", "directory_origin_preserved", "unique_conventional_yaml_selected",
        "stale_root_and_absolute_list_relocated", "all_91_threshold_points", "selected_png_json_csv_exports",
        "public_and_fixture_inputs_unchanged", "packaged_application_unchanged")
    require(folder.get("package_bound") is True and folder.get("read_only_application") is True and
            all(folder.get("checks", {}).get(name) is True for name in expected_folder_checks),
            "Final read-only package did not pass the folder-specific regression")
    private = reports["packaged-private-precheck.json"]
    require(private.get("selected_images", 0) > 0 and private.get("root_relocated") is True and
            private.get("manifest_matches_source_precheck") is True and
            private.get("original_yaml_sha256_unchanged") is True and private.get("dataset_writes") is False and
            all(private.get("source_helpers_match_staged", {}).get(name) is True for name in
                ("evaluation_worker.py", "evaluation_data.py", "evaluation_metrics.py")),
            "Private directory precheck is incomplete or inputs differ; never substitute it for inference accuracy")
    ctest = read(evidence / "ctest-qa.json")
    require(ctest.get("version") == VERSION and ctest.get("success") is True and not ctest.get("errors") and
            ctest.get("failed") == ctest.get("skipped") == ctest.get("blacklisted") == 0,
            "Complete effective CTest aggregation has not passed")
    require(ctest.get("effective_suite_count", len(ctest.get("suites", []))) == len(ctest.get("suites", [])),
            "CTest suite count is inconsistent")
    # Derive every count from the signed-by-hash effective raw log; never freeze
    # a target count such as 14/18/23/34 into a presumed acceptance result.
    require(len(ctest.get("suites", [])) == len(SUITES) == 20 and
            {row.get("name") for row in ctest["suites"]} == set(SUITES),
            "Final 2.6.0 requires exactly all 20 effective native/Python suites")
    require(ctest.get("source_code_input_sha256") == code_hashes(project) ==
            reports["source-bundle-qa.json"].get("production_worktree_input_sha256") and
            reports["source-bundle-qa.json"].get("production_worktree_matches_frozen_source") is True,
            "Tested worktree and final frozen package production inputs differ")
    needed = SUITES
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
    feature_methods = {
        "unified_directory_and_yaml_picker": ("vision-evaluation-panel-tests", "EvaluationPanelTests", "sameDatasetPickerAcceptsDirectoryAndYaml"),
        "invalid_source_inline": ("vision-evaluation-panel-tests", "EvaluationPanelTests", "unifiedPickerRejectsInvalidSourceInline"),
        "popup_dismissal_geometry_busy_lock": ("vision-evaluation-panel-tests", "EvaluationPanelTests", "settingsPopupKeepsToolbarGeometryAndDismissesWithoutConfigurationChanges"),
        "complete_long_error_accessible": ("vision-evaluation-panel-tests", "EvaluationPanelTests", "longImportErrorsAreReadableInPopupWithoutGrowingToolbar"),
        "full_workbench_pages_preserve_configuration": ("vision-evaluation-integration-tests", "EvaluationIntegrationTests", "fullWidthTabsChangeOnlyViewAndShareDetectionRange"),
        "lazy_queue_heartbeat_pixels_stale_cancel": ("vision-batch-responsiveness-tests", "BatchResponsivenessTests", "largeQueueLoadsOnlyVisibleThumbnailsAndCancelsStalePreviews"),
    }
    ui_checks = {name: named_pass(effective[suite_name][1], class_name, method)
                 for name, (suite_name, class_name, method) in feature_methods.items()}
    integration_log = effective["vision-evaluation-integration-tests"][1]
    geometry_rows = re.findall(r"(?m)^\s*PASS\s*:\s*EvaluationIntegrationTests::compactCommandBarPreservesCanvasAndCollapsesDatasetDetails\(([^)]+)\)\s*$", integration_log)
    expected_geometry = {f"{width}-{device}-layout-only" for width in (1260, 1380) for device in ("cpu", "cuda")}
    ui_checks["popup_workbench_geometry_all_four_layout_rows"] = len(geometry_rows) == 4 and set(geometry_rows) == expected_geometry
    require(all(ui_checks.values()), "2.6.0 unified picker/popup/full-page/lazy-queue feature assertions did not all pass")
    new_checks, new_observations = current_feature_checks(effective, reports["packaged-dataset-session-qa.json"], deb_sha)
    observations = re.findall(r"lazy_queue_evidence\s+standard_yolo_dataset\s+(\d+)\s+partial_count\s+(\d+)\s+population_controls_locked\s+(true|false)\s+stop_at_count\s+(\d+)\s+stop_dispatch_ms\s+(\d+)\s+initial_loaded_thumbnails\s+(\d+)\s+heartbeat_ticks\s+(\d+)\s+maximum_gap_ms\s+(\d+)\s+rapid_selection_final_index\s+(\d+)\s+clear_cancelled_jobs\s+(\d+)", effective["vision-batch-responsiveness-tests"][1])
    require(len(observations) == 1, "Missing or ambiguous actual lazy-queue observation log")
    raw_observation = observations[0]
    image_count, partial_count = map(int, raw_observation[:2])
    controls_locked = raw_observation[2] == "true"
    stopped_at, stop_dispatch_ms, loaded, ticks, maximum_gap_ms, final_index, remaining_jobs = map(int, raw_observation[3:])
    require(image_count > 50 and 0 < partial_count < image_count and controls_locked and
            0 < stopped_at < image_count and 0 <= stop_dispatch_ms < 250 and
            0 < loaded <= 128 and loaded < image_count and ticks >= 10 and
            maximum_gap_ms < 500 and 0 <= final_index < image_count and remaining_jobs == 0,
            "Actual lazy-queue observations disagree with feature acceptance bounds")
    ui = {**common, "checks": ui_checks,
        "scope": "Effective production-native unified picker, popup, full-workbench-page and asynchronous queue tests. DEB hash is traceability binding; this is not a separate installed-package UI run.",
        "details": {**details, "feature_methods": {name: {"suite": row[0], "class": row[1], "method": row[2]} for name, row in feature_methods.items()},
            "layout_rows": geometry_rows,
            "lazy_queue_observations": {"fixture_images": image_count, "initial_loaded_thumbnails": loaded,
                "dataset_format": "standard_yolo_dataset", "observed_partial_population_count": partial_count,
                "population_controls_locked": controls_locked, "stopped_at_partial_count": stopped_at,
                "stop_dispatch_ms": stop_dispatch_ms,
                "heartbeat_ticks": ticks, "maximum_gap_ms": maximum_gap_ms,
                "rapid_selection_final_index": final_index, "remaining_jobs_after_clear": remaining_jobs},
            "limitations": ["Synthetic small-image fixture validates scheduling and stale-result rejection, not deployment accuracy.",
                            "Heartbeat values are this execution's observations, not a cross-hardware performance guarantee."]}}
    final_gpu = {**common, "checks": checks, "scope": "Actual effective GPU/recording suites, CUDA kernel traces and four final read-only DEB CPU/CUDA cases.",
                 "details": {**details, "packaged_report": "packaged-gpu-qa.json", "cuda_profiles": profiles}}
    session_feature = {**common, "checks": new_checks,
        "scope": "Actual native JSON/session/review regressions and two separate final-package Qt process pairs; raw reports and logs re-opened and hashed during aggregation.",
        "details": {**details, "packaged_report": "packaged-dataset-session-qa.json", **new_observations,
                    "limitations": ["Native error-review cases use deterministic cached four-image candidates to verify FP/FN union, navigation and threshold changes without new inference.",
                                    "Packaged restart uses four public COCO8 Val images; six native source/split cases cover single-class review metadata and train/test restoration."]}}
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
    for name, value in (("gpu-qa.json", final_gpu), ("conversion-qa.json", conversion), ("evaluation-qa.json", final_evaluation),
                        ("ui-queue-qa.json", ui), ("dataset-session-qa.json", session_feature)):
        write(release / name, value)
    write(evidence / "final-assets.json", {"schema_version": 1, "version": VERSION, "assets": assets})
    shutil.copy2(evidence / "ctest-qa.json", release / "ctest-qa.json")
    for path in sorted(release.iterdir()):
        if path.is_file() and not path.is_symlink() and (path.suffix in (".json", ".log") or path.name in ("SHA256SUMS", "SOURCE-SHA256SUMS")):
            destination = evidence / path.name
            require(not destination.is_symlink(), "Refusing to replace an evidence symlink")
            shutil.copy2(path, destination)
    return {"success": True, "deb_sha256": deb_sha, "gpu_checks": checks,
            "ui_queue_checks": ui_checks,
            "dataset_session_and_review_checks": new_checks,
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
