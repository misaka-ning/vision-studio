# SPDX-License-Identifier: AGPL-3.0-only
"""Static public projection allowlist; no QA reports or prior results are copied."""
import hashlib
import json
from pathlib import PurePosixPath
import re
VERSION = "2.10.1"
SHA = re.compile(r"^[a-f0-9]{64}$")
ASSETS = {"vision-studio_2.10.1-1_amd64.deb", "vision-studio-2.10.1-complete-source.tar.xz", "SHA256SUMS"}
DECISION_CHECKS = frozenset({
    'required_widgets_exist', 'four_pages_and_order', 'decision_controls_share_threshold_page',
    'visible_at_1260x820', 'visible_at_1366x900', 'visible_at_1260x820_precision', 'visible_at_1260x820_fpr',
    'visible_at_1366x900_precision', 'visible_at_1366x900_fpr', 'splitter_has_two_noncollapsed_sections',
    'splitter_resizes_chart_and_table', 'real_full_val_recommendation_available',
    'current_and_recommended_markers_are_distinct', 'configuration_unchanged_before_recommendation',
    'recommendation_click_syncs_curve_table_and_visible_row', 'recommendation_does_not_adopt_or_reinfer',
    'table_selection_syncs_current_point_and_visible_row', 'real_review_button_enabled',
    'review_button_opens_real_error_queue', 'real_return_button_visible',
    'return_preserves_threshold_page_strategy_and_splitter', 'review_does_not_adopt_reinfer_or_rewrite_report',
    'explicit_apply_allowed_for_verified_native_report', 'only_explicit_apply_updates_detection_profile',
    'original_report_preserved_and_initial_export_threshold_restored'})
GEOMETRY_GROUPS = frozenset(size + suffix for size in ('1260x820', '1366x900') for suffix in ('', '_precision', '_fpr'))
NATIVE_METHODS = ('decisionSidebarSharesCurveTableAndIndependentApply',
    'recommendationScrollsLongThresholdTableAndReviewIsExplicit',
    'reportTabsAndPlotsDoNotChangeAnalysisOrRunConfiguration', 'responsiveDiagnosticLayouts')

DECISION_AUDIT_PARAMETERS = ('report', 'project', 'native')
DECISION_AUDIT_SCOPE = 'reopened_actual_final_readonly_package_and_native_originals'
DECISION_TEXT_WIDGETS = frozenset({'evaluationRecommendationHint', 'evaluationCounts',
    'evaluationMetric2', 'evaluationMetric3', 'evaluationMetric4', 'evaluationMetric5'})
DECISION_WIDGETS = frozenset({'evaluationAnalysisThresholdSpin', 'evaluationThresholdCurve',
    'evaluationThresholdTable', 'evaluationDecisionSidebar', 'evaluationRecommendationStrategyCombo',
    'evaluationUseRecommendationButton', 'evaluationReviewErrorsButton', 'evaluationApplyThresholdButton'}) | DECISION_TEXT_WIDGETS
DECISION_AUTOMATIC_GEOMETRY_CHECKS = frozenset({'six_groups_have_stable_observations',
    'initial_rows_fully_visible', 'label_contents_and_height_for_width_checked', 'ancestor_clipping_checked'})
# Only role names from this static set may identify local originals publicly.
# These are whole-report bytes/SHA commitments; report contents stay local.
REQUIRED_WHOLE_ORIGINAL_ROLES = frozenset({'source-freeze.json', 'production-inputs-before.json',
    'source-bundle-qa.json', 'ctest-qa.json', 'final-assets.json'})
OPTIONAL_BUILD_ORIGINAL_ROLES = frozenset({'initial-source-freeze.json', 'native-ui-object-link.json',
    'SOURCE-INVENTORY.json', 'SOURCE-SHA256SUMS', 'SHA256SUMS', 'packaged-decision-layout-qa.json',
    'decision-layout-final-gate.log'})

COUNTS = {"effective_suite_count", "qt_tests_passed", "python_tests_passed", "failed", "skipped", "blacklisted", "git_archive_files_verified"}
LEGACY_CHECK_FIELDS = frozenset({
    'actual_same_version_prerequisites_final_deb_elf_code_and_readonly',
    'actual_yaml_font_is_monospaced_after_qss',
    'all_91_threshold_points',
    'all_eleven_available_view_pngs_fully_decoded',
    'append_preserves_originals_gt_and_session',
    'appendedTemporaryImagesPreserveDatasetCacheTruthAndSelectedExports',
    'blueprintRowsPreserveAllOriginalTypesAndOrder',
    'busy_drop_does_not_mutate_queue',
    'cancel_preserves_dataset',
    'cancelled_queue_keeps_previous_success',
    'complete_long_error_accessible',
    'concurrentExportsOnlyOneWriterCreatesDestination',
    'concurrent_readers_see_only_complete_records',
    'cpu_regression',
    'currentAndTailOfTemporaryItemDoNotStartEvaluation',
    'datasetConflictHonorsReplaceAppendAndCancel',
    'default_missing_key_actual_process_matches_preference_without_inference_or_data_writes',
    'directory_drop_loads_dataset',
    'directory_origin_preserved',
    'directory_two_process_restore',
    'disabled_actual_process_matches_preference_without_inference_or_data_writes',
    'enabled_actual_process_matches_preference_without_inference_or_data_writes',
    'evaluationRangesOnlyIncludeOriginalLabeledSamples',
    'failed_atomic_replace_keeps_original',
    'failed_import_keeps_previous_success',
    'final_deb_binding_and_readonly',
    'final_readonly_deb_three_models_and_independent_static_oracle',
    'final_readonly_deb_twelve_stage_actual_drop_workflow',
    'four_invalid_startup_rows_recover',
    'full_evaluation_excludes_temporaries',
    'full_workbench_pages_preserve_configuration',
    'grayscale_c1_c3',
    'guide_evaluation_anchor_actually_scrolls',
    'guide_immediately_above_more_click_targets_preserved',
    'guide_visible_immediately_above_more',
    'invalidMixedDropDoesNotPartiallyReplaceValidInput',
    'invalid_save_preserves_valid_record',
    'invalid_source_inline',
    'largeTemporaryDropStaysResponsiveAndDiscardsStaleImageReplies',
    'lazy_queue_heartbeat_pixels_stale_cancel',
    'looseImagesKeepMimeOrderAndOnlyLiveInTheQueue',
    'maliciousCheckpointCannotExecuteNativePythonToShowYaml',
    'manual_selection_wins_both_orders',
    'missing_json_creates_nothing',
    'mixed_types_rejected',
    'modernScalesAndLegacyAnchorsArePreserved',
    'nativeDropAcceptsDatasetAndVideoSources',
    'nativePreviewChildDropCancelsPendingAutomaticSessionRestore',
    'native_clicks_persist_off_and_on_in_isolated_preferences',
    'native_tab_focus_follows_sidebar_visual_order',
    'newFileExportNeverOverwritesSourceExistingOrSymlink',
    'non_file_json_preserved',
    'onnx_cpu',
    'onnx_cpu_cuda',
    'onnx_cuda',
    'onnx_cuda_nodes',
    'onnx_unavailable_yaml_mode_hidden',
    'onnx_yaml_mode_hidden_and_unselectable',
    'optional_metadata_and_missing_source_record',
    'original_genuine_session_case_unchanged',
    'original_public_inputs_unchanged',
    'original_public_models_unchanged',
    'oversized_record_preserved',
    'packaged_application_unchanged',
    'packaged_directory_separate_process_restore',
    'packaged_onnx_cuda',
    'packaged_pt_cuda',
    'packaged_yaml_separate_process_restore',
    'picker_cancel_suppresses_pending_restore',
    'popup_dismissal_geometry_busy_lock',
    'popup_workbench_geometry_all_four_layout_rows',
    'pt_architecture_values_and_types_equal_static_original',
    'pt_cpu',
    'pt_cpu_cuda',
    'pt_cuda',
    'pt_torchscript_cpu',
    'pt_yaml_available_and_visible',
    'public_and_fixture_inputs_unchanged',
    'public_inputs_unchanged',
    'read_only_editor_and_line_number_gutter_visible',
    'real_models_yaml_available_views_and_hidden_unsupported_modes',
    'real_onnx_cpu_four_images',
    'replace_clears_dataset_context_without_saving_temporary_session',
    'restore_has_no_inference_history_or_report_writes',
    'runningDetectionRejectsDropWithoutReplacingItsQueue',
    'samePathReplacementInvalidatesAndRebuildsYamlCache',
    'selected_png_json_csv_exports',
    'seven_malformed_records_preserved',
    'six_source_split_metadata_restore_rows',
    'sourceChangedDuringParsingCannotCommitReadyOrYaml',
    'stale_root_and_absolute_list_relocated',
    'startup_preference_defaults_on',
    'startup_restore_preference_cancels_already_deferred_entry',
    'startup_restore_preference_defaults_persists_and_keeps_json',
    'stereo_recording',
    'stopped_completed_prefix_error_review',
    'stoppingPartialTemporaryAppendRestoresOriginalQueueAndSession',
    'store_three_splits',
    'strict_schema_and_resource_limits',
    'temporary_real_cpu_inference_without_gt',
    'temporary_selected_export',
    'temporary_suffix_and_single_evaluation_blocked',
    'torchscript_cpu',
    'unchanged_public_val_fixture_and_readonly',
    'unified_directory_and_yaml_picker',
    'unique_conventional_yaml_selected',
    'unique_fp_union_fn_threshold_navigation',
    'video_drop_selected',
    'yamlBridgeRejectsNonprimitiveCyclesAndLimitsWithoutTouchingCachedGraph',
    'yamlFastSwitchClearsOldDocumentAndRejectsStaleCallbacks',
    'yamlOneDarkEditorHasReadableTokensAndScrollingLineNumbers',
    'yaml_drop_loads_dataset',
    'yaml_two_process_restore',
    'yolov5n.onnx_available_views_cache_export_and_original_bytes',
    'yolov5n.pt_available_views_cache_export_and_original_bytes',
    'yolov8n.pt_available_views_cache_export_and_original_bytes',
    'zero_errors_and_current_range_remain_empty',
    'zero_result_id32_cpu',
})
EVALUATION_FEATURE_FIELDS = frozenset({
    'advanced_csv_latex_match_saved_metrics',
    'cpu_cuda_four_actual_cases_each',
    'final_deb_elf_helpers_readonly_and_frozen',
    'native_batch_one_three_and_tail_equivalent',
    'native_pt_batch_tail_stop_and_adoption',
    'native_qt_diagnostics_scene_report_and_reanalysis',
    'nine_charts_three_formats_reopened_and_hashed',
    'offline_migrated_snapshots_match_all_91_fresh_points',
    'python_metrics_reporting_snapshot_and_batch_all_passed',
    'reviewed_and_unknown_scene_scopes_preserved',
    'snapshot_three_qt_reserved_files_with_helper_log',
    'source_reports_and_review_bytes_unchanged',
})
EXTERNAL_IMPORT_FIELDS = frozenset({
    'original_thirteen_python_regressions_visible',
    'private_historical_1223_readonly_receipt',
    'qt_failed_import_restores_review_and_preview',
    'qt_import_without_model_blocks_unknown_profile_adoption',
    'qt_model_identity_preserves_adopted_profile',
    'qt_original_report_export_immutable',
    'qt_unknown_metadata_and_export_remain_unknown',
})
PACKAGED_EXTERNAL_IMPORT_FIELDS = frozenset({
    'all_sample_json_sha_and_manifest_image_hashes',
    'apply_profile_blocked_at_report_and_request',
    'expected_artifacts_valid',
    'final_deb_elf_helpers_runtime_readonly_binding',
    'final_package_application_unchanged',
    'frozen_clean_source_helpers_match',
    'model_dataset_original_ids_preserved',
    'packaged_python_helper_actual_process',
    'qt_reserved_three_files_accepted',
    'rebuilt_metrics_match_source_snapshot',
    'source_four_files_unchanged',
    'source_raw_copies_match',
    'source_runtime_arguments_nms_floor_maxdet_preserved',
    'source_scope_counts_preserved',
    'source_snapshot_matches_prior_verified_import',
    'unknown_inference_fields_not_invented',
})

MODEL_LIBRARY_DROP_FIELDS = frozenset({'real_pt_metadata_and_structure_loaded', 'real_onnx_metadata_loaded', 'native_modelLibraryExternalDropsAppendAtomicallyAndSelectOnce', 'real_torchscript_drop_and_metadata_loaded', 'external_files_referenced_without_copy_or_move', 'non_url_drag_ignored', 'canonical_duplicates_preserve_order_and_metadata', 'multi_child_drop_preserves_url_order_and_last_selection', 'internal_reorder_preserves_selection_and_metadata', 'selected_result_export_matches_global_model', 'context_menu_four_actions_preserved', 'directory_empty_and_missing_files_rejected', 'native_modelLibraryDropsRejectInvalidBatchesWithoutMutations', 'preferences_persist_actual_state', 'mixed_invalid_batch_is_atomic', 'viewport_copy_drop_accepted', 'native_modelLibraryInternalMovesPreserveGlobalSelectionAndPersistAfterRestart', 'original_and_external_files_unchanged', 'duplicate_preserves_structure_and_inspection_cache', 'move_and_link_only_rejected', 'single_page_drop_selects_global_model', 'native_modelLibraryDialogCancelKeepsLibraryAndSelection', 'actual_busy_drop_preserves_library_and_active_model', 'remote_urls_rejected', 'native_busyModelLibraryDropsCannotReplaceCurrentTask', 'selected_onnx_real_cpu_inference'})

APPROVED_BOOLEAN_FIELDS = {group: LEGACY_CHECK_FIELDS for group in (
    "checks", "gpu_checks", "ui_queue_checks", "dataset_session_and_review_checks",
    "preview_input_checks", "model_yaml_checks")}
APPROVED_BOOLEAN_FIELDS.update({
    "evaluation_features_checks": EVALUATION_FEATURE_FIELDS,
    "external_import_checks": EXTERNAL_IMPORT_FIELDS,
    "packaged_external_import_checks": PACKAGED_EXTERNAL_IMPORT_FIELDS,
    "model_library_drop_checks": MODEL_LIBRARY_DROP_FIELDS,
})
# Dedicated final reports may expose the same approved check dictionary under
# their conventional top-level `checks`; no other fields are granted by that.
APPROVED_BOOLEAN_FIELDS["checks"] |= (EVALUATION_FEATURE_FIELDS | EXTERNAL_IMPORT_FIELDS | PACKAGED_EXTERNAL_IMPORT_FIELDS | MODEL_LIBRARY_DROP_FIELDS)
APPROVED_COUNT_FIELDS = {group: frozenset() for group in APPROVED_BOOLEAN_FIELDS}
CHECK_GROUPS = frozenset(APPROVED_BOOLEAN_FIELDS)
SUITE_NAMES = frozenset({
    'vision-core-tests',
    'vision-inference-delivery-tests',
    'vision-recording-tests',
    'vision-pt-tests',
    'vision-gpu-tests',
    'vision-ui-tests',
    'vision-batch-responsiveness-tests',
    'vision-scope-filter-ui-tests',
    'vision-modelviewer-tests',
    'vision-conversion-ui-tests',
    'vision-conversion-inference-tests',
    'vision-model-conversion-tests',
    'vision-netron-server-tests',
    'vision-gpu-runtime-tests',
    'vision-evaluation-worker-tests',
    'vision-evaluation-panel-tests',
    'vision-evaluation-python-tests',
    'vision-evaluation-integration-tests',
    'vision-dataset-session-tests',
    'vision-dataset-session-ui-tests',
    'vision-model-yaml-tests',
    'vision-input-drop-ui-tests',
    'vision-evaluation-reporting-tests',
    'vision-evaluation-snapshot-tests',
    'vision-pt-batch-python-tests',
    'vision-evaluation-import-tests',
})
PUBLIC_KEY_ALIASES = {"private_historical_1223_readonly_receipt": "private_historical_readonly_receipt",
    "native_modelLibraryInternalMovesPreserveGlobalSelectionAndPersistAfterRestart": "native_window_recreation_preserves_model_order_and_selection"}
ARCHIVES = frozenset({"evaluation-artifacts.zip", "folder-artifacts.zip", "namespace-artifacts.zip",
    "dataset-session-artifacts.zip", "preview-drop-artifacts.zip", "model-yaml-artifacts.zip"})
PUBLIC_REPORT_NAMES = frozenset({"qa-report.json", "install-qa.json", "source-bundle-qa.json", "ctest-qa.json",
    "gpu-qa.json", "packaged-gpu-qa.json", "evaluation-qa.json", "packaged-evaluation-qa.json",
    "folder-package-qa.json", "packaged-private-precheck.json", "packaged-dataset-session-qa.json",
    "dataset-session-qa.json", "packaged-model-yaml-qa.json", "model-yaml-qa.json",
    "packaged-preview-drop-qa.json", "preview-input-qa.json", "packaged-startup-preference-qa.json",
    "conversion-qa.json", "ui-queue-qa.json", "final-assets.json", "finalize-reports.json",
    "evaluation-features-qa.json", "packaged-evaluation-features-qa.json", "packaged-external-import-qa.json", "packaged-model-library-drop-qa.json", "model-library-drop-qa.json"})

WHOLE_ORIGINAL_ROLES = (PUBLIC_REPORT_NAMES | ARCHIVES | REQUIRED_WHOLE_ORIGINAL_ROLES |
    OPTIONAL_BUILD_ORIGINAL_ROLES | frozenset(name.removesuffix('.zip') + '.json' for name in ARCHIVES))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()



def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON fields are not valid projection input")
            result[key] = value
        return result
    def constant(value):
        raise RuntimeError("Non-finite JSON constants are not valid projection input")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def safe_relative(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name,
            "Unsafe relative evidence filename")
    require(not re.search(r"(?:^|/)(?:home|tmp|root|proc|sys|Users)(?:/|$)", name),
            "Host-oriented relative evidence filename")
    return name


def approved_checks(group, value):
    """No free-form keys, nested details, strings or unapproved numeric fields."""
    if not isinstance(value, dict):
        return {}, {}
    booleans, counts = {}, {}
    for key in APPROVED_BOOLEAN_FIELDS[group]:
        if isinstance(value.get(key), bool):
            booleans[PUBLIC_KEY_ALIASES.get(key, key)] = value[key]
    for key in APPROVED_COUNT_FIELDS[group]:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            counts[PUBLIC_KEY_ALIASES.get(key, key)] = item
    return booleans, counts


def public_label(relative, digest):
    """Unknown local filenames are private too; identify their bytes opaquely."""
    if relative in PUBLIC_REPORT_NAMES or relative in ARCHIVES or relative in ASSETS:
        return relative
    if relative.startswith("qa-full/") and relative.removeprefix("qa-full/").removesuffix(".log") in SUITE_NAMES and relative.endswith(".log"):
        return relative
    return "original-sha256-" + digest


def report_projection(value):
    if not isinstance(value, dict) or value.get("version", VERSION) != VERSION:
        return None
    result = {"public_projection": True, "not_a_finalizer_input": True}
    if value.get("schema_version") == 1 and not isinstance(value["schema_version"], bool):
        result["original_schema_version"] = value["schema_version"]
    if value.get("version") == VERSION:
        result["version"] = VERSION
    if isinstance(value.get("success"), bool):
        result["original_success"] = value["success"]
    if isinstance(value.get("errors"), list):
        result["original_error_count"] = len(value["errors"])
    for key in COUNTS:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    checks, check_counts = {}, {}
    details = value.get("details") if isinstance(value.get("details"), dict) else {}
    for key in CHECK_GROUPS:
        raw = value.get(key, details.get(key))
        selected, selected_counts = approved_checks(key, raw)
        if selected:
            checks[key] = selected
        if selected_counts:
            check_counts[key] = selected_counts
    if checks:
        result["boolean_checks"] = checks
    if check_counts:
        result["approved_check_counts"] = check_counts
    suites = value.get("suites")
    if isinstance(suites, list):
        selected = []
        for row in suites:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] not in SUITE_NAMES:
                continue
            item = {"name": row["name"]}
            for field in ("success", "skipped"):
                if isinstance(row.get(field), bool):
                    item[field] = row[field]
            if isinstance(row.get("python_tests"), int) and not isinstance(row["python_tests"], bool) and row["python_tests"] >= 0:
                item["python_tests"] = row["python_tests"]
            if isinstance(row.get("qt_totals"), list):
                item["qt_totals"] = [{k: count for k, count in totals.items() if k in {"passed", "failed", "skipped", "blacklisted"} and isinstance(count, int) and not isinstance(count, bool) and count >= 0} for totals in row["qt_totals"] if isinstance(totals, dict)]
            selected.append(item)
        if selected:
            result["suite_outcomes"] = selected
    # Do not publish a source filename map that might contain a user model or
    # untracked local helper. Its exact original bytes remain committed below.
    fingerprints = value.get("source_code_input_sha256")
    if isinstance(fingerprints, dict) and all(isinstance(name, str) and isinstance(digest, str) and SHA.fullmatch(digest) for name, digest in fingerprints.items()):
        result["original_source_fingerprint_canonical_sha256"] = sha(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode())
    # No full original report, arbitrary strings, argv, host fields or nested details.
    return result if len(result) > 2 else None



# Refactor release additions. Only these fixed new booleans/suites are public.
REFACTOR_CHECKS = frozenset({
    "all_refactor_suites_executed_without_skips", "retained_required_assertions",
    "all_512_permission_combinations", "actual_stream_eof_stop_shutdown_failure",
    "actual_window_shutdown_assertions", "dependency_and_ownership_boundary",
    "tested_source_matches_frozen_package", "production_binary_unchanged"})
REBUILD_CHECKS = frozenset({
    "actual_complete_attachment_application_source", "extracted_source_members_unchanged",
    "fresh_independent_release_build", "exact_rebuilt_application_version",
    "real_rebuilt_onnx_cpu_inference", "actual_rebuilt_png_json_csv"})
REFACTOR_SUITE_NAMES = frozenset({
    "vision-app-services-tests", "vision-gpu-environment-service-tests",
    "vision-model-session-controller-tests", "vision-detection-controller-tests",
    "vision-input-session-controller-tests", "vision-evaluation-controller-tests",
    "vision-operation-coordinator-tests", "vision-recording-controller-tests",
    "vision-workbench-configuration-tests", "vision-image-result-store-tests",
    "vision-result-delivery-tests", "vision-workbench-page-tests",
    "vision-workspace-pages-tests", "vision-simple-pages-tests",
    "vision-component-shutdown-tests", "vision-runtime-setup-shutdown-tests",
    "vision-workspace-drop-filter-tests", "vision-window-shutdown-tests"})
SUITE_NAMES |= REFACTOR_SUITE_NAMES
APPROVED_BOOLEAN_FIELDS["checks"] |= REFACTOR_CHECKS | REBUILD_CHECKS
APPROVED_BOOLEAN_FIELDS["refactor_architecture_checks"] = REFACTOR_CHECKS
APPROVED_BOOLEAN_FIELDS["source_rebuild_checks"] = REBUILD_CHECKS
APPROVED_COUNT_FIELDS.update({"refactor_architecture_checks": frozenset(), "source_rebuild_checks": frozenset()})
CHECK_GROUPS = frozenset(APPROVED_BOOLEAN_FIELDS)
PUBLIC_REPORT_NAMES |= frozenset({"packaged-decision-layout-qa.json", "refactor-architecture-qa.json", "source-rebuild-qa.json"})
WHOLE_ORIGINAL_ROLES |= PUBLIC_REPORT_NAMES
