# SPDX-License-Identifier: AGPL-3.0-only
# Audited whole suites; dependency/exclusion rationale is in docs/ci/linux-core.md.
set(VISION_PR_CORE_TESTS
    vision-core-tests
    vision-inference-delivery-tests
    vision-model-yaml-tests
    vision-dataset-session-tests
    vision-app-services-tests
    vision-gpu-environment-service-tests
    vision-model-session-controller-tests
    vision-detection-controller-tests
    vision-evaluation-controller-tests
    vision-operation-coordinator-tests
    vision-recording-controller-tests
    vision-workbench-configuration-tests
    vision-image-result-store-tests
    vision-result-delivery-tests
    vision-workbench-page-tests
    vision-workspace-pages-tests
    vision-simple-pages-tests
    vision-component-shutdown-tests
    vision-runtime-setup-shutdown-tests
    vision-workspace-drop-filter-tests
    vision-window-shutdown-tests
)
foreach(test IN LISTS VISION_PR_CORE_TESTS)
    if(NOT TARGET ${test} OR NOT TEST ${test})
        message(FATAL_ERROR "Audited PR core test is missing: ${test}")
    endif()
    set_property(TEST ${test} APPEND PROPERTY LABELS pr-core)
    # QtTest can exit successfully after QSKIP/XFAIL. Neither is PR acceptance.
    set_property(TEST ${test} APPEND PROPERTY FAIL_REGULAR_EXPRESSION
        "SKIP[ \t]*:" "XFAIL[ \t]*:")
endforeach()
set_tests_properties(vision-core-tests PROPERTIES TIMEOUT 120)
string(JOIN "\n" _vision_pr_core_manifest ${VISION_PR_CORE_TESTS})
file(WRITE "${CMAKE_BINARY_DIR}/pr-core-tests.txt" "${_vision_pr_core_manifest}\n")
add_custom_target(vision-pr-core DEPENDS vision-studio ${VISION_PR_CORE_TESTS})
