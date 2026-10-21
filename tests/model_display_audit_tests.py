# SPDX-License-Identifier: AGPL-3.0-only
"""Unit checks of the package auditor's model-view contract, without Qt.

PNG signatures are synthetic fixtures for the existing signature check, not
evidence of rendering or visual acceptance. Real model screenshots remain part
of the final-DEB smoke and independent package matrix.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
from verify_deb import valid_model_display


class ModelDisplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.png = b"\x89PNG\r\n\x1a\n"
        for name in ("model-display.png", "model-hierarchy.png", "model-parameters.png"):
            (self.output / name).write_bytes(self.png)
        self.report = {
            "version": "2.8.0", "model": "model.onnx", "success": True,
            "nodes": 1, "hierarchy_items": 1, "parameter_rows": 1,
            "cache_preserved": True,
            "display_modes": {"graph": True, "hierarchy": True, "parameters": True},
            "yaml": {"available": False, "mode_visible": False, "reason": "No saved architecture",
                     "source_sha256_before": "1" * 64, "source_sha256_after": "1" * 64},
        }

    def valid(self, report=None, exit_code=0):
        return valid_model_display(self.output, report or self.report, exit_code)

    def add_supported_yaml(self):
        self.report["model"] = "model.pt"
        text = b"# [from, repeats, module, args]\nnc: 80\n"
        (self.output / "model-architecture.yaml").write_bytes(text)
        (self.output / "model-yaml.png").write_bytes(self.png)
        digest = hashlib.sha256(text).hexdigest()
        self.report["display_modes"]["yolo_yaml"] = True
        self.report["yaml"].update(available=True, mode_visible=True, exported=True,
                                   existing_export_refused=True, bytes=len(text),
                                   sha256=digest, export_sha256=digest)

    def test_onnx_without_architecture_has_three_actual_views(self):
        self.assertTrue(self.valid())

    def test_supported_pt_requires_visible_mode_and_real_export_bytes(self):
        self.add_supported_yaml()
        self.assertTrue(self.valid())
        self.report["yaml"]["mode_visible"] = False
        self.assertFalse(self.valid())

    def test_unavailable_mode_must_be_hidden(self):
        self.report["yaml"]["mode_visible"] = True
        self.assertFalse(self.valid())

    def test_unavailable_mode_cannot_be_counted_as_success_or_failure(self):
        for value in (True, False):
            with self.subTest(value=value):
                self.report["display_modes"]["yolo_yaml"] = value
                self.assertFalse(self.valid())

    def test_unavailable_yaml_screenshot_is_rejected(self):
        (self.output / "model-yaml.png").write_bytes(self.png)
        self.assertFalse(self.valid())

    def test_unavailable_generic_checkpoint_is_allowed_but_bundled_pt_is_not(self):
        self.report["model"] = "unsupported.pt"
        self.assertTrue(self.valid())
        for name in ("yolov5n.pt", "yolov8n.pt"):
            with self.subTest(name=name):
                self.report["model"] = name
                self.assertFalse(self.valid())

    def test_missing_availability_visibility_or_original_identity_is_rejected(self):
        for key in ("available", "mode_visible", "source_sha256_before", "source_sha256_after"):
            with self.subTest(key=key):
                report = deepcopy(self.report)
                del report["yaml"][key]
                self.assertFalse(self.valid(report))

    def test_270_keeps_legacy_fourth_empty_view_contract(self):
        self.report["version"] = "2.7.0"
        del self.report["yaml"]["mode_visible"]
        self.assertFalse(self.valid())
        self.report["display_modes"]["yolo_yaml"] = True
        (self.output / "model-yaml.png").write_bytes(self.png)
        self.assertTrue(self.valid())

    def test_pre_yaml_version_keeps_three_view_contract(self):
        self.report["version"] = "2.6.0"
        del self.report["yaml"]
        self.assertTrue(self.valid())

    def test_process_failure_and_stale_source_hash_rejected(self):
        self.assertFalse(self.valid(exit_code=2))
        self.report["yaml"]["source_sha256_after"] = "2" * 64
        self.assertFalse(self.valid())


if __name__ == "__main__":
    unittest.main()
