# SPDX-License-Identifier: AGPL-3.0-only
"""Cleanup guard tests use tiny temporary copies and synthetic receipts only.

No real release, old source directory, network or Git operation is modified.
These are safety-tool unit tests, not evidence that GitHub verification ran.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cleanup_old_2_1", Path(__file__).with_name("cleanup_old_2_1.py"))
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vision-old21-cleanup-selftest-")
        self.root = Path(self.temporary.name)
        self.base = self.root / "old"
        self.base.mkdir()
        self.source = self.base / "clean-source-verified"
        self.source.mkdir()
        (self.source / "source.cpp").write_text("// backed test fixture\n")
        self.tar = self.base / "clean-source-verified.tar"
        self.tar.write_bytes(b"tiny audited transport fixture")
        self.sibling = self.base / "unique-qa.py"
        self.sibling.write_text("# preserve this sibling\n")
        self.targets = (self.source, self.tar)
        self.audit = self.root / "audit.json"
        self.audit.write_text(json.dumps({"schema_version": 1, "version": "2.3.0", "success": True, "read_only": True,
            "backup": {"complete_source_sha256": "a" * 64, "complete_source_bytes": 17},
            "candidates": [{"path": str(p), "safe_to_delete_after_remote_verification": True,
                            "logical_bytes": 1, "inventory": cleanup.inventory(p)} for p in self.targets]}))

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, *arguments):
        output = io.StringIO()
        with patch.object(cleanup, "BASE", self.base), patch.object(cleanup, "CANDIDATES", self.targets), contextlib.redirect_stdout(output):
            code = cleanup.main(["--audit", str(self.audit), *arguments])
        return code, json.loads(output.getvalue())

    def receipts(self):
        options = []
        for version, switch in (("2.3.0", "2-3"), ("2.2.0", "2-2")):
            directory = self.root / version
            directory.mkdir()
            assets = []
            for index, name in enumerate((f"vision-studio_{version}-1_amd64.deb", f"vision-studio-{version}-complete-source.tar.xz", "SHA256SUMS")):
                path = directory / name
                path.write_bytes((version + name).encode())
                assets.append({"name": name, "bytes": path.stat().st_size, "sha256": cleanup.sha(path),
                               "asset_id": index + 1, "verification": "synthetic unit-test receipt; no remote request"})
            receipt = self.root / (version + "-verify.json")
            receipt.write_text(json.dumps({"success": True, "repository": cleanup.REPOSITORY, "tag": "v" + version,
                "prerelease": False, "commit": "b" * 40,
                "url": f"https://github.com/{cleanup.REPOSITORY}/releases/tag/v{version}", "verified_assets": assets}))
            options += ["--verify-" + switch, str(receipt), "--release-" + switch, str(directory)]
        assets = [{"name": name, "bytes": 17 if "complete-source" in name else 1, "sha256": "a" * 64,
                   "asset_id": index + 1, "verification": "synthetic test"} for index, name in enumerate(
                       ("vision-studio_2.1.0-1_amd64.deb", "vision-studio-2.1.0-complete-source.tar.xz", "SHA256SUMS"))]
        receipt = self.root / "old-prune.json"
        receipt.write_text(json.dumps({"success": True, "verified_candidates": [{"url":
            f"https://github.com/{cleanup.REPOSITORY}/releases/tag/v2.1.0", "commit": "c" * 40, "assets": assets}]}))
        return options + ["--prune-receipt", str(receipt)]

    def test_default_preview_does_not_delete(self):
        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["ready_to_execute"])
        self.assertEqual(report["removed"], [])
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_execute_without_remote_receipts_is_blocked(self):
        code, report = self.invoke("--execute")
        self.assertEqual(code, 1)
        self.assertFalse(report["success"])
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_changed_source_cannot_use_stale_audit(self):
        (self.source / "source.cpp").write_text("// unique new data\n")
        code, report = self.invoke("--execute", *self.receipts())
        self.assertEqual(code, 1)
        self.assertIn("changed", report["errors"][0])
        self.assertTrue(self.source.exists())

    def test_symlink_or_protected_runtime_is_rejected(self):
        (self.source / "outside").symlink_to(self.sibling)
        with self.assertRaises(cleanup.CleanupError): cleanup.inventory(self.source)
        (self.source / "outside").unlink()
        (self.source / "runtime").mkdir()
        with self.assertRaises(cleanup.CleanupError): cleanup.inventory(self.source)

    def test_wrong_version_and_changed_retained_asset_block_execution(self):
        options = self.receipts()
        receipt = self.root / "2.3.0-verify.json"
        data = json.loads(receipt.read_text()); data["tag"] = "v2.1.0"; receipt.write_text(json.dumps(data))
        code, _ = self.invoke("--execute", *options)
        self.assertEqual(code, 1); self.assertTrue(self.source.exists())
        data["tag"] = "v2.3.0"; receipt.write_text(json.dumps(data))
        (self.root / "2.2.0/SHA256SUMS").write_text("changed")
        code, report = self.invoke("--execute", *options)
        self.assertEqual(code, 1); self.assertFalse(report["ready_to_execute"])
        self.assertTrue(self.tar.exists())

    def test_two_exact_temporary_candidates_only(self):
        code, report = self.invoke("--execute", *self.receipts())
        self.assertEqual(code, 0)
        self.assertEqual(report["removed"], [str(p) for p in self.targets])
        self.assertTrue(self.sibling.exists())
        self.assertTrue((self.root / "2.3.0/SHA256SUMS").exists())

    def test_audit_cannot_select_another_path(self):
        data = json.loads(self.audit.read_text()); data["candidates"][0]["path"] = str(self.sibling)
        self.audit.write_text(json.dumps(data))
        code, _ = self.invoke("--execute", *self.receipts())
        self.assertEqual(code, 1)
        self.assertTrue(self.sibling.exists()); self.assertTrue(self.source.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
