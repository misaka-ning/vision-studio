#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Release safety tests: only temporary files and mocked GitHub commands."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/github_release.py"
SPEC = importlib.util.spec_from_file_location("github_release", MODULE_PATH)
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def fixture(project, version="1.5.0"):
    root = project / "output/releases" / version
    root.mkdir(parents=True)
    deb = f"vision-studio_{version}-1_amd64.deb"
    app = f"vision-studio-{version}-sources.tar.xz"
    companion = f"vision-studio-{version}-complete-source.tar.xz"
    inventory = {"version": version, "application_archive": app, "complete_companion": companion}
    data = {deb: b"test deb", app: b"test app source", companion: b"test complete source",
            "SOURCE-INVENTORY.json": json.dumps(inventory).encode(),
            "sources/upstream.tar.xz": b"test upstream source"}
    for name, contents in data.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for name, items in (("SHA256SUMS", (deb, companion)),
                        ("SOURCE-SHA256SUMS", (app, "SOURCE-INVENTORY.json", "sources/upstream.tar.xz"))):
        (root / name).write_text("".join(f"{checksum(data[item])}  {item}\n" for item in items))
    for name in ("qa-report.json", "install-qa.json", "ctest-qa.json", "source-bundle-qa.json"):
        report = {"success": True, "version": version,
                  "details": {"deb_sha256": checksum(data[deb])}}
        (root / name).write_text(json.dumps(report))
    return root


class FakeGitHub:
    repository = "misaka-ning/vision-studio"

    def __init__(self, release_dir, existing=False, draft=False):
        self.root = release_dir
        self.value = ({"id": 42, "tag_name": f"v{release_dir.name}", "draft": draft,
                       "prerelease": False, "body": "notes", "published_at": None if draft else "today",
                       "html_url": "https://github.com/misaka-ning/vision-studio/releases/test"}
                      if existing else None)
        self.rows = []
        self.commands = []
        self.next_id = 1

    def release(self, tag):
        return self.value

    def assets(self, value):
        return self.rows

    def remote_commit(self, tag):
        return "a" * 40

    def check_asset(self, expected, row):
        return release.GitHub.check_asset(self, expected, row)

    def verify_assets(self, value, expected, allow_missing=False):
        return release.GitHub.verify_assets(self, value, expected, allow_missing)

    def add_asset(self, path):
        self.rows.append({"id": self.next_id, "name": path.name,
                          "state": "uploaded", "size": path.stat().st_size,
                          "digest": "sha256:" + release.sha256(path)})
        self.next_id += 1

    def command(self, argv, cwd=None):
        self.commands.append(argv)
        action = argv[2]
        if action == "create":
            self.value = {"id": 42, "tag_name": f"v{self.root.name}", "draft": True,
                          "prerelease": False, "published_at": None, "body": "notes",
                          "html_url": "https://github.com/misaka-ning/vision-studio/releases/test"}
        elif action == "upload":
            self.add_asset(Path(argv[4]))
        elif action == "edit":
            self.value["draft"] = False
            self.value["published_at"] = "today"
        else:
            raise AssertionError(argv)
        return ""


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-release-tests-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / "CMakeLists.txt").write_text("project(VisionStudio VERSION 1.5.0 LANGUAGES CXX)")
        (self.project / "src").mkdir()
        self.root = fixture(self.project)
        notes = self.project / "notes.md"
        notes.write_text("notes\n")
        self.args = argparse.Namespace(version="1.5.0", release_dir=self.root,
                                      notes_file=notes, project_dir=self.project,
                                      dry_run=False, resume_draft=False)

    def publish(self, github):
        with patch.object(release, "verify_tag", return_value="a" * 40), \
                patch.object(release, "command", side_effect=github.command):
            return release.publish(self.args, github)

    def test_versions_use_numeric_order_and_reject_nonstable(self):
        self.assertGreater(release.version_key("1.10.0"), release.version_key("1.9.9"))
        for value in ("v1.5.0", "01.5.0", "1.5", "1.5.0-beta", "../1.5.0", "1.5.0;rm"):
            with self.subTest(value=value), self.assertRaises(release.ReleaseError):
                release.version_key(value)

    def test_passing_bundle_has_expected_archives_and_qa(self):
        assets = release.validate_bundle(self.root, "1.5.0")
        self.assertEqual(len(assets), 10)
        self.assertIn("SOURCE-SHA256SUMS", {asset.name for asset in assets})

    def test_missing_and_corrupt_checksums_rejected(self):
        (self.root / "vision-studio_1.5.0-1_amd64.deb").write_bytes(b"changed")
        with self.assertRaisesRegex(release.ReleaseError, "Checksum mismatch"):
            release.validate_bundle(self.root, "1.5.0")

    def test_duplicate_manifest_path_rejected(self):
        path = self.root / "SHA256SUMS"
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(release.ReleaseError, "Duplicate"):
            release.validate_bundle(self.root, "1.5.0")

    def test_manifest_traversal_rejected(self):
        (self.root / "SHA256SUMS").write_text(f"{'a' * 64}  ../secret\n")
        with self.assertRaisesRegex(release.ReleaseError, "Unsafe"):
            release.validate_bundle(self.root, "1.5.0")

    def test_symlink_source_rejected(self):
        file = self.root / "sources/upstream.tar.xz"
        file.unlink()
        file.symlink_to(self.project / "CMakeLists.txt")
        with self.assertRaisesRegex(release.ReleaseError, "Symlink"):
            release.validate_bundle(self.root, "1.5.0")

    def test_failed_or_other_deb_qa_rejected(self):
        for report in ({"success": False}, {"success": True, "details": {"deb_sha256": "wrong"}}):
            (self.root / "qa-report.json").write_text(json.dumps(report))
            with self.assertRaises(release.ReleaseError):
                release.validate_bundle(self.root, "1.5.0")

    def test_inventory_version_rejected_even_with_matching_hash(self):
        path = self.root / "SOURCE-INVENTORY.json"
        data = json.loads(path.read_text())
        data["version"] = "1.4.0"
        path.write_text(json.dumps(data))
        manifest = self.root / "SOURCE-SHA256SUMS"
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(f"{release.sha256(path)}  SOURCE-INVENTORY.json"
                                      if line.endswith("  SOURCE-INVENTORY.json") else line
                                      for line in lines) + "\n")
        with self.assertRaisesRegex(release.ReleaseError, "does not match"):
            release.validate_bundle(self.root, "1.5.0")

    def test_remote_state_size_and_digest_all_required(self):
        asset = release.make_asset(self.root, "SHA256SUMS")
        github = release.GitHub("misaka-ning/vision-studio")
        row = {"id": 1, "size": asset.size, "state": "uploaded", "digest": "sha256:" + asset.sha256}
        self.assertEqual(github.check_asset(asset, row)["verification"], "GitHub REST sha256 digest")
        for bad in ({"size": 0}, {"state": "starter"}, {"digest": "sha256:" + "a" * 64}):
            with self.subTest(bad=bad), self.assertRaises(release.ReleaseError):
                github.check_asset(asset, {**row, **bad})

    def test_remote_missing_digest_downloads_and_hashes(self):
        asset = release.make_asset(self.root, "SHA256SUMS")
        github = release.GitHub("misaka-ning/vision-studio")
        row = {"id": 17, "size": asset.size, "state": "uploaded", "digest": None}
        def download(argv, **kwargs):
            self.assertEqual(argv[-1], "repos/misaka-ning/vision-studio/releases/assets/17")
            kwargs["stdout"].write(asset.path.read_bytes())
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(release.subprocess, "run", side_effect=download):
            self.assertEqual(github.check_asset(asset, row)["verification"], "downloaded sha256")

    def test_dry_run_does_not_create_or_upload(self):
        self.args.dry_run = True
        github = FakeGitHub(self.root)
        plan = self.publish(github)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(github.commands, [])
        self.assertIsNone(github.value)

    def test_draft_verified_before_publish_and_no_overwrite(self):
        github = FakeGitHub(self.root)
        result = self.publish(github)
        self.assertTrue(result["published"])
        self.assertEqual(github.commands[0][2], "create")
        self.assertEqual(github.commands[-1][2], "edit")
        self.assertEqual(len(result["verified_assets"]), 11)
        self.assertFalse(any("--clobber" in command for command in github.commands))

    def test_existing_published_release_is_never_modified(self):
        github = FakeGitHub(self.root, existing=True)
        self.args.resume_draft = True
        with self.assertRaisesRegex(release.ReleaseError, "already exists"):
            self.publish(github)
        self.assertEqual(github.commands, [])

    def test_existing_draft_requires_explicit_resume(self):
        github = FakeGitHub(self.root, existing=True, draft=True)
        with self.assertRaisesRegex(release.ReleaseError, "already exists"):
            self.publish(github)
        self.args.resume_draft = True
        github.add_asset(self.root / "SHA256SUMS")
        self.assertTrue(self.publish(github)["published"])
        self.assertFalse(any(command[2] == "create" for command in github.commands))
        self.assertEqual(sum(row["name"] == "SHA256SUMS" for row in github.rows), 1)

    def test_draft_unexpected_asset_or_digest_mismatch_protects_release(self):
        for name in ("unreviewed.bin", "SHA256SUMS"):
            github = FakeGitHub(self.root, existing=True, draft=True)
            github.rows.append({"id": 1, "name": name, "size": 1, "state": "uploaded", "digest": "sha256:a"})
            self.args.resume_draft = True
            with self.assertRaises(release.ReleaseError):
                self.publish(github)
            self.assertEqual(github.commands, [])
            self.assertTrue(github.value["draft"])

    def test_prune_numeric_versions_preserves_latest_two_and_unknown_files(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.10.0")
        root = self.root.parent
        (root / "legacy.deb").write_bytes(b"legacy")
        (root / "99.0.0").write_text("A file must not count as a retained version")
        (root / "notes").mkdir()
        github = FakeGitHub(old, existing=True)
        for asset in release.validate_bundle(old, "1.3.0", require_qa=False):
            github.add_asset(asset.path)
        args = argparse.Namespace(releases_root=root, execute=False)
        plan = release.prune(args, github)
        self.assertEqual([Path(path).name for path in plan["kept"]], ["1.10.0", "1.5.0"])
        self.assertTrue(old.exists())
        args.execute = True
        result = release.prune(args, github)
        self.assertEqual(result["removed"], [str(old)])
        self.assertFalse(old.exists())
        self.assertTrue((root / "legacy.deb").exists())
        self.assertTrue((root / "99.0.0").exists())
        self.assertTrue((root / "notes").is_dir())

    def test_prune_unpublished_checksum_bad_or_userdata_cannot_delete(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.4.0")
        args = argparse.Namespace(releases_root=self.root.parent, execute=True)
        github = FakeGitHub(old)
        with self.assertRaisesRegex(release.ReleaseError, "not durably published"):
            release.prune(args, github)
        github.value = {"id": 42, "draft": False, "prerelease": False, "published_at": "today", "html_url": "url"}
        with self.assertRaisesRegex(release.ReleaseError, "Missing remote"):
            release.prune(args, github)
        (old / "preferences.ini").write_text("user data")
        with self.assertRaisesRegex(release.ReleaseError, "protected"):
            release.prune(args, github)
        self.assertTrue(old.is_dir())

    def test_prune_symlink_version_is_rejected_and_target_preserved(self):
        root = self.root.parent
        fixture(self.project, "1.4.0")
        (root / "1.3.0").symlink_to(self.project, target_is_directory=True)
        args = argparse.Namespace(releases_root=root, execute=True)
        with self.assertRaisesRegex(release.ReleaseError, "Symlink"):
            release.prune(args, FakeGitHub(self.root))
        self.assertTrue((self.project / "CMakeLists.txt").exists())

    def test_prune_root_scope_and_repository_are_restricted(self):
        for root in (self.project, self.project / "output", self.root):
            with self.subTest(root=root), self.assertRaises(release.ReleaseError):
                release.cleanup_root(root)
        for repo in ("--help", "https://github.com/a/b", "a/b;rm", "../b"):
            with self.subTest(repo=repo), self.assertRaises(release.ReleaseError):
                release.GitHub(repo)

    def test_tag_version_commit_and_dirty_checks(self):
        github = FakeGitHub(self.root)
        expected = "a" * 40
        base = [expected + "\n", expected + "\n", "", "project(VisionStudio VERSION 1.5.0 LANGUAGES CXX)"]
        with patch.object(release, "command", side_effect=base):
            self.assertEqual(release.verify_tag(github, self.project, "1.5.0"), expected)
        for responses in ([expected, "b" * 40], [expected, expected, " M src/file.cpp"],
                          [expected, expected, "", "project(VisionStudio VERSION 1.4.0 LANGUAGES CXX)"]):
            with patch.object(release, "command", side_effect=responses), self.assertRaises(release.ReleaseError):
                release.verify_tag(github, self.project, "1.5.0")


if __name__ == "__main__":
    unittest.main()
