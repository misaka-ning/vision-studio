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
        self.other_releases = {}
        self.commands = []
        self.next_id = 1

    def release(self, tag):
        if tag == f"v{self.root.name}":
            return self.value
        return self.other_releases.get(tag)

    def assets(self, value):
        return value.get("_assets", self.rows)

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

    def add_published_version(self, directory):
        value = {"id": 1000 + len(self.other_releases), "tag_name": f"v{directory.name}",
                 "draft": False, "prerelease": False, "published_at": "today",
                 "html_url": f"https://github.com/misaka-ning/vision-studio/releases/v{directory.name}",
                 "_assets": []}
        for asset in release.validate_bundle(directory, directory.name, require_qa=False):
            value["_assets"].append({"id": self.next_id, "name": asset.name, "state": "uploaded",
                                     "size": asset.size, "digest": "sha256:" + asset.sha256})
            self.next_id += 1
        self.other_releases[value["tag_name"]] = value
        return value

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
        self.published_tag_patch = patch.object(release, "verify_published_tag", return_value="a" * 40)
        self.published_tag_check = self.published_tag_patch.start()
        self.addCleanup(self.published_tag_patch.stop)

    def publish(self, github):
        with patch.object(release, "verify_tag", return_value="a" * 40), \
                patch.object(release, "command", side_effect=github.command):
            return release.publish(self.args, github)

    def prune_github(self, old, existing=True):
        github = FakeGitHub(old, existing=existing)
        for directory in self.root.parent.iterdir():
            if (directory.is_dir() and directory != old
                    and release.VERSION_RE.fullmatch(directory.name)):
                github.add_published_version(directory)
        if existing:
            for asset in release.validate_bundle(old, old.name, require_qa=False):
                github.add_asset(asset.path)
        return github

    def test_versions_use_numeric_order_and_reject_nonstable(self):
        self.assertGreater(release.version_key("1.10.0"), release.version_key("1.9.9"))
        for value in ("v1.5.0", "01.5.0", "1.5", "1.5.0-beta", "../1.5.0", "1.5.0;rm"):
            with self.subTest(value=value), self.assertRaises(release.ReleaseError):
                release.version_key(value)

    def test_passing_bundle_has_expected_archives_and_qa(self):
        assets = release.validate_bundle(self.root, "1.5.0")
        self.assertEqual({asset.name for asset in assets}, {
            "vision-studio_1.5.0-1_amd64.deb",
            "vision-studio-1.5.0-complete-source.tar.xz", "SHA256SUMS"})

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
        expected_names = {"vision-studio_1.5.0-1_amd64.deb",
                          "vision-studio-1.5.0-complete-source.tar.xz", "SHA256SUMS"}
        self.assertEqual({row["name"] for row in result["verified_assets"]}, expected_names)
        self.assertEqual({row["name"] for row in github.rows}, expected_names)
        self.assertEqual(sum(command[2] == "upload" for command in github.commands), 3)
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
        for name in ("unreviewed.bin", "SHA256SUMS", "SOURCE-SHA256SUMS", "RELEASE-ASSETS.json"):
            github = FakeGitHub(self.root, existing=True, draft=True)
            github.rows.append({"id": 1, "name": name, "size": 1, "state": "uploaded", "digest": "sha256:a"})
            self.args.resume_draft = True
            with self.assertRaises(release.ReleaseError):
                self.publish(github)
            self.assertEqual(github.commands, [])
            self.assertTrue(github.value["draft"])

    def test_three_asset_publish_still_checks_unuploaded_source_and_required_qa(self):
        names = ("vision-studio-1.5.0-sources.tar.xz", "sources/upstream.tar.xz",
                 "qa-report.json", "install-qa.json", "ctest-qa.json", "source-bundle-qa.json")
        for name in names:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(b'{"success": false}' if name.endswith(".json") else b"corrupt source")
                github = FakeGitHub(self.root)
                try:
                    with self.assertRaises(release.ReleaseError):
                        self.publish(github)
                    self.assertEqual(github.commands, [])
                    self.assertIsNone(github.value)
                finally:
                    path.write_bytes(original)

    def test_three_asset_verify_rejects_wrong_published_content_without_modification(self):
        github = FakeGitHub(self.root, existing=True)
        for asset in release.validate_bundle(self.root, "1.5.0"):
            github.add_asset(asset.path)
        github.rows[0]["digest"] = "sha256:" + "b" * 64
        args = argparse.Namespace(version="1.5.0", release_dir=self.root,
                                  project_dir=self.project, legacy=False)
        with self.assertRaisesRegex(release.ReleaseError, "Remote checksum mismatch"):
            release.verify(args, github)
        self.assertEqual(github.commands, [])
        self.assertFalse(github.value["draft"])

    def test_public_checksum_manifest_cannot_require_an_unattached_split_archive(self):
        manifest = self.root / "SHA256SUMS"
        split_name = "vision-studio-1.5.0-sources.tar.xz"
        manifest.write_text(manifest.read_text() + f"{release.sha256(self.root / split_name)}  {split_name}\n")
        with self.assertRaisesRegex(release.ReleaseError, "exactly the DEB and complete"):
            release.validate_bundle(self.root, "1.5.0")

    def test_prune_numeric_versions_preserves_latest_two_and_unknown_files(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.10.0")
        root = self.root.parent
        (root / "legacy.deb").write_bytes(b"legacy")
        (root / "99.0.0").write_text("A file must not count as a retained version")
        (root / "notes").mkdir()
        github = self.prune_github(old)
        args = argparse.Namespace(releases_root=root, execute=False, project_dir=self.project)
        plan = release.prune(args, github)
        self.assertEqual([Path(path).name for path in plan["kept"]], ["1.10.0", "1.5.0"])
        self.assertEqual(len(plan["verified_kept"]), 2)
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
        args = argparse.Namespace(releases_root=self.root.parent, execute=True, project_dir=self.project)
        github = self.prune_github(old, existing=False)
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
        args = argparse.Namespace(releases_root=root, execute=True, project_dir=self.project)
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
        cmake = "project(VisionStudio VERSION 1.5.0 LANGUAGES CXX)"
        base = [expected + "\n", cmake, expected + "\n", ""]
        with patch.object(release, "command", side_effect=base):
            self.assertEqual(release.verify_tag(github, self.project, "1.5.0"), expected)
        for responses in ([expected, cmake, "b" * 40], [expected, cmake, expected, " M src/file.cpp"],
                          [expected, "project(VisionStudio VERSION 1.4.0 LANGUAGES CXX)"]):
            with patch.object(release, "command", side_effect=responses), self.assertRaises(release.ReleaseError):
                release.verify_tag(github, self.project, "1.5.0")

    def test_prune_remote_tag_change_protects_snapshot(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.4.0")
        github = self.prune_github(old)
        args = argparse.Namespace(releases_root=self.root.parent, execute=True, project_dir=self.project)
        self.published_tag_check.side_effect = ["a" * 40] * 3 + [release.ReleaseError("Tag moved")]
        with self.assertRaisesRegex(release.ReleaseError, "Tag moved"):
            release.prune(args, github)
        self.assertTrue(old.is_dir())

    def test_prune_retained_draft_missing_or_prerelease_protects_all_old_versions(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.4.0")
        args = argparse.Namespace(releases_root=self.root.parent, execute=True, project_dir=self.project)
        for version in ("1.5.0", "1.4.0"):
            for state in ("draft", "missing", "prerelease"):
                with self.subTest(version=version, state=state):
                    github = self.prune_github(old)
                    value = github.other_releases[f"v{version}"]
                    if state == "missing":
                        del github.other_releases[f"v{version}"]
                    else:
                        value[state] = True
                    with self.assertRaisesRegex(release.ReleaseError, "Retained version is not durably published"):
                        release.prune(args, github)
                    self.assertTrue(old.is_dir())
                    self.assertTrue(self.root.is_dir())
                    self.assertTrue((self.root.parent / "1.4.0").is_dir())

    def test_prune_retained_release_missing_or_corrupt_asset_protects_all_old_versions(self):
        old = fixture(self.project, "1.3.0")
        fixture(self.project, "1.4.0")
        args = argparse.Namespace(releases_root=self.root.parent, execute=True, project_dir=self.project)
        for version in ("1.5.0", "1.4.0"):
            for failure in ("missing", "size", "digest", "starter"):
                with self.subTest(version=version, failure=failure):
                    github = self.prune_github(old)
                    assets = github.other_releases[f"v{version}"]["_assets"]
                    if failure == "missing":
                        assets.pop()
                    elif failure == "size":
                        assets[0]["size"] = 0
                    elif failure == "digest":
                        assets[0]["digest"] = "sha256:" + "b" * 64
                    else:
                        assets[0]["state"] = "starter"
                    with self.assertRaises(release.ReleaseError):
                        release.prune(args, github)
                    self.assertTrue(old.is_dir())
                    self.assertTrue(self.root.is_dir())
                    self.assertTrue((self.root.parent / "1.4.0").is_dir())


class GitHubReadTests(unittest.TestCase):
    def setUp(self):
        self.github = release.GitHub("misaka-ning/vision-studio")
        self.tag = "v1.5.0"
        self.draft = {"id": 42, "tag_name": self.tag, "draft": True, "published_at": None}

    def test_by_tag_404_finds_draft_through_paginated_list(self):
        requests = []
        def read(argv, **kwargs):
            endpoint = argv[-1]
            requests.append(endpoint)
            if "/releases/tags/" in endpoint:
                return subprocess.CompletedProcess(argv, 1, '{"message":"Not Found"}', "gh: Not Found (HTTP 404)")
            if endpoint.endswith("page=1"):
                rows = [{"id": number, "tag_name": f"other-{number}", "draft": False}
                        for number in range(100)]
            else:
                self.assertTrue(endpoint.endswith("page=2"))
                rows = [self.draft]
            return subprocess.CompletedProcess(argv, 0, json.dumps(rows), "")
        with patch.object(release.subprocess, "run", side_effect=read):
            self.assertEqual(self.github.release(self.tag), self.draft)
        self.assertEqual(len(requests), 3)

    def test_by_tag_published_does_not_need_list(self):
        published = {**self.draft, "draft": False, "published_at": "today"}
        with patch.object(release.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 0, json.dumps(published), "")) as run:
            self.assertEqual(self.github.release(self.tag), published)
            self.assertEqual(run.call_count, 1)

    def test_duplicate_tag_drafts_across_pages_are_rejected(self):
        page_one = [self.draft] + [{"id": number, "tag_name": f"other-{number}"}
                                  for number in range(99)]
        responses = [subprocess.CompletedProcess([], 1, "{}", "gh: Not Found (HTTP 404)"),
                     subprocess.CompletedProcess([], 0, json.dumps(page_one), ""),
                     subprocess.CompletedProcess([], 0, json.dumps([{**self.draft, "id": 43}]), "")]
        with patch.object(release.subprocess, "run", side_effect=responses), \
                self.assertRaisesRegex(release.ReleaseError, "Multiple releases"):
            self.github.release(self.tag)

    def test_auth_network_and_list_permission_errors_do_not_become_missing(self):
        for message in ("gh: Unauthorized (HTTP 401)", "gh: Forbidden (HTTP 403)", "TLS handshake failed"):
            with self.subTest(message=message), patch.object(release.subprocess, "run", return_value=
                    subprocess.CompletedProcess([], 1, "{}", message)) as run:
                with self.assertRaises(release.ReleaseError):
                    self.github.release(self.tag)
                self.assertEqual(run.call_count, 1)
        results = [subprocess.CompletedProcess([], 1, "{}", "gh: Not Found (HTTP 404)"),
                   subprocess.CompletedProcess([], 1, "{}", "gh: Forbidden (HTTP 403)")]
        with patch.object(release.subprocess, "run", side_effect=results), self.assertRaises(release.ReleaseError):
            self.github.release(self.tag)

    def test_missing_and_malformed_lists_are_distinct(self):
        for rows, expected_error in (([], False), ({"message": "unexpected"}, True)):
            responses = [subprocess.CompletedProcess([], 1, "{}", "gh: Not Found (HTTP 404)"),
                         subprocess.CompletedProcess([], 0, json.dumps(rows), "")]
            with patch.object(release.subprocess, "run", side_effect=responses):
                if expected_error:
                    with self.assertRaisesRegex(release.ReleaseError, "Malformed"):
                        self.github.release(self.tag)
                else:
                    self.assertIsNone(self.github.release(self.tag))

    def test_mutation_reads_retry_only_within_short_bound(self):
        with patch.object(self.github, "release", side_effect=[None, None, self.draft]) as read, \
                patch.object(release.time, "sleep") as sleep:
            self.assertEqual(release.release_after_mutation(self.github, self.tag, draft=True), self.draft)
            self.assertEqual(read.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        with patch.object(self.github, "release", return_value=None) as read, \
                patch.object(release.time, "sleep") as sleep, self.assertRaises(release.ReleaseError):
            release.release_after_mutation(self.github, self.tag, draft=True)
        self.assertEqual(read.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])

    def test_mutation_read_auth_error_is_not_retried(self):
        with patch.object(self.github, "release", side_effect=release.ReleaseError("HTTP 403")) as read, \
                patch.object(release.time, "sleep") as sleep, self.assertRaises(release.ReleaseError):
            release.release_after_mutation(self.github, self.tag, draft=True)
        self.assertEqual(read.call_count, 1)
        sleep.assert_not_called()

    def test_published_tag_checks_local_commit_and_version_without_head(self):
        project = Path("/unused")
        cmake = "project(VisionStudio VERSION 1.5.0 LANGUAGES CXX)"
        with patch.object(release, "command", side_effect=["a" * 40, cmake]) as command, \
                patch.object(self.github, "remote_commit", return_value="a" * 40):
            self.assertEqual(release.verify_published_tag(self.github, project, "1.5.0"), "a" * 40)
            self.assertEqual(command.call_count, 2)
        with patch.object(release, "command", side_effect=["a" * 40, cmake]), \
                patch.object(self.github, "remote_commit", return_value="b" * 40), \
                self.assertRaisesRegex(release.ReleaseError, "different commits"):
            release.verify_published_tag(self.github, project, "1.5.0")


if __name__ == "__main__":
    unittest.main()
