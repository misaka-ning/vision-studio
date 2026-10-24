# SPDX-License-Identifier: AGPL-3.0-only
"""Supplement publication safety: temporary files and mocked GitHub only."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("github_release", Path(__file__).resolve().parents[1] / "scripts/github_release.py")
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


class WindowsReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.delivery = self.root / "output/windows-2.1.0"
        self.delivery.mkdir(parents=True)
        self.evidence = self.root / "docs/releases/evidence/2.1.0/windows"
        self.evidence.mkdir(parents=True)
        self.commit = "a" * 40
        self.old_commit = "b" * 40
        self.baseline = {"release_id": 42, "tag_name": "v2.1.0", "tag_commit": self.old_commit,
                         "body": "original Linux notes", "assets": [
                             {"id": 1, "name": "SHA256SUMS", "size": 10, "digest": "sha256:" + "c" * 64}]}
        self.dump("release-before-windows.json", self.baseline)
        installer = self.delivery / "vision-studio-2.1.0-windows-x64-setup.exe"
        installer.write_bytes(b"frozen installer")
        source = self.delivery / "vision-studio-2.1.0-windows-sources.zip"
        self.source_data = b"set(VISION_STUDIO_APP_VERSION 2.1.0)\n"
        prefix = "vision-studio-2.1.0-windows-sources/"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(prefix + "CMakeLists.txt", self.source_data)
            archive.writestr(prefix + "WINDOWS-SOURCE-SNAPSHOT.json", json.dumps({"files": [
                {"path": "CMakeLists.txt", "sha256": hashlib.sha256(self.source_data).hexdigest()}]}))
        (self.delivery / "SHA256SUMS-windows-x64").write_text(
            "".join(f"{release.sha256(path)}  {path.name}\n" for path in (installer, source)))
        self.qa = {"success": True, "version": "2.1.0", "package_revision": "windows.1",
                   "installer": {"sha256": release.sha256(installer)},
                   "source_snapshot": {"sha256": release.sha256(source)}}
        self.dump("windows-package-qa.json", self.qa)
        names = ("install-custom-directory all-payload-paths-and-sizes installed-onnx-cpu installed-pt-cpu "
                 "installed-model-views installed-conversion-onnx installed-conversion-torchscript "
                 "installed-onnx-cuda installed-pt-cuda upgrade-reinstall upgrade-preserves-user-files "
                 "uninstall uninstall-preserves-user-files").split()
        self.install_qa = {"success": True, "sha256": release.sha256(installer),
                           "checks": [{"name": name, "passed": True} for name in names]}
        self.dump("installer-qa.json", self.install_qa)
        self.notes = self.root / "notes.md"
        self.notes.write_text(f"<!-- windows-2.1.0-1 -->\nhttps://github.com/test/repo/tree/{self.commit}/docs\n")
        self.args = argparse.Namespace(version="2.1.0", project_dir=self.root,
                                       release_dir=self.delivery, operation="append-windows",
                                       notes_file=self.notes, dry_run=False)
        self.release_value = {"id": 42, "tag_name": "v2.1.0", "body": self.baseline["body"],
                              "draft": False, "prerelease": False, "published_at": "today", "html_url": "url"}
        self.rows = [{**self.baseline["assets"][0], "state": "uploaded"}]
        self.commands = []

        class FakeGitHub:
            repository = "test/repo"
            check_asset = release.GitHub.check_asset
            verify_assets = release.GitHub.verify_assets
            release = lambda _, tag: self.release_value
            assets = lambda _, value: self.rows
            remote_commit = lambda _, tag: self.commit if tag.startswith("windows/") else self.old_commit

        self.github = FakeGitHub()

    def dump(self, name, data):
        (self.evidence / name).write_text(json.dumps(data), encoding="utf-8")

    def archive(self, arguments, **kwargs):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tree:
            member = tarfile.TarInfo("CMakeLists.txt")
            member.size = len(self.source_data)
            tree.addfile(member, io.BytesIO(self.source_data))
            if hasattr(self, "native_data"):
                member = tarfile.TarInfo("packaging/windows/windows-native-sources.json")
                member.size = len(self.native_data)
                tree.addfile(member, io.BytesIO(self.native_data))
        return subprocess.CompletedProcess(arguments, 0, buffer.getvalue(), b"")

    def command(self, arguments, cwd=None):
        if arguments[0] == "git":
            return self.commit
        self.commands.append(arguments)
        if arguments[2] == "upload":
            path = Path(arguments[4])
            self.rows.append({"id": len(self.rows) + 10, "name": path.name, "size": path.stat().st_size,
                              "digest": "sha256:" + release.sha256(path), "state": "uploaded"})
        elif arguments[2] == "edit":
            self.assertNotIn(b"\r\n", Path(arguments[-1]).read_bytes())
            self.release_value["body"] = Path(arguments[-1]).read_text(encoding="utf-8")
        return ""

    def publish(self):
        with patch.object(release.subprocess, "run", side_effect=self.archive), \
                patch.object(release, "command", side_effect=self.command):
            return release.windows_release(self.args, self.github)

    def test_publishes_three_windows_files_and_preserves_linux(self):
        result = self.publish()
        self.assertTrue(result["success"])
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(len(result["verified_assets"]), 3)
        self.assertTrue(self.release_value["body"].startswith(self.baseline["body"]))
        self.assertFalse(any("--clobber" in command for command in self.commands))

    def test_retry_is_idempotent(self):
        self.publish()
        self.commands.clear()
        self.publish()
        self.assertEqual(self.commands, [])

    def test_dry_run_does_not_upload_or_edit(self):
        self.args.dry_run = True
        self.assertTrue(self.publish()["dry_run"])
        self.assertEqual(self.commands, [])

    def test_retry_repairs_crlf_only_notes_without_reupload(self):
        self.publish()
        self.release_value["body"] = self.release_value["body"].replace("\n", "\r\n")
        self.commands.clear()
        self.publish()
        self.assertEqual([command[2] for command in self.commands], ["edit"])
        self.assertNotIn("\r\n", self.release_value["body"])

    def test_changed_linux_asset_is_rejected(self):
        self.rows[0]["digest"] = "sha256:" + "d" * 64
        with self.assertRaisesRegex(release.ReleaseError, "Original release asset changed"):
            self.publish()
        self.assertEqual(self.commands, [])

    def test_changed_windows_asset_is_never_overwritten(self):
        self.publish()
        self.commands.clear()
        self.rows[1]["digest"] = "sha256:" + "d" * 64
        with self.assertRaisesRegex(release.ReleaseError, "checksum mismatch"):
            self.publish()
        self.assertEqual(self.commands, [])

    def test_notes_changed_since_baseline_is_rejected(self):
        self.release_value["body"] = "someone edited notes"
        with self.assertRaisesRegex(release.ReleaseError, "notes changed since baseline"):
            self.publish()
        self.assertEqual(self.commands, [])

    def test_missing_install_check_is_rejected(self):
        self.install_qa["checks"].pop()
        self.dump("installer-qa.json", self.install_qa)
        with self.assertRaisesRegex(release.ReleaseError, "acceptance is incomplete"):
            self.publish()

    def test_final_installer_tampering_is_rejected(self):
        (self.delivery / self.qa["installer"].get("file", "vision-studio-2.1.0-windows-x64-setup.exe")).write_bytes(b"changed")
        with self.assertRaisesRegex(release.ReleaseError, "Checksum mismatch"):
            self.publish()

    def test_wrong_source_tag_is_rejected(self):
        self.source_data = b"different committed source"
        with self.assertRaisesRegex(release.ReleaseError, "tag differs from delivered source"):
            self.publish()

    def test_verify_requires_public_notes(self):
        self.publish()
        self.args.operation = "verify-windows"
        self.commands.clear()
        self.assertTrue(self.publish()["success"])
        self.assertEqual(self.commands, [])
        self.release_value["body"] = self.baseline["body"]
        with self.assertRaisesRegex(release.ReleaseError, "notes are missing"):
            self.publish()

    def test_cli_output_uses_utf8_on_windows(self):
        text = release.command([sys.executable, "-c", "import os; os.write(1, bytes.fromhex('e4b8ade69687'))"])
        self.assertEqual(text, "中文")

    def dual_fixture(self):
        """Mock acceptance records only; no generated installer is called real QA."""
        prefix = 'vision-studio-2.1.0-windows-'
        self.native_data = json.dumps({'ffmpeg':[{'archive':'ffmpeg.tar.xz',
            'sha256':hashlib.sha256(b'upstream-source-fixture').hexdigest()}]}).encode()
        self.dual_names = [prefix+'x64-'+p+'-setup.exe' for p in ('offline','online')]
        for name in self.dual_names: (self.delivery/name).write_bytes(b'mocked installer '+name.encode())
        self.dual_runtime = self.delivery/(prefix+'x64-cpu-runtime.zip')
        with zipfile.ZipFile(self.dual_runtime,'w') as archive:
            archive.writestr('python.exe',b'not a real interpreter')
            archive.writestr('CPU-FILES.json',json.dumps({'schema_version':1,'files':[
                {'path':'python.exe','bytes':22,'sha256':hashlib.sha256(b'not a real interpreter').hexdigest()}]}))
        self.dual_source = self.delivery/(prefix+'sources.zip')
        folder=prefix+'sources/'
        with zipfile.ZipFile(self.dual_source,'w') as archive:
            files={'CMakeLists.txt':self.source_data,'packaging/windows/windows-native-sources.json':self.native_data}
            for name,data in files.items(): archive.writestr(folder+name,data)
            archive.writestr(folder+'third-party/ffmpeg/ffmpeg.tar.xz',b'upstream-source-fixture')
            archive.writestr(folder+'WINDOWS-SOURCE-SNAPSHOT.json',json.dumps({'files':[
                {'path':p,'sha256':hashlib.sha256(d).hexdigest()} for p,d in files.items()], 'third_party_files':[
                {'path':'third-party/ffmpeg/ffmpeg.tar.xz','sha256':hashlib.sha256(b'upstream-source-fixture').hexdigest()}]}))
        self.dual_names += [self.dual_runtime.name,self.dual_source.name]
        self.qa={'schema_version':2,'success':True,'version':'2.1.0','package_revision':'windows.1',
            'profiles':{p:{'sha256':release.sha256(self.delivery/(prefix+f'x64-{p}-setup.exe'))} for p in ('offline','online')},
            'cpu_runtime':{'sha256':release.sha256(self.dual_runtime)},
            'source_snapshot':{'sha256':release.sha256(self.dual_source)}}
        common={r['name'] for r in self.install_qa['checks']} | {'installed-evaluation','installed-dataset-session',
            'installed-preview-drop','no-source-docs-examples'}
        for p in ('offline','online'):
            extra={'offline-cpu-without-network'} if p=='offline' else {
                'online-no-runtime-onnx','online-runtime-preparation','online-runtime-offline-reuse'}
            self.dump(f'installer-{p}-qa.json',{'success':True,'sha256':self.qa['profiles'][p]['sha256'],
                'checks':[{'name':n,'passed':True} for n in sorted(common|extra)]})
        self.dump('cpu-runtime-qa.json',{'success':True,'sha256':self.qa['cpu_runtime']['sha256'],
            'checks':[{'name':n,'passed':True} for n in ['dependency-self-test','cancel-and-retry','checksum-rejection',
                'concurrent-preparation','archive-path-protection','upgrade-cleanup-protection','profile-switching']]})
        self.dump('windows-package-qa.json',self.qa)
        self.refresh_dual_manifest()

    def refresh_dual_manifest(self):
        (self.delivery/'SHA256SUMS-windows-x64').write_text(''.join(
            release.sha256(self.delivery/name)+'  '+name+'\n' for name in self.dual_names))

    def test_dual_publishes_five_assets_without_changing_linux(self):
        self.dual_fixture(); result=self.publish()
        self.assertTrue(result['success']); self.assertEqual(len(result['verified_assets']),5)
        self.assertEqual(self.rows[0]['digest'],self.baseline['assets'][0]['digest'])
        self.commands.clear(); self.publish(); self.assertEqual(self.commands,[])

    def test_dual_rejects_missing_online_acceptance(self):
        self.dual_fixture()
        report=json.loads((self.evidence/'installer-online-qa.json').read_text())
        report['checks']=[r for r in report['checks'] if r['name']!='online-no-runtime-onnx']
        self.dump('installer-online-qa.json',report)
        with self.assertRaisesRegex(release.ReleaseError,'online installation acceptance is incomplete'): self.publish()
        self.assertEqual(self.commands,[])

    def test_dual_rejects_unbound_runtime(self):
        self.dual_fixture(); self.qa['cpu_runtime']['sha256']='0'*64; self.dump('windows-package-qa.json',self.qa)
        with self.assertRaisesRegex(release.ReleaseError,'bind the final cpu_runtime'): self.publish()
        self.assertEqual(self.commands,[])

    def test_dual_rejects_corrupt_runtime_member_even_with_new_zip_checksum(self):
        self.dual_fixture()
        with zipfile.ZipFile(self.dual_runtime) as archive: inventory=archive.read('CPU-FILES.json')
        with zipfile.ZipFile(self.dual_runtime,'w') as archive:
            archive.writestr('CPU-FILES.json',inventory); archive.writestr('python.exe',b'corrupt')
        sha=release.sha256(self.dual_runtime); self.qa['cpu_runtime']['sha256']=sha
        self.dump('windows-package-qa.json',self.qa)
        runtime_qa=json.loads((self.evidence/'cpu-runtime-qa.json').read_text()); runtime_qa['sha256']=sha
        self.dump('cpu-runtime-qa.json',runtime_qa); self.refresh_dual_manifest()
        with self.assertRaisesRegex(release.ReleaseError,'member checksum mismatch'): self.publish()
        self.assertEqual(self.commands,[])

    def test_dual_rejects_extra_asset_in_checksum_list(self):
        self.dual_fixture(); (self.delivery/'unrequested.zip').write_bytes(b'no')
        self.dual_names.append('unrequested.zip'); self.refresh_dual_manifest()
        with self.assertRaisesRegex(release.ReleaseError,'must list both installers'): self.publish()


if __name__ == "__main__":
    unittest.main()
