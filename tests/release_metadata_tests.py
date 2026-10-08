#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pure release metadata and packaging tests; no real packages or remote writes."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import release_metadata as metadata


def load_tool(name, relative):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_tool("metadata_build_deb", "packaging/build_deb.py")
gpu_verifier = load_tool("metadata_gpu_verifier", "packaging/verify_gpu_deb.py")
source_builder = load_tool("metadata_source_builder", "docs/release/build_source_archive.py")


class ReleaseMetadataTests(unittest.TestCase):
    def test_numeric_beta_precedence_before_final_and_stable_order_unchanged(self):
        values = ["2.0.0", "2.0.0-beta.10", "1.9.9", "2.0.0-beta.2", "2.0.0-beta.1"]
        self.assertEqual(sorted(values, key=metadata.version_key), [
            "1.9.9", "2.0.0-beta.1", "2.0.0-beta.2", "2.0.0-beta.10", "2.0.0"])
        self.assertGreater(metadata.version_key("1.10.0"), metadata.version_key("1.9.9"))
        self.assertFalse(metadata.is_prerelease("1.6.0"))
        self.assertTrue(metadata.is_prerelease("2.0.0-beta.1"))

    def test_debian_roundtrip_and_revision(self):
        for version in ("1.6.0", "2.0.0-beta.0", "2.0.0-beta.1", "2.0.0-beta.10"):
            with self.subTest(version=version):
                deb = metadata.debian_version(version, 2)
                self.assertEqual(metadata.application_version_from_debian(deb), version)
        self.assertEqual(metadata.debian_version("1.6.0"), "1.6.0-1")
        self.assertEqual(metadata.debian_version("2.0.0-beta.1"), "2.0.0~beta.1-1")
        self.assertEqual(metadata.deb_filename("2.0.0~beta.1-1"), "vision-studio_2.0.0-beta.1-1_amd64.deb")
        self.assertEqual(metadata.deb_filename("1.6.0-1"), "vision-studio_1.6.0-1_amd64.deb")
        for revision in (0, -1, True, "1"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                metadata.debian_version("2.0.0-beta.1", revision)

    @unittest.skipUnless(shutil.which("dpkg"), "Debian version comparator unavailable")
    def test_real_dpkg_orders_beta_numerically_and_before_final(self):
        pairs = [("2.0.0~beta.1-1", "2.0.0~beta.2-1"),
                 ("2.0.0~beta.2-1", "2.0.0~beta.10-1"),
                 ("2.0.0~beta.10-1", "2.0.0-1"),
                 ("1.6.0-1", "2.0.0~beta.1-1")]
        for earlier, later in pairs:
            with self.subTest(earlier=earlier, later=later):
                result = subprocess.run(["dpkg", "--compare-versions", earlier, "lt", later],
                                        capture_output=True, check=False)
                self.assertEqual(result.returncode, 0)

    def test_rejects_noncanonical_and_unsupported_application_versions(self):
        for value in ("v2.0.0-beta.1", "02.0.0", "2.0.0-beta", "2.0.0-beta.01",
                      "2.0.0-rc.1", "2.0.0+build.1", "../2.0.0", "2.0.0\n", "2.0.0;rm"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metadata.version_key(value)

    def test_rejects_debian_format_that_could_sort_beta_after_final(self):
        for value in ("2.0.0-beta.1-1", "2.0.0~beta.01-1", "2.0.0~beta.1-0",
                      "2.0.0~beta.1", "2.0.0-01", "1:2.0.0-1", "2.0.0~rc.1-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metadata.application_version_from_debian(value)

    def test_cmake_full_version_and_frozen_stable_tags(self):
        stable = "project(VisionStudio VERSION 1.6.0 LANGUAGES CXX)\n"
        beta = ('project(VisionStudio VERSION 2.0.0 LANGUAGES CXX)\n'
                'set(VISION_STUDIO_APP_VERSION "2.0.0-beta.1")\n')
        self.assertEqual(metadata.cmake_application_version(stable), "1.6.0")
        self.assertEqual(metadata.cmake_application_version(beta), "2.0.0-beta.1")
        self.assertEqual(metadata.cmake_application_version('# set(VISION_STUDIO_APP_VERSION "9.0.0")\n' + stable), "1.6.0")

    def test_cmake_refuses_mismatch_duplicate_or_malformed_full_version(self):
        base = "project(VisionStudio VERSION 2.0.0 LANGUAGES CXX)\n"
        assignments = [
            'set(VISION_STUDIO_APP_VERSION "1.6.0")',
            'set(VISION_STUDIO_APP_VERSION "2.0.0-beta.01")',
            'set(VISION_STUDIO_APP_VERSION "2.0.0-beta.1" CACHE STRING "version")',
            'set(VISION_STUDIO_APP_VERSION "2.0.0-beta.1")\nset(VISION_STUDIO_APP_VERSION "2.0.0")',
        ]
        for value in assignments:
            with self.subTest(value=value), self.assertRaises(ValueError):
                metadata.cmake_application_version(base + value)

    def test_gpu_package_metadata_preserves_full_beta_version(self):
        for deb, app in [("1.6.0-1", "1.6.0"), ("2.0.0~beta.1-1", "2.0.0-beta.1")]:
            result = subprocess.CompletedProcess([], 0,
                f"Package: vision-studio\nArchitecture: amd64\nVersion: {deb}\n", "")
            with patch.object(gpu_verifier.subprocess, "run", return_value=result):
                self.assertEqual(gpu_verifier.metadata(Path("unused.deb"))[1], app)
        bad = subprocess.CompletedProcess([], 0,
            "Package: vision-studio\nArchitecture: amd64\nVersion: 2.0.0-beta.1-1\n", "")
        with patch.object(gpu_verifier.subprocess, "run", return_value=bad), self.assertRaises(RuntimeError):
            gpu_verifier.metadata(Path("unused.deb"))

    def test_build_default_maps_source_beta_and_retains_semver_directory(self):
        with tempfile.TemporaryDirectory(prefix="vision-metadata-tests-") as directory:
            project = Path(directory)
            for version, declaration in [
                ("1.6.0", "project(VisionStudio VERSION 1.6.0 LANGUAGES CXX)"),
                ("2.0.0-beta.1", 'project(VisionStudio VERSION 2.0.0 LANGUAGES CXX)\nset(VISION_STUDIO_APP_VERSION "2.0.0-beta.1")')]:
                with self.subTest(version=version):
                    (project / "CMakeLists.txt").write_text(declaration)
                    argv = ["build_deb.py", "--source-root", str(project),
                            "--qt-prefix", str(project / "qt"), "--runtime", str(project / "runtime")]
                    with patch.object(sys, "argv", argv), patch.object(builder, "build") as build:
                        self.assertEqual(builder.main(), 0)
                    args = build.call_args.args[0]
                    deb = metadata.debian_version(version)
                    self.assertEqual(args.version, deb)
                    self.assertEqual(args.output, project / "output/releases" / version / metadata.deb_filename(deb))
                    self.assertEqual(args.stage, project / "output/deb-stage" / deb)
                    self.assertFalse((project / "output").exists())

    def test_build_refuses_debian_and_source_version_mismatch_before_staging(self):
        with tempfile.TemporaryDirectory(prefix="vision-metadata-tests-") as directory:
            project = Path(directory)
            (project / "CMakeLists.txt").write_text('project(VisionStudio VERSION 2.0.0 LANGUAGES CXX)\nset(VISION_STUDIO_APP_VERSION "2.0.0-beta.1")')
            argv = ["build_deb.py", "--source-root", str(project), "--qt-prefix", str(project / "qt"),
                    "--runtime", str(project / "runtime"), "--version", "2.0.0-1"]
            with patch.object(sys, "argv", argv), patch.object(builder, "build") as build, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                builder.main()
            self.assertEqual(error.exception.code, 2)
            build.assert_not_called()
            self.assertFalse((project / "output").exists())

    def test_source_inventory_records_beta_application_and_debian_versions(self):
        with tempfile.TemporaryDirectory(prefix="vision-source-metadata-") as directory:
            project = Path(directory)
            release = project / "output/releases/2.0.0-beta.1"
            sources = release / "sources"
            sources.mkdir(parents=True)
            names = [f"{module}-everywhere-src-6.8.3.tar.xz" for module in source_builder.QT_MODULES]
            names += list(source_builder.REQUIRED_OTHER)
            provenance = []
            for name in names:
                content = f"temporary fixture: {name}".encode()
                (sources / name).write_bytes(content)
                provenance.append({"file": name, "sha256": hashlib.sha256(content).hexdigest(),
                                   "url": "https://example.invalid/test-only"})
            (sources / "DOWNLOAD-PROVENANCE.json").write_text(json.dumps(provenance))
            license_dir = project / "packaging/licenses"
            license_dir.mkdir(parents=True)
            (license_dir / "PYTHON-DISTRIBUTIONS.json").write_text('{"distributions": []}')
            source_builder.inventory(project, release, "2.0.0-beta.1")
            inventory = json.loads((release / "SOURCE-INVENTORY.json").read_text())
            self.assertEqual(inventory["version"], "2.0.0-beta.1")
            self.assertEqual(inventory["debian_version"], "2.0.0~beta.1-1")
            self.assertEqual(inventory["application_archive"], "vision-studio-2.0.0-beta.1-sources.tar.xz")
            self.assertEqual(inventory["complete_companion"], "vision-studio-2.0.0-beta.1-complete-source.tar.xz")


if __name__ == "__main__":
    unittest.main()
