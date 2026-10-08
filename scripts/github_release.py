#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Publish locally tested packages, verify GitHub assets, and retain two versions.

Requires an authenticated GitHub CLI; never reads or stores its token. This tool
does not build packages, create/move tags, overwrite assets, or install software.
The read-only GitHub Actions workflow validates source/tag metadata separately.

Examples:
  python3 scripts/github_release.py publish --version 1.6.0 \
    --release-dir output/releases/1.6.0 --notes-file docs/releases/1.6.0.md --dry-run
  python3 scripts/github_release.py publish --version 1.6.0 \
    --release-dir output/releases/1.6.0 --notes-file docs/releases/1.6.0.md
  python3 scripts/github_release.py verify --version 1.6.0 \
    --release-dir output/releases/1.6.0
  python3 scripts/github_release.py prune --releases-root output/releases
  python3 scripts/github_release.py prune --releases-root output/releases --execute

Interrupted uploads remain drafts. --resume-draft explicitly resumes a matching
draft without replacing anything. Published releases are always refused by the
publish command; use verify to audit them. prune is a dry run unless --execute.

Reference: https://cli.github.com/manual/gh_release_create
Reference: https://docs.github.com/en/rest/releases/assets
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_metadata


DEFAULT_REPOSITORY = "misaka-ning/vision-studio"
# Retention intentionally recognizes stable directories only. Beta publication
# must not displace a stable backup or make previews automatic cleanup targets.
VERSION_RE = release_metadata.STABLE_VERSION_RE
REPOSITORY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
CHECKSUM_RE = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)\Z")
PROTECTED = {".git", "runtime", "output", "recordings", "preferences.ini", "history.json"}
OPTIONAL_REPORTS = (
    "desktop-qa.json", "metadata-boundary-qa.json", "home-qa.json",
    "source-bundle-qa.json", "source-qa.json", "ctest-qa.json", "gpu-qa.json",
)
GPU_CHECKS = ("onnx_cuda_nodes", "pt_cuda", "packaged_onnx_cuda", "packaged_pt_cuda",
              "cpu_regression", "grayscale_c1_c3", "stereo_recording")


class ReleaseError(RuntimeError):
    pass


def version_key(version: str) -> tuple[int, int, int, int, int]:
    try:
        return release_metadata.version_key(version)
    except ValueError as error:
        raise ReleaseError(str(error)) from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_directory(path: Path) -> Path:
    absolute = path.absolute()
    # Resolve neither a symlink release directory nor any symlink ancestor.
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ReleaseError(f"Symlink directory is not allowed: {component}")
    if not absolute.is_dir():
        raise ReleaseError(f"Directory does not exist: {absolute}")
    return absolute.resolve()


def safe_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or "\\" in name or any(ord(c) < 32 for c in name)):
        raise ReleaseError(f"Unsafe manifest path: {name!r}")
    path = root.joinpath(*relative.parts)
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ReleaseError(f"Symlink file is not allowed: {component}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ReleaseError(f"Missing regular file: {path}")
    return path


def read_manifest(root: Path, name: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in safe_file(root, name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = CHECKSUM_RE.fullmatch(line)
        if not match:
            raise ReleaseError(f"Malformed checksum line in {name}: {line!r}")
        digest, filename = match.groups()
        if filename in entries:
            raise ReleaseError(f"Duplicate checksum path in {name}: {filename}")
        actual = sha256(safe_file(root, filename))
        if actual != digest.lower():
            raise ReleaseError(f"Checksum mismatch: {filename}")
        entries[filename] = actual
    if not entries:
        raise ReleaseError(f"Empty checksum manifest: {name}")
    return entries


@dataclass(frozen=True)
class Asset:
    name: str
    path: Path
    size: int
    sha256: str

    def metadata(self) -> dict:
        return {"name": self.name, "bytes": self.size, "sha256": self.sha256}


def make_asset(root: Path, name: str) -> Asset:
    # GitHub assets have a flat namespace; upstream sources live in the companion.
    if PurePosixPath(name).name != name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ReleaseError(f"Unsafe GitHub asset name: {name!r}")
    path = safe_file(root, name)
    return Asset(name, path, path.stat().st_size, sha256(path))


def validate_bundle(directory: Path, version: str, require_qa: bool = True) -> list[Asset]:
    """Validate the complete local delivery; return only the three public assets."""
    version_key(version)
    root = canonical_directory(directory)
    if root.name != version:
        raise ReleaseError("Release directory name must exactly equal the application version")
    public = read_manifest(root, "SHA256SUMS")
    sources = read_manifest(root, "SOURCE-SHA256SUMS")
    overlap = public.keys() & sources.keys()
    if any(public[name] != sources[name] for name in overlap):
        raise ReleaseError("Conflicting checksum manifests")
    all_checksums = {**sources, **public}
    app = f"vision-studio-{version}-sources.tar.xz"
    companion = f"vision-studio-{version}-complete-source.tar.xz"
    debs = sorted(name for name in public if re.fullmatch(
        rf"vision-studio_{re.escape(version)}-[1-9]\d*_amd64\.deb", name))
    if len(debs) != 1:
        raise ReleaseError("SHA256SUMS must contain exactly one versioned amd64 DEB")
    if set(public) != {debs[0], companion}:
        raise ReleaseError("Public SHA256SUMS must list exactly the DEB and complete source companion")
    for name in (companion, app, "SOURCE-INVENTORY.json"):
        if name not in all_checksums:
            raise ReleaseError(f"Required asset is not checksummed: {name}")
    inventory = json.loads(safe_file(root, "SOURCE-INVENTORY.json").read_text())
    if (inventory.get("version") != version or inventory.get("application_archive") != app
            or inventory.get("complete_companion") != companion):
        raise ReleaseError("SOURCE-INVENTORY.json does not match this version")
    if "debian_version" in inventory:
        try:
            declared_deb = release_metadata.deb_filename(inventory["debian_version"])
        except ValueError as error:
            raise ReleaseError("SOURCE-INVENTORY.json has an invalid Debian version") from error
        if declared_deb != debs[0]:
            raise ReleaseError("SOURCE-INVENTORY.json Debian version does not match the DEB")
    # The complete source companion contains the application source archive,
    # SOURCE-SHA256SUMS, and SOURCE-INVENTORY.json. Keep validating the original
    # local copies above; they do not need separate Release attachments.
    assets = [make_asset(root, name) for name in (debs[0], companion, "SHA256SUMS")]
    if require_qa:
        required_reports = ["qa-report.json", "install-qa.json", "ctest-qa.json"]
        if version_key(version)[:3] >= (1, 6, 0):
            required_reports.append("gpu-qa.json")
        source_report = next((name for name in ("source-bundle-qa.json", "source-qa.json")
                              if (root / name).is_file()), None)
        if source_report is None:
            raise ReleaseError("Required passing source companion report is missing")
        required_reports.append(source_report)
        reports = list(dict.fromkeys(required_reports + [name for name in OPTIONAL_REPORTS
                                                         if (root / name).is_file()]))
        for name in reports:
            report = json.loads(safe_file(root, name).read_text())
            if report.get("success") is not True or report.get("errors"):
                raise ReleaseError(f"Release QA is not passing: {name}")
            if "version" in report and report["version"] != version:
                raise ReleaseError(f"QA report version mismatch: {name}")
            if name in ("qa-report.json", "install-qa.json", "gpu-qa.json"):
                if report.get("details", {}).get("deb_sha256") != all_checksums[debs[0]]:
                    raise ReleaseError(f"QA report is for a different DEB: {name}")
            if name == "gpu-qa.json":
                checks = report.get("checks", {})
                if (report.get("schema_version") != 1 or not isinstance(checks, dict) or
                        any(checks.get(check) is not True for check in GPU_CHECKS)):
                    raise ReleaseError("GPU QA must prove actual and packaged CUDA inference, CPU regression, grayscale and stereo recording")
            # QA remains mandatory even though reports are archived in the Git
            # repository under docs/releases/evidence rather than as assets.
            make_asset(root, name)
    return assets


def command(arguments: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(arguments, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ReleaseError(f"Command failed ({result.returncode}): {arguments[0]} "
                           f"{arguments[1] if len(arguments) > 1 else ''}\n{result.stderr.strip()}")
    return result.stdout


class GitHub:
    def __init__(self, repository: str):
        if not REPOSITORY_RE.fullmatch(repository) or repository.startswith("-"):
            raise ReleaseError("Repository must use OWNER/REPO format")
        self.repository = repository

    def api(self, endpoint: str) -> object:
        return json.loads(command(["gh", "api", "-H", "Accept: application/vnd.github+json",
                                   f"repos/{self.repository}/{endpoint}"]))

    def release(self, tag: str) -> dict | None:
        result = subprocess.run(["gh", "api", f"repos/{self.repository}/releases/tags/{quote(tag, safe='')}"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        if result.returncode:
            if "(HTTP 404)" not in result.stderr:
                raise ReleaseError(f"Could not read release {tag}: {result.stderr.strip()}")
        else:
            return json.loads(result.stdout)
        # The by-tag endpoint returns published releases only. Authenticated
        # users with push access can find drafts through the releases list.
        # Do not treat permission, network, or malformed list errors as absence.
        matches = []
        for page in range(1, 1001):
            rows = self.api(f"releases?per_page=100&page={page}")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ReleaseError("Malformed GitHub releases list")
            matches.extend(row for row in rows if row.get("tag_name") == tag)
            if len(matches) > 1:
                raise ReleaseError(f"Multiple releases use {tag}; inspect drafts manually")
            if len(rows) < 100:
                return matches[0] if matches else None
        raise ReleaseError("Unreasonable number of releases")

    def assets(self, release: dict) -> list[dict]:
        assets: list[dict] = []
        for page in range(1, 1001):
            rows = self.api(f"releases/{release['id']}/assets?per_page=100&page={page}")
            assets.extend(rows)
            if len(rows) < 100:
                return assets
        raise ReleaseError("Unreasonable number of release assets")

    def remote_commit(self, tag: str) -> str:
        ref = self.api(f"git/ref/tags/{quote(tag, safe='')}")
        value = ref["object"]
        for _ in range(8):
            if value["type"] == "commit":
                return value["sha"]
            if value["type"] != "tag":
                break
            value = self.api(f"git/tags/{value['sha']}")["object"]
        raise ReleaseError("Tag does not resolve to a commit")

    def check_asset(self, expected: Asset, remote: dict) -> dict:
        if remote.get("state") != "uploaded" or remote.get("size") != expected.size:
            raise ReleaseError(f"Incomplete asset or size mismatch: {expected.name}")
        digest = remote.get("digest")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            actual = digest[7:].lower()
            method = "GitHub REST sha256 digest"
        else:
            # Older assets may lack digest. Download through gh's authenticated
            # API into a temporary directory, never trust an arbitrary URL.
            with tempfile.TemporaryDirectory(prefix="vision-release-verify-") as temp:
                destination = Path(temp) / "asset"
                with destination.open("wb") as stream:
                    result = subprocess.run([
                        "gh", "api", "-H", "Accept: application/octet-stream",
                        f"repos/{self.repository}/releases/assets/{remote['id']}"],
                        stdout=stream, stderr=subprocess.PIPE, check=False)
                if result.returncode:
                    raise ReleaseError(f"Asset download failed: {expected.name}")
                if destination.stat().st_size != expected.size:
                    raise ReleaseError(f"Downloaded asset size mismatch: {expected.name}")
                actual = sha256(destination)
            method = "downloaded sha256"
        if actual != expected.sha256:
            raise ReleaseError(f"Remote checksum mismatch: {expected.name}")
        return {**expected.metadata(), "asset_id": remote["id"], "verification": method}

    def verify_assets(self, release: dict, expected: list[Asset],
                      allow_missing: bool = False) -> list[dict]:
        rows = self.assets(release)
        by_name: dict[str, dict] = {}
        for row in rows:
            if row["name"] in by_name:
                raise ReleaseError(f"Duplicate remote asset: {row['name']}")
            by_name[row["name"]] = row
        verified = []
        for asset in expected:
            if asset.name not in by_name:
                if allow_missing:
                    continue
                raise ReleaseError(f"Missing remote asset: {asset.name}")
            verified.append(self.check_asset(asset, by_name[asset.name]))
        return verified


def cmake_version(text: str) -> str:
    try:
        return release_metadata.cmake_application_version(text)
    except ValueError as error:
        raise ReleaseError(str(error)) from error


def check_release_type(value: dict, version: str) -> None:
    expected = release_metadata.is_prerelease(version)
    if value.get("tag_name") != f"v{version}" or value.get("prerelease") is not expected:
        raise ReleaseError("Release tag or prerelease type does not match the application version")


def local_tag_commit(project: Path, version: str) -> str:
    version_key(version)
    tag = f"v{version}"
    local = command(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], project).strip()
    tagged_cmake = command(["git", "show", f"refs/tags/{tag}:CMakeLists.txt"], project)
    if cmake_version(tagged_cmake) != version:
        raise ReleaseError("Tag's CMakeLists.txt version does not match release version")
    return local


def verify_published_tag(github: GitHub, project: Path, version: str) -> str:
    """Verify an archived version without requiring that it is the current HEAD."""
    local = local_tag_commit(project, version)
    if github.remote_commit(f"v{version}") != local:
        raise ReleaseError("Local and GitHub tag point to different commits")
    return local


def verify_tag(github: GitHub, project: Path, version: str) -> str:
    tag = f"v{version}"
    local = local_tag_commit(project, version)
    head = command(["git", "rev-parse", "HEAD"], project).strip()
    if local != head:
        raise ReleaseError("Publish from the tagged commit; local tag does not equal HEAD")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], project).strip():
        raise ReleaseError("Tracked source changes must be committed before publishing")
    if github.remote_commit(tag) != local:
        raise ReleaseError("Local and GitHub tag point to different commits")
    return local


def release_after_mutation(github: GitHub, tag: str, draft: bool) -> dict:
    """Allow bounded read consistency delays; propagate permissions/network errors."""
    for attempt in range(4):
        value = github.release(tag)
        if (value is not None and value.get("draft") is draft
                and (draft or value.get("published_at"))):
            return value
        if attempt < 3:
            time.sleep(1 << attempt)
    state = "draft" if draft else "published"
    raise ReleaseError(f"Expected {state} release after mutation: {tag}")


def publish(args: argparse.Namespace, github: GitHub) -> dict:
    assets = validate_bundle(args.release_dir, args.version)
    prerelease = release_metadata.is_prerelease(args.version)
    notes = args.notes_file.resolve(strict=True)
    if not notes.is_file() or not notes.read_text(encoding="utf-8").strip():
        raise ReleaseError("Release notes file is empty")
    commit = verify_tag(github, args.project_dir.resolve(), args.version)
    tag = f"v{args.version}"
    existing = github.release(tag)
    if existing:
        if not existing.get("draft") or not args.resume_draft:
            raise ReleaseError("Release already exists; only --resume-draft may continue a draft")
        if (existing.get("tag_name") != tag or existing.get("prerelease") is not prerelease
                or existing.get("body", "").strip() != notes.read_text(encoding="utf-8").strip()):
            raise ReleaseError("Existing draft has different tag, notes, or release type")
        allowed_names = {asset.name for asset in assets}
        if any(row["name"] not in allowed_names for row in github.assets(existing)):
            raise ReleaseError("Existing draft contains unexpected assets; inspect it manually")
        github.verify_assets(existing, assets, allow_missing=True)
    plan = {"success": True, "dry_run": args.dry_run, "repository": github.repository,
            "tag": tag, "commit": commit, "prerelease": prerelease,
            "latest": not prerelease, "assets": [item.metadata() for item in assets]}
    if args.dry_run:
        return plan
    if existing is None:
        create = ["gh", "release", "create", tag, "--repo", github.repository,
                  "--verify-tag", "--draft", "--title", f"Vision Studio {tag}",
                  "--notes-file", str(notes)]
        if prerelease:
            create += ["--prerelease", "--latest=false"]
        command(create)
    release = release_after_mutation(github, tag, draft=True)
    check_release_type(release, args.version)
    verified = github.verify_assets(release, assets, allow_missing=True)
    present = {row["name"] for row in verified}
    for asset in assets:
        if asset.name not in present:
            command(["gh", "release", "upload", tag, str(asset.path), "--repo",
                     github.repository])  # Never use --clobber.
    verified = github.verify_assets(release, assets)
    # A concurrent tag mutation must not publish mismatched code.
    if github.remote_commit(tag) != commit:
        raise ReleaseError("GitHub tag changed during upload; release remains a draft")
    edit = ["gh", "release", "edit", tag, "--repo", github.repository, "--draft=false"]
    edit += ["--prerelease", "--latest=false"] if prerelease else ["--latest"]
    command(edit)
    release = release_after_mutation(github, tag, draft=False)
    check_release_type(release, args.version)
    plan.update({"url": release["html_url"], "verified_assets": verified,
                 "published": True})
    return plan


def verify(args: argparse.Namespace, github: GitHub) -> dict:
    assets = validate_bundle(args.release_dir, args.version, require_qa=not args.legacy)
    tag = f"v{args.version}"
    release = github.release(tag)
    if release is None or release.get("draft") or not release.get("published_at"):
        raise ReleaseError("A published release is required")
    check_release_type(release, args.version)
    commit = verify_published_tag(github, args.project_dir.resolve(), args.version)
    return {"success": True, "repository": github.repository, "tag": tag,
            "commit": commit, "url": release["html_url"], "prerelease": release["prerelease"],
            "verified_assets": github.verify_assets(release, assets)}


def cleanup_root(path: Path) -> Path:
    root = canonical_directory(path)
    if root.name != "releases":
        raise ReleaseError("Only a directory named releases can be pruned")
    if root.parent.name == "output":
        project = root.parent.parent
    else:
        project = root.parent
    if not (project / "CMakeLists.txt").is_file() or not (project / "src").is_dir():
        raise ReleaseError("Prune root must be <project>/output/releases or <project>/releases")
    return root


def inspect_cleanup_tree(directory: Path) -> tuple[int, int]:
    original = directory.stat()
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if name in PROTECTED or stat.S_ISLNK(mode) or not (
                    stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ReleaseError(f"Unsafe or protected cleanup entry: {path}")
            if path.lstat().st_dev != original.st_dev:
                raise ReleaseError(f"Cleanup cannot cross a filesystem: {path}")
    return original.st_dev, original.st_ino


def cleanup_candidates(root: Path) -> tuple[list[Path], list[Path]]:
    directories = []
    for path in root.iterdir():
        if VERSION_RE.fullmatch(path.name):
            if path.is_symlink():
                raise ReleaseError(f"Symlink version directory is not allowed: {path}")
            if path.is_dir():
                directories.append(path)
    versions = sorted(directories,
                      key=lambda path: version_key(path.name), reverse=True)
    # Unknown names, historical root-level files and active builds are untouched.
    return versions[:2], versions[2:]


def prune(args: argparse.Namespace, github: GitHub) -> dict:
    root = cleanup_root(args.releases_root)
    kept, candidates = cleanup_candidates(root)
    plan = {"success": True, "dry_run": not args.execute, "root": str(root),
            "kept": [str(path) for path in kept], "verified_kept": [],
            "verified_candidates": []}
    # Preflight the entire plan before any deletion. Any failure protects all.
    identities = {}
    verified_commits = {}
    # A draft, absent release, or incomplete upload must never displace a fully
    # published version in the two-version retention policy. Fail closed rather
    # than guessing which locally present versions are safe to count.
    for retained in kept:
        directory = canonical_directory(retained)
        if directory.parent != root:
            raise ReleaseError("Retained version escaped its parent")
        assets = validate_bundle(directory, directory.name, require_qa=False)
        release = github.release(f"v{directory.name}")
        if (release is None or release.get("draft") or release.get("prerelease")
                or not release.get("published_at")):
            raise ReleaseError(f"Retained version is not durably published: {directory.name}")
        commit = verify_published_tag(github, args.project_dir.resolve(), directory.name)
        verified = github.verify_assets(release, assets)
        verified_commits[directory] = commit
        plan["verified_kept"].append({"path": str(directory), "url": release["html_url"],
                                      "commit": commit, "assets": verified})
    for candidate in candidates:
        directory = canonical_directory(candidate)
        if directory.parent != root:
            raise ReleaseError("Cleanup candidate escaped its parent")
        identities[directory] = inspect_cleanup_tree(directory)
        assets = validate_bundle(directory, directory.name, require_qa=False)
        release = github.release(f"v{directory.name}")
        if (release is None or release.get("draft") or release.get("prerelease")
                or not release.get("published_at")):
            raise ReleaseError(f"Version is not durably published: {directory.name}")
        commit = verify_published_tag(github, args.project_dir.resolve(), directory.name)
        verified = github.verify_assets(release, assets)
        verified_commits[directory] = commit
        plan["verified_candidates"].append({"path": str(directory),
                                            "url": release["html_url"], "commit": commit,
                                            "assets": verified})
    if args.execute:
        if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            raise ReleaseError("This platform does not provide safe descriptor-based rmtree")
        # Re-evaluate protection in case a new version appeared during remote QA.
        new_kept, new_candidates = cleanup_candidates(root)
        if kept != new_kept or candidates != new_candidates:
            raise ReleaseError("Local versions changed during verification; retry cleanup")
        for path, identity in identities.items():
            if inspect_cleanup_tree(path) != identity or path.parent != root:
                raise ReleaseError("Candidate identity changed during verification")
        # Both the retained releases and candidates must still use the exact
        # tags verified in preflight before any local snapshot is deleted.
        for path, expected_commit in verified_commits.items():
            if verify_published_tag(github, args.project_dir.resolve(), path.name) != expected_commit:
                raise ReleaseError("Version tag changed during cleanup verification")
        removed = []
        for path in identities:
            shutil.rmtree(path)
            removed.append(str(path))
        plan["removed"] = removed
    return plan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = result.add_subparsers(dest="operation", required=True)
    for operation in ("publish", "verify"):
        item = sub.add_parser(operation)
        item.add_argument("--repo", default=DEFAULT_REPOSITORY)
        item.add_argument("--version", required=True)
        item.add_argument("--release-dir", required=True, type=Path)
        item.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent.parent,
                          help="Git source checkout containing local version tags")
        if operation == "publish":
            item.add_argument("--notes-file", required=True, type=Path)
            item.add_argument("--dry-run", action="store_true", help="Validate locally/remotely; do not publish")
            item.add_argument("--resume-draft", action="store_true")
        else:
            item.add_argument("--legacy", action="store_true", help="Verify critical archives/manifests without newer QA reports")
    item = sub.add_parser("prune")
    item.add_argument("--repo", default=DEFAULT_REPOSITORY)
    item.add_argument("--releases-root", required=True, type=Path)
    item.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent.parent,
                      help="Git source checkout containing local version tags")
    item.add_argument("--execute", action="store_true", help="Remove only remotely verified old version directories")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        github = GitHub(args.repo)
        result = {"publish": publish, "verify": verify, "prune": prune}[args.operation](args, github)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
