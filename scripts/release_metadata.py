#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pure version metadata shared by local packaging, tag checks and publishing.

This project accepts stable X.Y.Z and numbered X.Y.Z-beta.N releases. Debian
uses '~beta.N' so a beta sorts before the corresponding final release.
No network calls, filesystem mutations or application runtime behavior live here.
"""
from __future__ import annotations

import re


_NUMBER = r"(0|[1-9][0-9]*)"
STABLE_VERSION_RE = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}\Z")
APPLICATION_VERSION_RE = re.compile(
    rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:-beta\.{_NUMBER})?\Z")
DEBIAN_VERSION_RE = re.compile(
    rf"({_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:~beta\.{_NUMBER})?)-([1-9][0-9]*)\Z")


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Semantic precedence, including numeric beta ordering and final > beta."""
    match = APPLICATION_VERSION_RE.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"Expected X.Y.Z or X.Y.Z-beta.N version, got {version!r}")
    major, minor, patch, beta = match.groups()
    return (int(major), int(minor), int(patch), int(beta is None),
            int(beta) if beta is not None else 0)


def is_prerelease(version: str) -> bool:
    return version_key(version)[3] == 0


def debian_upstream(version: str) -> str:
    version_key(version)
    return version.replace("-beta.", "~beta.")


def debian_version(version: str, revision: int = 1) -> str:
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("Debian package revision must be a positive integer")
    return f"{debian_upstream(version)}-{revision}"


def application_version_from_debian(version: str) -> str:
    match = DEBIAN_VERSION_RE.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"Expected X.Y.Z-N or X.Y.Z~beta.N-N project Debian version, got {version!r}")
    result = match.group(1).replace("~beta.", "-beta.")
    version_key(result)
    return result


def deb_filename(debian_package_version: str) -> str:
    """Safe flat GitHub filename; dpkg still sorts the version inside the DEB."""
    application = application_version_from_debian(debian_package_version)
    revision = debian_package_version.rsplit("-", 1)[1]
    return f"vision-studio_{application}-{revision}_amd64.deb"


def cmake_application_version(text: str) -> str:
    """Read exact project metadata, retaining compatibility with old stable tags."""
    text = re.sub(r"#[^\n]*", "", text)
    projects = re.findall(
        r"\bproject\s*\(\s*VisionStudio\s+VERSION\s+(\d+\.\d+\.\d+)(?=\s|\))",
        text, re.IGNORECASE)
    if len(projects) != 1:
        raise ValueError("Expected exactly one numeric VisionStudio project version in CMakeLists.txt")
    project_version = projects[0]
    if STABLE_VERSION_RE.fullmatch(project_version) is None:
        raise ValueError("CMake project version must use canonical X.Y.Z")
    assignments = re.findall(r"\bset\s*\(\s*VISION_STUDIO_APP_VERSION\b([^)]*)\)",
                             text, re.IGNORECASE)
    if not assignments:
        # Frozen versions before beta support contain project(VERSION) only.
        return project_version
    if len(assignments) != 1:
        raise ValueError("VISION_STUDIO_APP_VERSION must be assigned exactly once")
    value = assignments[0].strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    core = version_key(value)[:3]
    if core != version_key(project_version)[:3]:
        raise ValueError("Application and numeric CMake project versions differ")
    return value
