#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create source companions and checksums from this release's actual files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

VERSION = "1.1.0"
SOURCE_ITEMS = ("src", "tests", "scripts", "packaging", "docs", "assets", "models", "vendor",
                "CMakeLists.txt", "resources.qrc", "run.sh", "VisionStudio.desktop", "README.md",
                "LICENSE", "requirements-pt.txt", "requirements-pt.lock.txt", "runtime-info.json",
                ".clang-format", ".gitignore")
SKIP = {".git", "__pycache__", ".pytest_cache", ".cache"}
QT_MODULES = ("qtbase", "qtsvg", "qtwayland", "qtimageformats")
REQUIRED_OTHER = ("ultralytics-8.4.173.tar.gz", "ultralytics_thop-2.2.2.tar.gz",
                  "opencv-python-headless-4.11.0.86.tar.gz", "opencv-python-86.tar.gz",
                  "opencv-python-86-Dockerfile_x86_64", "ffmpeg-5.1.6.tar.xz")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inventory(project, release):
    provenance = json.loads((release / "sources/DOWNLOAD-PROVENANCE.json").read_text())
    downloads = {Path(row["file"]).name: row for row in provenance}
    rows = []
    names = [f"{module}-everywhere-src-6.8.3.tar.xz" for module in QT_MODULES] + list(REQUIRED_OTHER)
    for name in names:
        path = release / "sources" / name
        if not path.is_file():
            raise RuntimeError(f"Required original source missing: {path}")
        checksum = sha256(path)
        if name in downloads:
            if checksum != downloads[name]["sha256"]:
                raise RuntimeError(f"Downloaded source checksum mismatch: {name}")
            url = downloads[name]["url"]
            verification = "checked against preserved download provenance"
        else:
            url = "https://download.qt.io/archive/qt/6.8/6.8.3/submodules/" + name
            verification = "official .sha256 checked at download; local archive digest preserved"
        rows.append({"file": "sources/" + name, "bytes": path.stat().st_size,
                     "sha256": checksum, "url": url, "verification": verification})
    data = {"application": "Vision Studio", "version": VERSION, "debian_version": "1.1.0-1",
            "maintainer": "misaka_ning <1468549029@qq.com>",
            "application_license": "AGPL-3.0-only",
            "application_archive": f"vision-studio-{VERSION}-sources.tar.xz",
            "complete_companion": f"vision-studio-{VERSION}-complete-source.tar.xz",
            "public_checksums": "SHA256SUMS", "source_checksums": "SOURCE-SHA256SUMS",
            "upstream_archives": rows,
            "python_distributions": "packaging/licenses/PYTHON-DISTRIBUTIONS.json",
            "qt_license_records": "packaging/licenses/QT-SDK-PACKAGES.json",
            "ffmpeg_build": "packaging/licenses/FFMPEG-BUILD.json",
            "model_provenance": ["models/yolov8n.provenance.json", "models/yolov5n-pt.provenance.json",
                                 "models/MODEL-PROVENANCE.json"],
            "notes": "Application archive includes full project and vendored YOLOv5 source. "
                     "Complete companion includes that archive and all listed original upstream archives. "
                     "System-provided Python and shared libraries are Debian dependencies."}
    write_json(project / "docs/release/SOURCE-INVENTORY.json", data)
    write_json(release / "SOURCE-INVENTORY.json", data)
    distributions = json.loads((project / "packaging/licenses/PYTHON-DISTRIBUTIONS.json").read_text())
    lock = "# Actual bundled release distributions; pip excluded.\n" + "\n".join(
        f'{row["name"]}=={row["version"]}' for row in distributions["distributions"]) + "\n"
    (project / "docs/release/requirements-release.lock.txt").write_text(lock)
    return names


def file_list(root, items):
    for item in items:
        path = root / item
        if not path.exists():
            raise RuntimeError(f"Required application source missing: {item}")
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if any(part in SKIP for part in relative.parts) or candidate.suffix in {".pyc", ".pyo"}:
                continue
            if candidate.is_symlink():
                target = candidate.resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"External symlink not allowed in source: {relative}")
            if candidate.is_file() or candidate.is_symlink():
                yield candidate, relative.as_posix()


def make_archive(destination, entries, prefix, overwrite, preset=3):
    if destination.exists() and not overwrite:
        raise RuntimeError(f"Source archive already exists: {destination}; use --overwrite to rebuild")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "1791158400"))
    try:
        with tarfile.open(temporary, "w:xz", preset=preset) as archive:
            for path, relative in entries:
                info = archive.gettarinfo(str(path), arcname=prefix + "/" + relative)
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = epoch
                if info.isfile():
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--release-directory", type=Path, default=Path("output/releases"))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.source_root.resolve()
    release = args.release_directory.resolve()
    names = inventory(project, release)
    if args.inventory_only:
        print("Verified source inventory and release dependency lock written")
        return
    application = release / f"vision-studio-{VERSION}-sources.tar.xz"
    make_archive(application, file_list(project, SOURCE_ITEMS), f"vision-studio-{VERSION}", args.overwrite)
    split_files = [application, release / "SOURCE-INVENTORY.json",
                   release / "sources/DOWNLOAD-PROVENANCE.json"]
    split_files += [release / "sources" / name for name in names]
    source_checksums = release / "SOURCE-SHA256SUMS"
    source_checksums.write_text("".join(
        f"{sha256(path)}  {path.relative_to(release).as_posix()}\n" for path in split_files))
    # Put the original source archives inside the complete companion unchanged.
    entries = [(application, application.name), (release / "SOURCE-INVENTORY.json", "SOURCE-INVENTORY.json"),
               (source_checksums, "SOURCE-SHA256SUMS")]
    entries += [(release / "sources" / name, "sources/" + name) for name in names]
    entries += [(release / "sources/DOWNLOAD-PROVENANCE.json", "sources/DOWNLOAD-PROVENANCE.json")]
    entries += [(project / "docs/release" / name, "docs/" + name)
                for name in ("对应源码与重建.md", "发行说明.md", "第三方许可清单.md", "requirements-release.lock.txt")]
    entries += [(project / "LICENSE", "LICENSE"),
                (project / "docs/release/README-SOURCES.md", "README.md")]
    complete = release / f"vision-studio-{VERSION}-complete-source.tar.xz"
    make_archive(complete, entries, f"vision-studio-{VERSION}-complete-source", args.overwrite, preset=0)
    deb = release / "vision-studio_1.1.0-1_amd64.deb"
    if not deb.is_file():
        raise RuntimeError("The final DEB is required before public SHA256SUMS can be generated")
    files = [deb, complete]
    (release / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(release).as_posix()}\n" for path in files))
    print(f"Created {application.name} and {complete.name}; public and source checksums written")


if __name__ == "__main__":
    main()
