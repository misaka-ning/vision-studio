#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify a final Vision Studio 2.2.0 corresponding-source delivery read-only.

No archives are extracted and no application or source module is executed.
Only the explicitly requested, previously absent report is written. The final
DEB is streamed through dpkg-deb to bind embedded sources and ELF to staging.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import email.parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import time


VERSION = "2.2.0"
FOREIGN = {"docs/足球模型微调.md", "scripts/finetune_ball.py",
           "scripts/evaluate_ball.py", "scripts/run_finetune_ball.sh"}
SKIP = {".git", "__pycache__", ".pytest_cache", ".cache"}
REQUIRED_NOTICES = {
    "LICENSE", "vendor/yolov5/LICENSE", "vendor/onnxruntime/LICENSE",
    "vendor/onnxruntime/PROVENANCE.txt", "packaging/copyright",
    "packaging/licenses/PYTHON-DISTRIBUTIONS.json", "packaging/licenses/QT-SDK-PACKAGES.json",
    "packaging/licenses/FFMPEG-BUILD.json", "packaging/licenses/netron/LICENSE",
    "packaging/licenses/netron/dagre-LICENSE.txt", "packaging/licenses/netron/graphlib-LICENSE.txt",
    "packaging/licenses/model-conversion/PROVENANCE.json",
    "packaging/licenses/model-conversion/onnx-1.17.0-LICENSE",
    "packaging/licenses/model-conversion/protobuf-33.0-LICENSE",
    "packaging/licenses/model-conversion/protobuf-cpp-3.21.12-LICENSE",
}


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def sha256(path):
    with path.open("rb") as stream:
        return digest_stream(stream)[0]


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in value and "\x00" not in value,
            f"Unsafe relative path: {value!r}")
    require(path.as_posix() == value, f"Noncanonical relative path: {value!r}")
    return path


def regular(root, relative):
    safe_relative(relative)
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        require(not parent.is_symlink(), f"Symlink input is not accepted: {parent}")
    require(path.is_file(), f"Required regular file missing: {path}")
    return path


def archive_relative(name, prefix):
    require(name.startswith(prefix + "/"), f"Unexpected archive prefix: {name!r}")
    relative = name[len(prefix) + 1:]
    safe_relative(relative)
    return relative


def excluded(relative):
    path = PurePosixPath(relative)
    return bool(SKIP.intersection(path.parts)) or path.suffix in {".pyc", ".pyo"}


def parse_manifest(root, name):
    result = {}
    for line in regular(root, name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"Malformed checksum line in {name}: {line!r}")
        checksum, relative = match.groups()
        require(relative not in result, f"Duplicate manifest entry in {name}: {relative}")
        path = regular(root, relative)
        actual = sha256(path)
        require(actual == checksum.lower(), f"Checksum mismatch: {relative}")
        result[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    require(bool(result), f"Empty manifest: {name}")
    return result


def source_constants(clean):
    """Read literal build definitions without importing or executing source."""
    tree = ast.parse(regular(clean, "docs/release/build_source_archive.py").read_text(encoding="utf-8"))
    wanted = {"SOURCE_ITEMS", "QT_MODULES", "REQUIRED_OTHER", "SKIP"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == wanted, "Cannot read frozen source archive definitions")
    require(set(values["SKIP"]) == SKIP, "Source exclusion rules changed; verifier review required")
    return values


def collect_source(clean, items):
    rows = {}
    for item in items:
        safe_relative(item)
        root = clean / item
        require(root.exists(), f"Required SOURCE_ITEMS entry missing: {item}")
        candidates = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in candidates:
            relative = path.relative_to(clean).as_posix()
            if excluded(relative) or (path.is_dir() and not path.is_symlink()):
                continue
            require(relative not in FOREIGN, f"Foreign user file found in frozen source: {relative}")
            regular(clean, relative)
            rows[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                              "mode": 0o755 if path.stat().st_mode & 0o111 else 0o644}
    require(REQUIRED_NOTICES.issubset(rows), "Frozen source lacks required license/provenance files: "
            + ", ".join(sorted(REQUIRED_NOTICES - rows.keys())))
    require("runtime-info.json" in rows and "src/main.cpp" in rows, "Actual runtime info or application source missing")
    return rows


def verify_complete(release, clean, inventory, source_manifest, companion):
    prefix = f"vision-studio-{VERSION}-complete-source"
    expected = dict(source_manifest)
    expected["SOURCE-SHA256SUMS"] = {
        "sha256": sha256(regular(release, "SOURCE-SHA256SUMS")),
        "bytes": (release / "SOURCE-SHA256SUMS").stat().st_size}
    for name in ("对应源码与重建.md", "发行说明.md", "第三方许可清单.md", "requirements-release.lock.txt"):
        path = regular(clean, "docs/release/" + name)
        expected["docs/" + name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for relative, source in (("LICENSE", "LICENSE"), ("README.md", "docs/release/README-SOURCES.md")):
        path = regular(clean, source)
        expected[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    seen = set()
    with tarfile.open(release / companion, mode="r|xz") as archive:
        for member in archive:
            relative = archive_relative(member.name, prefix)
            require(member.isfile(), f"Complete companion member is not regular: {relative}")
            require(relative in expected and relative not in seen, f"Unexpected/duplicate complete member: {relative}")
            checksum, size = digest_stream(archive.extractfile(member))
            require({"sha256": checksum, "bytes": size} == expected[relative]
                    and size == member.size, f"Complete companion member mismatch: {relative}")
            seen.add(relative)
    require(seen == set(expected), "Complete companion lacks members: " + ", ".join(sorted(set(expected) - seen)))
    return len(seen)


def verify_application(release, name, expected):
    seen = set()
    with tarfile.open(release / name, mode="r|xz") as archive:
        for member in archive:
            relative = archive_relative(member.name, f"vision-studio-{VERSION}")
            require(relative not in FOREIGN, f"Foreign user file in application archive: {relative}")
            require(relative in expected and relative not in seen, f"Unexpected/duplicate application member: {relative}")
            require(member.isfile(), f"Nonregular application source member: {relative}")
            checksum, size = digest_stream(archive.extractfile(member))
            require(checksum == expected[relative]["sha256"] and size == expected[relative]["bytes"]
                    and size == member.size, f"Application source differs from frozen source: {relative}")
            require(member.mode == expected[relative]["mode"] and member.uid == member.gid == 0,
                    f"Application source normalized mode/ownership mismatch: {relative}")
            seen.add(relative)
    require(seen == set(expected), "Application archive lacks frozen files: " + ", ".join(sorted(set(expected) - seen)))
    return len(seen)


def canonical_distribution(name):
    return re.sub(r"[-_.]+", "-", name.lower())


def mapped_stage_files(clean, stage):
    """Reproduce build_deb's copied/generated text inputs without modifying them."""
    expected = {}
    def add_file(source, target):
        path = regular(clean, source)
        expected[target] = {"sha256": sha256(path), "bytes": path.stat().st_size, "source": source}
    def add_tree(source, target, omit=None):
        require((clean / source).is_dir(), f"Source tree missing: {source}")
        for path in sorted((clean / source).rglob("*")):
            relative = path.relative_to(clean).as_posix()
            if excluded(relative) or (path.is_dir() and not path.is_symlink()):
                continue
            tail = path.relative_to(clean / source).as_posix()
            if omit and omit(tail):
                continue
            add_file(relative, target + "/" + tail)
    def generated(target, data):
        expected[target] = {"sha256": digest_bytes(data), "bytes": len(data), "source": "generated-by-build_deb.py"}

    app = "opt/VisionStudio"
    for name in ("assets", "models", "vendor"):
        add_tree(name, app + "/" + name, (lambda tail: tail.startswith("source/")) if name == "models" else None)
    for name in ("pt_worker.py", "netron_server.py", "model_convert.py", "gpu_setup.py", "gpu_probe.py", "setup_gpu.sh"):
        add_file("scripts/" + name, app + "/scripts/" + name)
    for name in ("requirements-gpu.txt", "requirements-gpu.lock.txt"):
        add_file(name, app + "/" + name)
    for path in sorted((clean / "docs").glob("*.md")):
        if path.name != "验证报告.md":
            add_file("docs/" + path.name, app + "/docs/" + path.name)
    add_tree("docs/release", app + "/docs/release", lambda tail:
             PurePosixPath(tail).suffix == ".log" or ".log.tmp" in PurePosixPath(tail).name)
    add_file("packaging/run-installed.sh", app + "/run-installed.sh")
    add_file("packaging/copyright", "usr/share/doc/vision-studio/copyright")
    add_file("LICENSE", "usr/share/doc/vision-studio/LICENSE")
    add_file("packaging/README.md", "usr/share/doc/vision-studio/README.Debian")
    add_tree("packaging/licenses", "usr/share/doc/vision-studio/licenses")
    add_file("assets/app-icon.svg", "usr/share/icons/hicolor/scalable/apps/vision-studio.svg")

    release_info_path = regular(stage, app + "/release-info.json")
    release_info = json.loads(release_info_path.read_text(encoding="utf-8"))
    require(release_info.get("package") == "vision-studio" and release_info.get("version") == VERSION + "-1"
            and release_info.get("install_root") == "/opt/VisionStudio", "Staging release identity is not 2.2.0-1 /opt/VisionStudio")
    packages = release_info["python_runtime"]["distributions"]
    declared = {canonical_distribution(k): v for k, v in packages.items()}
    license_manifest = json.loads(regular(clean, "packaging/licenses/PYTHON-DISTRIBUTIONS.json").read_text(encoding="utf-8"))
    licensed = {canonical_distribution(row["name"]): row["version"] for row in license_manifest["distributions"]}
    actual = {}
    runtime_members = {}
    for metadata in sorted((stage / app / "runtime/lib/python3.10/site-packages").glob("*.dist-info/METADATA")):
        relative = metadata.relative_to(stage).as_posix()
        regular(stage, relative)
        info = email.parser.BytesParser().parsebytes(metadata.read_bytes())
        require(info.get("Name") and info.get("Version"), f"Invalid runtime METADATA: {relative}")
        name = canonical_distribution(info["Name"])
        require(name not in actual, f"Duplicate runtime distribution: {name}")
        actual[name] = info["Version"]
        runtime_members[relative] = {"sha256": sha256(metadata), "bytes": metadata.stat().st_size,
                                     "source": "actual-runtime-METADATA"}
    require(actual and actual == declared == licensed, "Actual staging distributions, release metadata and source license manifest differ")
    for name, version in {"torch": "2.9.1+cpu", "netron": "9.3.1", "onnx": "1.17.0",
                          "protobuf": "6.33.0", "opencv-python-headless": "4.11.0.86"}.items():
        require(actual.get(name) == version, f"Unexpected runtime pin: {name}")
    require("opencv-python" not in actual and "pip" not in actual, "Unexpected GUI OpenCV or pip in packaged runtime")
    lock = "\n".join(f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda row: row[0].lower()))
    generated(app + "/requirements-pt.lock.txt", ("# Actual distributions bundled in this release (pip excluded).\n" + lock + "\n").encode())
    generated(app + "/requirements-pt.txt", ("# Release runtime: see lock for every distribution.\n"
        "# Install this complete lock with pip --no-deps to retain headless OpenCV.\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n-r requirements-pt.lock.txt\n").encode())
    runtime_info = json.loads(regular(clean, "runtime-info.json").read_text(encoding="utf-8"))
    runtime_info = {key: value for key, value in runtime_info.items() if not key.startswith("verified_")}
    runtime_info.update({"interpreter": "/opt/VisionStudio/runtime/bin/python", "base_interpreter": "/usr/bin/python3.10",
        "runtime_directory": "/opt/VisionStudio/runtime", "opencv_python": ".".join(packages["opencv-python-headless"].split(".")[:3]),
        "opencv_python_package": packages["opencv-python-headless"], "opencv_python_distribution": "opencv-python-headless",
        "resolved_distributions": packages})
    generated(app + "/runtime-info.json", (json.dumps(runtime_info, ensure_ascii=False, indent=2) + "\n").encode())
    copied = dict(expected)
    expected.update(runtime_members)
    for relative in (app + "/build/bin/vision-studio", app + "/release-info.json"):
        path = regular(stage, relative)
        expected[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size, "source": "final-staging"}
    require(release_info["binary_sha256"] == expected[app + "/build/bin/vision-studio"]["sha256"],
            "Staging release-info binary hash differs from actual staged C++ ELF")
    for relative, row in expected.items():
        path = regular(stage, relative)
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                f"Staged source/generated file differs from frozen input: {relative}")
    # Do not silently ignore extra scripts, copied models, documentation or license files.
    for subtree in (app + "/scripts", app + "/assets", app + "/models", app + "/vendor",
                    app + "/docs", "usr/share/doc/vision-studio"):
        for path in (stage / subtree).rglob("*"):
            if path.is_dir() and not path.is_symlink():
                continue
            relative = path.relative_to(stage).as_posix()
            require(relative in expected, f"Unmapped staged source/document: {relative}")
    return expected, copied, actual


def verify_deb(deb, stage, expected, report):
    command = ["dpkg-deb", "--field", str(deb)]
    report["commands"].append(command)
    control = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", timeout=30).stdout
    fields = email.parser.Parser().parsestr(control)
    require(fields.get("Package") == "vision-studio" and fields.get("Version") == VERSION + "-1"
            and fields.get("Architecture") == "amd64", "Unexpected DEB control identity")
    staged_fields = email.parser.Parser().parsestr(regular(stage, "DEBIAN/control").read_text(encoding="utf-8"))
    require(dict(fields.items()) == dict(staged_fields.items()), "DEB control differs from final staging")
    command = ["dpkg-deb", "--fsys-tarfile", str(deb)]
    report["commands"].append(command)
    seen = set()
    all_seen = set()
    with tempfile.TemporaryFile() as error_stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_stream)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    name = member.name
                    while name.startswith("./"):
                        name = name[2:]
                    if name in ("", ".") and member.isdir():
                        continue
                    name = name.rstrip("/") if member.isdir() else name
                    safe_relative(name)
                    require(name not in all_seen, f"Duplicate DEB payload entry: {name}")
                    all_seen.add(name)
                    if name not in expected:
                        continue
                    require(member.isfile(), f"Expected regular source/ELF in DEB: {name}")
                    checksum, size = digest_stream(archive.extractfile(member))
                    require(checksum == expected[name]["sha256"] and size == expected[name]["bytes"]
                            and size == member.size, f"DEB source/ELF differs from frozen source or staging: {name}")
                    seen.add(name)
            require(process.wait(timeout=30) == 0, "dpkg-deb failed while streaming payload")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            error_stream.seek(0)
            report["details"]["dpkg_stderr"] = error_stream.read(1024 * 1024).decode("utf-8", errors="replace")
    require(seen == set(expected), "DEB lacks staged source/ELF members: " + ", ".join(sorted(set(expected) - seen)))
    return len(seen), len(all_seen)


def verify(args, report):
    clean, release, stage = args.clean_source, args.release_dir, args.stage
    require(release.name == VERSION, "Release directory name must equal 2.2.0")
    cmake = regular(clean, "CMakeLists.txt").read_text(encoding="utf-8")
    require(re.search(r'project\(VisionStudio\s+VERSION\s+2\.2\.0\b', cmake)
            and re.search(r'set\(VISION_STUDIO_APP_VERSION\s+"2\.2\.0"\)', cmake), "Frozen source version is not 2.2.0")
    print("Verifying public and source manifests...", flush=True)
    public = parse_manifest(release, "SHA256SUMS")
    sources = parse_manifest(release, "SOURCE-SHA256SUMS")
    app = f"vision-studio-{VERSION}-sources.tar.xz"
    companion = f"vision-studio-{VERSION}-complete-source.tar.xz"
    deb_name = f"vision-studio_{VERSION}-1_amd64.deb"
    require(set(public) == {deb_name, companion}, "Public manifest must list exactly the 2.2.0-1 DEB and complete source")
    require(all(row["bytes"] < 2 * 1024**3 for row in public.values()), "Public asset exceeds GitHub 2 GiB limit")
    inventory = json.loads(regular(release, "SOURCE-INVENTORY.json").read_text(encoding="utf-8"))
    require(inventory.get("version") == VERSION and inventory.get("debian_version") == VERSION + "-1"
            and inventory.get("application_archive") == app and inventory.get("complete_companion") == companion,
            "Source inventory version/archive identity mismatch")
    require(regular(clean, "docs/release/SOURCE-INVENTORY.json").read_bytes()
            == regular(release, "SOURCE-INVENTORY.json").read_bytes(), "Frozen and release inventory bytes differ")
    definitions = source_constants(clean)
    upstream_names = [f"{name}-everywhere-src-6.8.3.tar.xz" for name in definitions["QT_MODULES"]] + list(definitions["REQUIRED_OTHER"])
    require(len(upstream_names) == len(set(upstream_names)) == 18, "Expected exactly 18 original upstream inputs")
    expected_source_names = {app, "SOURCE-INVENTORY.json", "sources/DOWNLOAD-PROVENANCE.json"}
    expected_source_names.update("sources/" + name for name in upstream_names)
    require(set(sources) == expected_source_names, "Source checksum manifest member set is incomplete or unexpected")
    inventory_rows = {row["file"]: row for row in inventory["upstream_archives"]}
    require(len(inventory["upstream_archives"]) == len(inventory_rows) == 18
            and set(inventory_rows) == {"sources/" + name for name in upstream_names}, "Inventory must declare 18 distinct upstream inputs")
    for relative, row in inventory_rows.items():
        require(row["sha256"] == sources[relative]["sha256"] and row["bytes"] == sources[relative]["bytes"]
                and isinstance(row.get("url"), str) and row["url"].startswith("https://"), "Upstream inventory/provenance mismatch: " + relative)
    downloads = json.loads(regular(release, "sources/DOWNLOAD-PROVENANCE.json").read_text(encoding="utf-8"))
    for row in downloads:
        name = "sources/" + PurePosixPath(row["file"]).name
        if name in inventory_rows:
            require(row["sha256"] == sources[name]["sha256"], "Download provenance checksum mismatch: " + name)
    expected = collect_source(clean, definitions["SOURCE_ITEMS"])
    print(f"Verifying complete companion and {len(expected)} frozen application source files...", flush=True)
    complete_count = verify_complete(release, clean, inventory, sources, companion)
    source_count = verify_application(release, app, expected)
    print("Verifying staged embedded source mappings and runtime metadata...", flush=True)
    staged, embedded, distributions = mapped_stage_files(clean, stage)
    print(f"Verifying {len(staged)} embedded source/metadata/ELF members in actual DEB...", flush=True)
    deb_count, payload_count = verify_deb(release / deb_name, stage, staged, report)
    manifest = json.loads(regular(release, deb_name + ".manifest.json").read_text(encoding="utf-8"))
    binary = staged["opt/VisionStudio/build/bin/vision-studio"]
    require(manifest.get("sha256") == public[deb_name]["sha256"] and manifest.get("bytes") == public[deb_name]["bytes"]
            and manifest.get("binary_sha256") == binary["sha256"], "DEB build manifest does not bind final package and staged C++ binary")
    # Detect a concurrent input rewrite rather than reporting a mixed freeze.
    for name, row in {**public, **sources}.items():
        require(sha256(regular(release, name)) == row["sha256"], "Release input changed during verification: " + name)
    for relative, row in expected.items():
        require(sha256(regular(clean, relative)) == row["sha256"], "Frozen source changed during verification: " + relative)
    report.update({"deb_sha256_confirmed": public[deb_name]["sha256"],
        "application_source_sha256": sources[app]["sha256"], "complete_source_sha256": public[companion]["sha256"],
        "main_assets": {**public, "SHA256SUMS": {"sha256": sha256(release / "SHA256SUMS"), "bytes": (release / "SHA256SUMS").stat().st_size}},
        "upstream_archives_verified": 18, "complete_companion_members_verified": complete_count,
        "frozen_application_sources_byte_equal": source_count, "embedded_source_files_sha256_verified": len(embedded),
        "deb_embedded_files_verified": deb_count, "runtime_distribution_count": len(distributions),
        "foreign_user_files_excluded": sorted(FOREIGN),
        "application_source_members": expected, "embedded_source_mappings": embedded})
    report["details"].update({"deb_sha256": public[deb_name]["sha256"], "deb_bytes": public[deb_name]["bytes"],
        "packaged_binary_sha256": binary["sha256"], "stage_binary_sha256": binary["sha256"],
        "deb_payload_member_count": payload_count, "runtime_distributions": distributions,
        "input_tree_unchanged_during_verification": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-source", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    for name in ("clean_source", "release_dir", "stage"):
        setattr(args, name, getattr(args, name).resolve())
        if not getattr(args, name).is_dir():
            parser.error(f"Input directory does not exist: {getattr(args, name)}")
    args.report = args.report.absolute()
    if args.report.exists() or args.report.is_symlink():
        parser.error("Report already exists; refusing to overwrite evidence")
    actual_report = args.report.parent.resolve() / args.report.name
    if any(actual_report.is_relative_to(root) for root in (args.clean_source, args.stage)):
        parser.error("Report must be outside frozen source and staging inputs")
    started = time.monotonic()
    report = {"schema_version": 1, "version": VERSION, "success": False, "errors": [],
        "scope": "Read-only full source member hashing, original upstream provenance, embedded source and final DEB/staging ELF identity; no application inference executed",
        "clean_export": str(args.clean_source), "release_directory": str(args.release_dir), "stage_directory": str(args.stage),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "verifier_sha256": sha256(Path(__file__).resolve()), "commands": [], "details": {}}
    try:
        verify(args, report)
    except (OSError, ValueError, KeyError, TypeError, VerificationError, tarfile.TarError,
            subprocess.SubprocessError, SyntaxError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["success"] = not report["errors"] and report.get("upstream_archives_verified") == 18
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"success": report["success"], "errors": report["errors"], "report": str(args.report),
        "application_files": report.get("frozen_application_sources_byte_equal"),
        "embedded_files": report.get("embedded_source_files_sha256_verified"),
        "seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
