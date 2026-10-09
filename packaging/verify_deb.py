#!/usr/bin/env python3
"""Audit a Vision Studio DEB without installing it or changing system state.

Static checks inspect control metadata, archive ownership/permissions, packaged
paths, symlinks, launchers, Qt plugins, ELF runtime paths and system dependencies.
--smoke additionally extracts a private copy, removes its write permissions and
runs real ONNX, modern PT and legacy PT inference with isolated HOME/XDG paths.
Only this script's newly-created temporary directory is ever chmod'ed/removed.
"""

import argparse
import base64
import configparser
from email.parser import BytesParser
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_metadata


PREFIX = PurePosixPath("opt/VisionStudio")
REQUIRED = (
    "opt/VisionStudio/build/bin/vision-studio",
    "opt/VisionStudio/build/bin/qt.conf",
    "opt/VisionStudio/run-installed.sh",
    "opt/VisionStudio/runtime/bin/python",
    "opt/VisionStudio/scripts/pt_worker.py",
    "opt/VisionStudio/scripts/netron_server.py",
    "opt/VisionStudio/libexec/QtWebEngineProcess",
    "opt/VisionStudio/libexec/qt.conf",
    "opt/VisionStudio/lib/qt/libQt6WebEngineCore.so.6",
    "opt/VisionStudio/lib/qt/libQt6WebEngineWidgets.so.6",
    "opt/VisionStudio/lib/qt/libQt6WebChannel.so.6",
    "opt/VisionStudio/resources/qtwebengine_resources.pak",
    "opt/VisionStudio/resources/qtwebengine_devtools_resources.pak",
    "opt/VisionStudio/resources/qtwebengine_resources_100p.pak",
    "opt/VisionStudio/resources/qtwebengine_resources_200p.pak",
    "opt/VisionStudio/resources/icudtl.dat",
    "opt/VisionStudio/resources/v8_context_snapshot.bin",
    "opt/VisionStudio/translations/qtwebengine_locales/en-US.pak",
    "opt/VisionStudio/models/yolov5n.onnx",
    "opt/VisionStudio/models/yolov8n.pt",
    "opt/VisionStudio/models/yolov5n.pt",
    "opt/VisionStudio/assets/bus.jpg",
    "opt/VisionStudio/vendor/yolov5/models/yolo.py",
    "opt/VisionStudio/plugins/platforms/libqxcb.so",
    "opt/VisionStudio/plugins/platforms/libqoffscreen.so",
    "opt/VisionStudio/plugins/imageformats/libqjpeg.so",
    "opt/VisionStudio/plugins/imageformats/libqsvg.so",
    "opt/VisionStudio/plugins/imageformats/libqwebp.so",
    "opt/VisionStudio/plugins/iconengines/libqsvgicon.so",
    "usr/bin/vision-studio",
)
HOME_REFERENCE = re.compile(r"/(?:home|root)/[^\s\"'<>\n]+")


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.details = {}

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def warn(self, message):
        self.warnings.append(message)


def command(arguments, *, env=None, timeout=60, check=True):
    result = subprocess.run(arguments, env=env, timeout=timeout, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"{arguments[0]} exited {result.returncode}: {result.stderr[-1800:]}")
    return result


def deb_fields(text):
    fields = {}
    key = None
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and key:
            fields[key] += "\n" + line.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


def archive_members(deb, audit):
    process = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)], stdout=subprocess.PIPE)
    members = {}
    installed_bytes = 0
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            name = member.name.removeprefix("./").rstrip("/")
            if not name or name == ".":
                continue
            path = PurePosixPath(name)
            audit.require(not path.is_absolute() and ".." not in path.parts,
                          f"Unsafe archive path: {member.name}")
            if path.is_absolute() or ".." in path.parts:
                continue
            audit.require(name not in members, f"Duplicate archive entry: {name}")
            members[name] = member
            installed_bytes += member.size if member.isfile() else 0
            allowed = (path == PREFIX or PREFIX in path.parents or name in ("opt", "usr", "usr/bin", "usr/share")
                       or name == "usr/bin/vision-studio" or name.startswith("usr/share/"))
            audit.require(allowed, f"Unexpected installed path outside application/integration directories: {name}")
            audit.require(member.uid == 0 and member.gid == 0,
                          f"Installed entry is not root:root: {name} ({member.uid}:{member.gid})")
            if not member.issym():
                audit.require(not member.mode & 0o022, f"Group/other writable installed entry: {name}")
                audit.require(not member.mode & 0o7000, f"Special permission bits on installed entry: {name}")
            audit.require(member.isfile() or member.isdir() or member.issym() or member.islnk(),
                          f"Unexpected device/FIFO archive entry: {name}")
            if member.issym() or member.islnk():
                target = member.linkname
                audit.require(not HOME_REFERENCE.search(target), f"Symlink depends on a developer home: {name} -> {target}")
                if target.startswith("/"):
                    audit.require(member.issym() and target == "/usr/bin/python3.10",
                                  f"Unexpected external absolute symlink: {name} -> {target}")
                else:
                    resolved = posixpath.normpath(posixpath.join(str(path.parent) if member.issym() else "", target))
                    audit.require(not resolved.startswith("../") and resolved != "..",
                                  f"Archive link escapes extraction directory: {name} -> {target}")
            parts = path.parts
            private = {".git", ".aws", ".ssh", ".codex", ".agents", ".env", "preferences.ini", "history.json"}
            audit.require(not private.intersection(parts), f"Development/user state included in DEB: {name}")
            application_output = PREFIX / "output"
            audit.require(not (application_output in path.parents and not name.endswith("/.gitkeep")),
                          f"Runtime output included in DEB: {name}")
    process.stdout.close()
    audit.require(process.wait() == 0, "dpkg-deb could not read the data archive")
    for name in REQUIRED:
        audit.require(name in members, f"Required application file/plugin missing: {name}")
    audit.details["archive"] = {"entries": len(members), "uncompressed_file_bytes": installed_bytes}
    return members


def dependency_groups(value):
    groups = []
    pattern = re.compile(r"^([a-z0-9][a-z0-9+.-]*)(?::[a-z0-9-]+)?\s*(?:\(([<>=]+)\s*([^\)]+)\))?")
    for group in value.split(","):
        alternatives = []
        for alternative in group.split("|"):
            match = pattern.match(alternative.strip())
            if match:
                alternatives.append(match.groups())
        if alternatives:
            groups.append(alternatives)
    return groups


def installed_packages():
    output = command(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Status}\t${Depends}\t${Pre-Depends}\n"]).stdout
    packages = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) >= 5 and fields[2] == "install ok installed":
            packages[fields[0].split(":")[0]] = (fields[1], fields[3] + "," + fields[4])
    return packages


def version_matches(version, operator, minimum):
    if not operator:
        return True
    operators = {">=": "ge", "<=": "le", "=": "eq", ">>": "gt", "<<": "lt"}
    return command(["dpkg", "--compare-versions", version, operators.get(operator, operator), minimum], check=False).returncode == 0


def audit_dependencies(control, audit):
    packages = installed_packages()
    closure = set()
    pending = []
    roots = dependency_groups(control.get("Depends", ""))
    audit.require(bool(roots), "DEB has no runtime Depends")
    for group in roots:
        satisfied = [name for name, operator, minimum in group if name in packages
                     and version_matches(packages[name][0], operator, minimum)]
        audit.require(bool(satisfied), "Host does not satisfy declared dependency: " + " | ".join(item[0] for item in group))
        pending.extend(satisfied[:1])
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        for group in dependency_groups(packages.get(name, ("", ""))[1]):
            present = [item[0] for item in group if item[0] in packages]
            pending.extend(present[:1])
    for required in ("python3.10", "libopencv-core4.5d", "libopencv-imgproc4.5d",
                     "libopencv-imgcodecs4.5d", "libopencv-dnn4.5d", "libopencv-videoio4.5d",
                     "libxcb-cursor0", "libxkbcommon-x11-0"):
        audit.require(required in closure, f"Declared dependencies do not cover required Ubuntu22.04 runtime: {required}")
    audit.details["dependencies"] = {"declared": control.get("Depends", ""),
                                     "host_required_versions": {name: packages[name][0] for name in sorted(closure) if name in packages},
                                     "closure_packages": len(closure)}
    return closure


def audit_native_dependencies(app, elf_paths, closure, audit):
    """Compare actual loader resolutions with Debian's declared dependency closure."""
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    external = set()
    bundled = set()
    missing = []
    loader_contexts = {}
    for path in elf_paths:
        relative = str(path.relative_to(app))
        search = [app / "lib/qt"]
        # torchvision imports torch before its extensions; libtorch libraries
        # have already been loaded into the Python process at that point.
        site_packages = next((parent for parent in path.parents if parent.name == "site-packages"), None)
        if site_packages and path.is_relative_to(site_packages / "torchvision"):
            search.append(site_packages / "torch/lib")
        # auditwheel's entry extension supplies the search path for its private
        # peer libraries. Auditing an individual peer requires that context.
        if path.parent.name.endswith(".libs"):
            search.append(path.parent)
        current_environment = dict(environment)
        current_environment["LD_LIBRARY_PATH"] = ":".join(str(directory) for directory in search)
        if len(search) > 1:
            loader_contexts[relative] = [str(directory.relative_to(app)) for directory in search]
        result = command(["ldd", str(path)], env=current_environment, timeout=30, check=False)
        for line in result.stdout.splitlines():
            if "=> not found" in line:
                missing.append(f"{relative}: {line.strip()}")
            match = re.search(r"(?:=>\s*)?(/[^\s]+)\s+\(0x[0-9a-fA-F]+\)", line)
            if not match:
                continue
            # Resolve ../ and symlink aliases before deciding whether a loader
            # result really belongs to the extracted payload.
            resolved = Path(match.group(1)).resolve()
            if resolved.is_relative_to(app):
                bundled.add(str(resolved.relative_to(app)))
            else:
                external.add(str(resolved))
                audit.require(not HOME_REFERENCE.search(str(resolved)),
                              f"ELF dependency resolved through a developer home: {relative}: {resolved}")
        if result.returncode and "not a dynamic executable" not in result.stderr:
            audit.require(False, f"Cannot resolve native dependencies for {relative}: {result.stderr[-500:]}")
    audit.require(not missing, "Unresolved native libraries: " + "; ".join(missing))
    # Ubuntu's merged /usr means dpkg may register a different spelling from ldd.
    variants = {}
    for path in sorted(external):
        choices = {path, str(Path(path).resolve())}
        if path.startswith(("/lib/", "/lib64/")):
            choices.add("/usr" + path)
        elif path.startswith(("/usr/lib/", "/usr/lib64/")):
            choices.add(path.removeprefix("/usr"))
        variants[path] = choices
    queried = sorted(set().union(*variants.values())) if variants else []
    owners = {}
    for start in range(0, len(queried), 80):
        result = command(["dpkg-query", "-S", *queried[start:start + 80]], check=False)
        for line in result.stdout.splitlines():
            if ": " not in line:
                continue
            package, path = line.rsplit(": ", 1)
            owners.setdefault(path, set()).update(item.strip().split(":")[0] for item in package.split(","))
    resolved_owners = {}
    for path, choices in variants.items():
        packages = set().union(*(owners.get(choice, set()) for choice in choices))
        resolved_owners[path] = sorted(packages)
        audit.require(bool(packages), f"External native library is not supplied by an installed Debian package: {path}")
        audit.require(bool(packages.intersection(closure)),
                      f"Native dependency is absent from declared dependency closure: {path} ({', '.join(sorted(packages))})")
    audit.details["native_dependencies"] = {"bundled_libraries": sorted(bundled),
                                            "system_library_packages": resolved_owners,
                                            "unresolved": missing, "component_loader_contexts": loader_contexts}


def audit_control(deb, directory, audit):
    command(["dpkg-deb", "-e", str(deb), str(directory)])
    control = deb_fields((directory / "control").read_text(encoding="utf-8"))
    for key in ("Package", "Version", "Architecture", "Maintainer", "Description", "Installed-Size"):
        audit.require(bool(control.get(key)), f"Required control field missing: {key}")
    audit.require(control.get("Architecture") == "amd64", "Release architecture must be amd64")
    audit.require(control.get("Package") == "vision-studio", "Unexpected package name (expected vision-studio)")
    try:
        audit.details["application_version"] = release_metadata.application_version_from_debian(
            control.get("Version", ""))
    except ValueError as error:
        audit.require(False, str(error))
    audit.require(not HOME_REFERENCE.search(json.dumps(control)), "Control metadata contains developer home paths")
    scripts = []
    for script in ("preinst", "postinst", "prerm", "postrm", "config"):
        path = directory / script
        if path.exists():
            content = path.read_text(encoding="utf-8")
            scripts.append(script)
            audit.require(content.startswith("#!/"), f"Maintainer script has no interpreter: {script}")
            audit.require(not HOME_REFERENCE.search(content), f"Maintainer script touches a home directory: {script}")
            audit.require(not re.search(r"\b(?:pip|pip3|curl|wget|git\s+clone|apt-get\s+install)\b", content),
                          f"Maintainer script installs/downloads runtime dependencies: {script}")
            command(["sh", "-n", str(path)])
    audit.details["control"] = control
    audit.details["maintainer_scripts"] = scripts
    return control


def canonical_distribution(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def python_metadata(path):
    with path.open("rb") as stream:
        header = BytesParser().parse(stream, headersonly=True)
    return header.get("Name", ""), header.get("Version", "")


def audit_python_runtime(app, docs, audit):
    packages = {}
    for metadata in (app / "runtime/lib").glob("python*/site-packages/*.dist-info/METADATA"):
        name, version = python_metadata(metadata)
        key = canonical_distribution(name)
        audit.require(bool(key) and bool(version) and key not in packages,
                      f"Missing/duplicate Python distribution metadata: {metadata.relative_to(app)}")
        packages[key] = version
    for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python-headless", "pandas", "seaborn", "tqdm", "netron"):
        audit.require(name in packages, f"Required runtime Python distribution missing: {name}")
    audit.require(packages.get("netron") == "9.3.1", "The deployed Netron viewer version differs from pinned 9.3.1")
    if release_metadata.version_key(audit.details["application_version"])[:3] >= (2, 1, 0):
        for name, version in (("onnx", "1.17.0"), ("protobuf", "6.33.0")):
            audit.require(packages.get(name) == version,
                          f"Model conversion requires {name}=={version} in the bundled runtime")
    audit.require("opencv-python" not in packages and "opencv-contrib-python" not in packages,
                  "Graphical OpenCV wheel is present instead of the intended headless runtime")
    audit.require(all("+cpu" in packages.get(name, "") for name in ("torch", "torchvision")),
                  "The release does not contain CPU-only PyTorch/torchvision wheels")
    audit.require(not list((app / "runtime").rglob("libQt5*.so*")), "Headless PT runtime unexpectedly includes Qt5 libraries")
    lock = {}
    lock_file = app / "requirements-pt.lock.txt"
    if lock_file.exists():
        for line in lock_file.read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.startswith("#"):
                name, version = line.strip().split("==", 1)
                lock[canonical_distribution(name)] = version
    for name, version in packages.items():
        audit.require(lock.get(name) == version,
                      f"Python lock does not match installed distribution: {name}=={version} (lock:{lock.get(name)})")
    for name in set(lock) - set(packages) - {"pip"}:
        audit.require(False, f"Python lock lists a distribution missing from the installed runtime: {name}")
    info = json.loads((app / "runtime-info.json").read_text(encoding="utf-8"))
    for name in ("torch", "torchvision", "ultralytics", "numpy"):
        audit.require(info.get(name) == packages.get(name), f"runtime-info.json does not match actual {name} version")
    audit.require(info.get("opencv_python_package") == packages.get("opencv-python-headless"),
                  "runtime-info.json OpenCV wheel version does not match headless runtime")
    audit.require(info.get("interpreter") == "/opt/VisionStudio/runtime/bin/python" and info.get("device") == "cpu"
                  and info.get("torch_cuda_build") is None and info.get("cuda_available") is False,
                  "runtime-info.json has the wrong installation path or CPU backend information")
    manifest_path = docs / "licenses/PYTHON-DISTRIBUTIONS.json"
    if audit.require(manifest_path.exists(), "Python license/distribution manifest missing"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = {canonical_distribution(item["name"]): item["version"] for item in manifest.get("distributions", [])}
        audit.require(declared == packages, "Python copyright/distribution manifest does not match shipped distributions")
        audit.require(manifest.get("runtime") == "/opt/VisionStudio/runtime", "Python license manifest contains the wrong runtime path")
        notices = {}
        for metadata in (docs / "licenses/python").glob("*/METADATA"):
            name, version = python_metadata(metadata)
            notices[canonical_distribution(name)] = (version, metadata.parent)
        for name, version in packages.items():
            audit.require(notices.get(name, (None,))[0] == version, f"Preserved Python METADATA/copyright missing: {name}=={version}")
        for distribution in manifest.get("distributions", []):
            name = canonical_distribution(distribution["name"])
            notice = notices.get(name)
            if not notice:
                continue
            for relative, digest in distribution.get("preserved_files", {}).items():
                preserved = notice[1] / relative
                audit.require(preserved.is_file(), f"Preserved Python license file missing: {name}/{relative}")
                if preserved.is_file():
                    audit.require(hashlib.sha256(preserved.read_bytes()).hexdigest() == digest,
                                  f"Preserved Python license file differs from manifest: {name}/{relative}")
    audit.details["python_runtime"] = {"count": len(packages), "distributions": packages,
                                       "lock_matches_runtime": all(lock.get(name) == version for name, version in packages.items())}


def audit_extracted(root, control, closure, audit):
    app = root / str(PREFIX)
    version = release_metadata.version_key(
        release_metadata.application_version_from_debian(control.get("Version", "")))[:3]
    if version >= (2, 1, 0):
        audit.require((app / "scripts/model_convert.py").is_file(), "Model conversion helper missing")
    if version >= (1, 6, 0):
        for relative in ("scripts/gpu_setup.py", "scripts/gpu_probe.py", "scripts/setup_gpu.sh",
                         "requirements-gpu.txt", "requirements-gpu.lock.txt"):
            audit.require((app / relative).is_file(), "GPU support configuration file missing: " + relative)
        audit.require("python3.10-venv" in control.get("Depends", ""), "Optional GPU setup requires python3.10-venv")
        audit.require(not (app / "gpu-runtime").exists(), "User GPU runtime must not be bundled in the CPU DEB")
        gpu_lock = app / "requirements-gpu.lock.txt"
        if gpu_lock.is_file():
            text = gpu_lock.read_text(encoding="utf-8")
            for pin in ("torch==2.9.1+cu128", "torchvision==0.24.1+cu128", "onnxruntime-gpu==1.23.2"):
                audit.require(pin in text, "GPU dependency pin missing: " + pin)
            audit.require("--hash=sha256:" in text, "GPU dependencies lack publisher wheel hashes")
        audit.details["optional_gpu"] = {"bundled": False, "setup": "per-user explicit request",
                                          "torch": "2.9.1+cu128", "onnxruntime": "1.23.2"}
    desktop_files = list((root / "usr/share/applications").glob("*.desktop"))
    audit.require(bool(desktop_files), "No system desktop entry provided")
    for desktop in desktop_files:
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        parser.optionxform = str
        parser.read(desktop, encoding="utf-8")
        section = parser["Desktop Entry"] if parser.has_section("Desktop Entry") else {}
        audit.require(section.get("Type") == "Application", "Desktop entry Type is not Application")
        audit.require(bool(section.get("Name")) and bool(section.get("Icon")), "Desktop entry lacks Name/Icon")
        audit.require(section.get("Exec", "").split(" ")[0] in ("vision-studio", "/usr/bin/vision-studio"),
                      "Desktop entry does not use the installed launcher")
        audit.require(not HOME_REFERENCE.search(desktop.read_text()), "Desktop entry depends on a developer home")
        if shutil.which("desktop-file-validate"):
            result = command(["desktop-file-validate", str(desktop)], check=False)
            audit.require(result.returncode == 0, "Desktop validation failed: " + result.stdout + result.stderr)
    for relative in ("run-installed.sh", "runtime/pyvenv.cfg", "runtime-info.json"):
        path = app / relative
        if path.exists():
            audit.require(not HOME_REFERENCE.search(path.read_text(encoding="utf-8")),
                          f"Installed runtime text contains a developer home path: {relative}")
    wrapper = root / "usr/bin/vision-studio"
    if wrapper.is_file():
        content = wrapper.read_text(encoding="utf-8")
        audit.require("/opt/VisionStudio/run-installed.sh" in content, "System wrapper does not start the installed application")
        audit.require(not HOME_REFERENCE.search(content), "System wrapper contains a developer home path")
        command(["sh", "-n", str(wrapper)])
    for relative in ("run-installed.sh", "build/bin/vision-studio", "libexec/QtWebEngineProcess"):
        path = app / relative
        audit.require(path.exists() and os.access(path, os.X_OK), f"Installed launcher/binary is not executable: {relative}")
    application_binary = app / "build/bin/vision-studio"
    if application_binary.is_file():
        developer_paths = re.findall(rb"/(?:home|root)/[^\x00\s\"'<>]+", application_binary.read_bytes())
        audit.require(not developer_paths,
                      "Release application embeds developer home paths: " + "; ".join(
                          path.decode("utf-8", "replace")[:250] for path in sorted(set(developer_paths))))
    runtime_bin = app / "runtime/bin"
    for path in runtime_bin.iterdir() if runtime_bin.exists() else ():
        if path.is_symlink():
            audit.require(not HOME_REFERENCE.search(os.readlink(path)), f"Runtime interpreter link contains a home path: {path.name}")
        elif path.is_file():
            with path.open("rb") as stream:
                first_line = stream.readline(4096)
            audit.require(not HOME_REFERENCE.search(first_line.decode("utf-8", "replace")),
                          f"Runtime entrypoint shebang contains a home path: {path.name}")
    python = runtime_bin / "python"
    audit.require(python.exists() and python.resolve() == Path("/usr/bin/python3.10").resolve(),
                  "Packaged runtime interpreter does not resolve to the declared Python3.10")
    qt_config = configparser.ConfigParser(interpolation=None)
    qt_config.optionxform = str
    qt_config.read(app / "build/bin/qt.conf")
    qt_paths = qt_config["Paths"] if qt_config.has_section("Paths") else {}
    audit.require(qt_paths.get("Prefix") == "../.." and qt_paths.get("Libraries") == "lib/qt"
                  and qt_paths.get("Plugins") == "plugins" and qt_paths.get("LibraryExecutables") == "libexec"
                  and qt_paths.get("Data") == "." and qt_paths.get("Translations") == "translations",
                  "qt.conf does not use the installed WebEngine relative layout")
    launcher = (app / "run-installed.sh").read_text(encoding="utf-8")
    for key in ("QTWEBENGINEPROCESS_PATH", "QTWEBENGINE_RESOURCES_PATH", "QTWEBENGINE_LOCALES_PATH"):
        audit.require(key in launcher, "Launcher has no explicit deployed WebEngine path: " + key)
    audit.require("QTWEBENGINE_DISABLE_SANDBOX" not in launcher and "--no-sandbox" not in launcher,
                  "The normal application launcher disables Chromium sandboxing")
    docs = root / "usr/share/doc" / control.get("Package", "vision-studio")
    audit.require((docs / "copyright").is_file(), "No Debian copyright/third-party notices file")
    audit.require((app / "docs/使用指南.md").is_file(), "Installed user manual missing")
    for license_name in ("AGPL-3.0.txt", "GPL-3.txt", "LGPL-3.txt", "LGPL-2.1.txt"):
        audit.require((docs / "licenses/texts" / license_name).is_file(), f"Full license text missing: {license_name}")
    for module in ("qtbase", "qtsvg", "qtwayland", "qtimageformats"):
        audit.require(bool(list((docs / "licenses/qt/sbom").glob(f"{module}-6.8.3.spdx*"))),
                      f"Bundled Qt6.8.3 license/SBOM inventory missing: {module}")
    qt_manifest = docs / "licenses/QT-SDK-PACKAGES.json"
    if audit.require(qt_manifest.is_file(), "Qt module/third-party source notice manifest is missing"):
        qt_records = {row["module"]: row for row in json.loads(qt_manifest.read_text())}
        for module in ("qtdeclarative", "qtwebchannel", "qtpositioning", "qtwebengine"):
            record = qt_records.get(module, {})
            audit.require(record.get("version") == "6.8.3" and bool(record.get("source_notice_files")),
                          "New Qt module lacks matching complete source notices: " + module)
    for name in ("LICENSE", "dagre-LICENSE.txt", "graphlib-LICENSE.txt", "THIRD-PARTY-PROVENANCE.json"):
        audit.require((docs / "licenses/netron" / name).is_file(), "Netron/frontend upstream notice missing: " + name)
    audit_python_runtime(app, docs, audit)
    elf_paths = []
    rpaths = {}
    for directory, _, files in os.walk(app):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            elf_paths.append(path)
            output = command(["readelf", "-d", str(path)], timeout=20).stdout
            values = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]*)\]", output)
            relative = str(path.relative_to(app))
            rpaths[relative] = values
            for value in values:
                audit.require(not HOME_REFERENCE.search(value), f"ELF runtime path refers to developer home: {relative}: {value}")
                audit.require(not re.search(r"/(?:tmp|var/tmp)/", value), f"ELF runtime path refers to a staging directory: {relative}: {value}")
    audit.require(any("$ORIGIN/../../lib/qt" in value for value in rpaths.get("build/bin/vision-studio", [])),
                  "Application has no relative RUNPATH to its bundled Qt libraries")
    audit.require(any("$ORIGIN/../lib/qt" in value for value in rpaths.get("libexec/QtWebEngineProcess", [])),
                  "QtWebEngineProcess has no relative RUNPATH to its bundled Qt libraries")
    audit.details["elf"] = {"count": len(elf_paths), "runtime_paths": rpaths}
    audit_native_dependencies(app, elf_paths, closure, audit)
    return app


def readonly_tree(tree):
    for directory, _, files in os.walk(tree, topdown=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        Path(directory).chmod(Path(directory).stat().st_mode & ~0o222)


def restore_temporary_permissions(tree):
    for directory, dirs, files in os.walk(tree):
        Path(directory).chmod(0o700)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)


def valid_model_display(output, report, exit_code):
    counts = (report.get("nodes", 0), report.get("hierarchy_items", 0), report.get("parameter_rows", 0))
    modes = report.get("display_modes", {})
    screenshots = (output / name for name in ("model-display.png", "model-hierarchy.png", "model-parameters.png"))
    return (exit_code == 0 and report.get("success") is True and report.get("cache_preserved") is True
            and all(isinstance(count, int) and not isinstance(count, bool) and count > 0 for count in counts)
            and isinstance(modes, dict) and all(modes.get(mode) is True for mode in ("graph", "hierarchy", "parameters"))
            and all(path.is_file() and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in screenshots))


def smoke_tests(app, workspace, audit):
    if not audit.require(os.geteuid() != 0, "Smoke tests must run as a non-root user"):
        return
    state = workspace / "user-state"
    state.mkdir()
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "HOME": str(state / "home"), "XDG_CONFIG_HOME": str(state / "config"),
           "XDG_DATA_HOME": str(state / "data"), "XDG_CACHE_HOME": str(state / "cache"),
           "XDG_STATE_HOME": str(state / "state"), "XDG_RUNTIME_DIR": str(state / "runtime"),
           "YOLO_CONFIG_DIR": str(state / "ultralytics"), "YOLOV5_CONFIG_DIR": str(state / "yolov5"),
           "MPLCONFIGDIR": str(state / "matplotlib"), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1", "QT_QPA_PLATFORM": "offscreen",
           # These two flags choose software rendering for this headless QA.
           # Chromium sandboxing remains enabled in both QA and the launcher.
           "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu", "QT_QUICK_BACKEND": "software"}
    for key in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
                "XDG_RUNTIME_DIR", "YOLO_CONFIG_DIR", "YOLOV5_CONFIG_DIR", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, mode=0o700)
    readonly_tree(app)
    probe = app / ".qa-write-probe"
    try:
        probe.write_text("write test")
        audit.errors.append("Extracted application tree was writable during smoke test")
        probe.unlink()
    except PermissionError:
        pass
    results = {}
    for name, flag in (("onnx", "--smoke"), ("pt", "--smoke-pt")):
        output = state / name
        started = time.monotonic()
        process = command([str(app / "run-installed.sh"), flag, str(output)], env=env, timeout=180, check=False)
        (state / f"{name}-stdout.log").write_text(process.stdout, encoding="utf-8")
        (state / f"{name}-stderr.log").write_text(process.stderr, encoding="utf-8")
        report_file = output / "smoke-report.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        passed = process.returncode == 0 and report.get("success") is True and report.get("predictions", 0) > 0
        audit.require(passed, f"Extracted read-only {name} smoke failed (exit{process.returncode}): {process.stderr[-1200:]}")
        results[name] = {"success": passed, "exit_code": process.returncode,
                         "elapsed_seconds": round(time.monotonic() - started, 2), "report": report}
    for name, model in (("model_display", "yolov5n.onnx"), ("model_display_pt", "yolov8n.pt")):
        output = state / name
        started = time.monotonic()
        process = command([str(app / "run-installed.sh"), "--smoke-model", str(output),
                           "--display-model", str(app / "models" / model)],
                          env=env, timeout=180, check=False)
        (state / f"{name}-stdout.log").write_text(process.stdout, encoding="utf-8")
        (state / f"{name}-stderr.log").write_text(process.stderr, encoding="utf-8")
        report_file = output / "model-display-report.json"
        report = json.loads(report_file.read_text()) if report_file.is_file() else {}
        passed = valid_model_display(output, report, process.returncode)
        audit.require(passed, f"Read-only non-root {model} three-view/cache smoke failed: " + process.stderr[-1800:])
        results[name] = {"success": passed, "exit_code": process.returncode,
                         "elapsed_seconds": round(time.monotonic() - started, 2), "report": report,
                         "rendering_flags": env["QTWEBENGINE_CHROMIUM_FLAGS"],
                         "sandbox_disabled_by_test": False}
    if release_metadata.version_key(release_metadata.application_version_from_debian(
            audit.details["control"]["Version"]))[:3] >= (2, 1, 0):
        for format in ("onnx", "torchscript"):
            name = "conversion_" + format
            output = state / name
            started = time.monotonic()
            process = command([str(app / "run-installed.sh"), "--smoke-conversion", str(output),
                               "--conversion-format", format], env=env, timeout=180, check=False)
            (state / f"{name}-stdout.log").write_text(process.stdout, encoding="utf-8")
            (state / f"{name}-stderr.log").write_text(process.stderr, encoding="utf-8")
            report_file = output / "conversion-smoke-report.json"
            report = json.loads(report_file.read_text()) if report_file.is_file() else {}
            artifact = report.get("conversion", {})
            artifact_file = Path(artifact.get("output", ""))
            passed = (process.returncode == 0 and report.get("success") is True
                      and report.get("actual_inference") is True and report.get("prediction_count", 0) > 0
                      and artifact.get("format") == format
                      and artifact_file.is_file() and hashlib.sha256(artifact_file.read_bytes()).hexdigest() == artifact.get("sha256"))
            audit.require(passed, "Read-only packaged model conversion/inference failed: " + process.stderr[-1800:])
            results[name] = {"success": passed, "exit_code": process.returncode,
                             "elapsed_seconds": round(time.monotonic() - started, 2), "report": report}
    image = base64.b64encode((app / "assets/bus.jpg").read_bytes()).decode("ascii")
    request = {"id": 1, "command": "infer", "image": image, "input_size": 640, "confidence": 0.25, "iou": 0.45}
    helper_env = dict(env)
    helper_env["VISION_STUDIO_HOME"] = str(app)
    process = subprocess.run([str(app / "runtime/bin/python"), str(app / "scripts/pt_worker.py"),
                              "--model", str(app / "models/yolov5n.pt"), "--task", "v5"],
                             input=json.dumps(request) + "\n" + '{"command":"quit"}\n', env=helper_env,
                             text=True, capture_output=True, timeout=120)
    (state / "legacy-stderr.log").write_text(process.stderr, encoding="utf-8")
    messages = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    ready = messages[0] if messages else {}
    inference = messages[1] if len(messages) > 1 else {}
    predictions = inference.get("predictions", [])
    passed = (process.returncode == 0 and ready.get("ok") and inference.get("ok") and bool(predictions)
              and Path(ready.get("model_path", "")).resolve() == (app / "models/yolov5n.pt").resolve())
    audit.require(passed, "Extracted legacy PT helper smoke failed: " + process.stderr[-1200:])
    labels = ready.get("labels", [])
    seen = {labels[p["class_id"]] for p in predictions if 0 <= p.get("class_id", -1) < len(labels)}
    audit.require({"bus", "person"}.issubset(seen), "Legacy PT smoke did not identify expected bus/person classes")
    for prediction in predictions:
        box = prediction.get("box", [])
        audit.require(len(box) == 4 and all(math.isfinite(value) for value in box)
                      and box[0] >= -0.1 and box[1] >= -0.1 and box[2] > 0 and box[3] > 0
                      and box[0] + box[2] <= 810.1 and box[1] + box[3] <= 1080.1,
                      "Legacy PT prediction is outside original image pixel coordinates")
    results["legacy_pt"] = {"success": bool(passed), "predictions": len(predictions),
                             "backend": ready.get("backend"), "inference_ms": inference.get("inference_ms")}
    audit.require(bool(list(state.rglob("preferences.ini"))), "Preferences were not saved in isolated user state")
    audit.require(bool(list(state.rglob("history.json"))), "History was not saved in isolated user state")
    audit.details["smoke"] = {"uid": os.geteuid(), "platform": "offscreen", "read_only_application": True,
                               "isolated_user_state": str(state), "results": results,
                               "limitation": "This is a non-root extracted-package test; it does not install the DEB or validate a live X11/Wayland session."}


def apt_simulation(deb, audit):
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    result = command(["apt-get", "-s", "-o", "Debug::NoLocking=1", "install", str(deb.resolve())],
                     env=environment, timeout=180, check=False)
    audit.require(result.returncode == 0, "APT installation simulation failed: " + result.stderr[-1800:])
    audit.require(not any(line.startswith("Remv ") for line in result.stdout.splitlines()),
                  "APT installation simulation would remove an existing system package")
    audit.details["apt_simulation"] = {"exit_code": result.returncode, "stdout": result.stdout,
                                       "stderr": result.stderr, "changes_system": False}


def audit_source_bundle(directory, deb, audit):
    checksum_file = directory / "SHA256SUMS"
    if not audit.require(checksum_file.is_file(), "Release source bundle has no SHA256SUMS"):
        return
    checksums = {}
    checksum_files = [checksum_file]
    source_checksums = directory / "SOURCE-SHA256SUMS"
    if source_checksums.exists():
        checksum_files.append(source_checksums)
    for checksum_file in checksum_files:
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
            if not audit.require(bool(match), f"Invalid {checksum_file.name} line: " + line):
                continue
            relative = PurePosixPath(match.group(2))
            if not audit.require(not relative.is_absolute() and ".." not in relative.parts,
                                 "Unsafe source bundle checksum path: " + str(relative)):
                continue
            path = directory / str(relative)
            if not audit.require(path.is_file(), "Release bundle file missing: " + str(relative)):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            audit.require(digest.hexdigest() == match.group(1).lower(), "Release bundle checksum mismatch: " + str(relative))
            checksums[str(relative)] = digest.hexdigest()
    audit.require(checksums.get(deb.name) == audit.details.get("deb_sha256"),
                  "Release bundle checksum file does not cover the verified DEB")
    archives = [directory / name for name in checksums if name.endswith((".tar.gz", ".tar.xz", ".tgz"))]
    expected = {"qtbase": "src/corelib/global/qglobal.cpp", "qtsvg": "src/svg/qsvgrenderer.cpp",
                "qtwayland": "src/client/qwaylanddisplay.cpp",
                "qtimageformats": "src/plugins/imageformats/webp/qwebphandler.cpp",
                "qtdeclarative": "src/qml/qml/qqmlengine.cpp",
                "qtwebchannel": "src/webchannel/qwebchannel.cpp",
                "qtpositioning": "src/positioning/qgeocoordinate.cpp",
                "qtwebengine": "src/core/web_engine_context.cpp"}
    sources = {}
    for module, required in expected.items():
        matching = [archive for archive in archives if module in archive.name and "6.8.3" in archive.name]
        if not audit.require(bool(matching), f"Corresponding Qt6.8.3 source archive missing: {module}"):
            continue
        archive = matching[0]
        with tarfile.open(archive, mode="r|*") as stream:
            found = any(member.name.endswith("/" + required) or member.name == required for member in stream)
        audit.require(found, f"Qt source archive does not contain the expected module source: {archive.name}")
        sources[module] = archive.name
    matching = [archive for archive in archives if (archive.name.startswith("vision-studio")
                or archive.name.startswith("app-source")) and ("source" in archive.name or "src" in archive.name)
                and "complete-source" not in archive.name]
    if audit.require(bool(matching), "Corresponding Vision Studio application source archive missing"):
        with tarfile.open(matching[0], mode="r|*") as stream:
            names = [member.name for member in stream if member.isfile()]
        for required in ("src/main.cpp", "src/core/visionengine.cpp", "scripts/pt_worker.py",
                         "scripts/netron_server.py", "CMakeLists.txt", "LICENSE"):
            audit.require(any(name == required or name.endswith("/" + required) for name in names),
                          "Application source archive lacks corresponding source/build file: " + required)
        sources["application"] = matching[0].name
    matching = [archive for archive in archives if archive.name == "netron-9.3.1-source.tar.gz"]
    if audit.require(bool(matching), "Corresponding Netron 9.3.1 upstream source archive missing"):
        with tarfile.open(matching[0], mode="r:gz") as stream:
            package = json.loads(stream.extractfile("netron-9.3.1/package.json").read())
            license_text = stream.extractfile("netron-9.3.1/LICENSE").read()
            dagre = stream.extractfile("netron-9.3.1/source/dagre.js").read()
        audit.require(package.get("version") == "9.3.1" and package.get("license") == "MIT"
                      and b"Copyright (c) Lutz Roeder" in license_text and b"dagrejs/graphlib" in dagre,
                      "Netron source tag/license/frontend provenance is invalid")
        sources["netron"] = matching[0].name
    audit.details["source_bundle"] = {"directory": str(directory.resolve()), "checked_files": checksums,
                                       "checksum_manifests": [path.name for path in checksum_files],
                                       "corresponding_source_archives": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deb", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run real inference from a read-only extracted application tree")
    parser.add_argument("--apt-simulate", action="store_true", help="Also run read-only apt-get -s install dependency/conflict resolution")
    parser.add_argument("--source-dir", type=Path, help="Verify accompanying SHA256SUMS and application/Qt6.8.3 source archives")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the new extraction/log directory for inspection")
    parser.add_argument("--report", type=Path, help="Write a JSON audit report")
    args = parser.parse_args()
    audit = Audit()
    workspace = Path(tempfile.mkdtemp(prefix="vision-studio-deb-qa-"))
    try:
        for tool in ("dpkg-deb", "dpkg-query", "dpkg", "readelf", "ldd"):
            if not shutil.which(tool):
                raise RuntimeError(f"Required audit tool missing: {tool}")
        if not args.deb.is_file():
            raise RuntimeError(f"DEB does not exist: {args.deb}")
        audit.details["deb"] = str(args.deb.resolve())
        digest = hashlib.sha256()
        with args.deb.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        audit.details["deb_sha256"] = digest.hexdigest()
        os_release = deb_fields(Path("/etc/os-release").read_text().replace("=", ": "))
        host_architecture = command(["dpkg", "--print-architecture"]).stdout.strip()
        audit.details["host"] = {"distribution": os_release.get("ID", "").strip('"'),
                                  "version": os_release.get("VERSION_ID", "").strip('"'),
                                  "architecture": host_architecture}
        audit.require(audit.details["host"] == {"distribution": "ubuntu", "version": "22.04", "architecture": "amd64"},
                      "This release verifier requires Ubuntu22.04 amd64 to audit exact distribution dependencies")
        control = audit_control(args.deb, workspace / "control", audit)
        archive_members(args.deb, audit)
        closure = audit_dependencies(control, audit)
        if audit.errors:
            raise RuntimeError("Static archive/control audit failed; unsafe or incomplete package was not extracted")
        root = workspace / "extracted"
        root.mkdir()
        command(["dpkg-deb", "-x", str(args.deb), str(root)], timeout=300)
        app = audit_extracted(root, control, closure, audit)
        contents = command(["dpkg-deb", "-c", str(args.deb)], timeout=300).stdout
        (workspace / "dpkg-contents.txt").write_text(contents, encoding="utf-8")
        if args.source_dir and not audit.errors:
            audit_source_bundle(args.source_dir, args.deb, audit)
        if args.apt_simulate and not audit.errors:
            apt_simulation(args.deb, audit)
        if args.smoke and not audit.errors:
            smoke_tests(app, workspace, audit)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired, KeyError,
            tarfile.TarError, configparser.Error) as error:
        audit.errors.append(str(error))
    finally:
        report = {"success": not audit.errors, "errors": audit.errors, "warnings": audit.warnings,
                  "details": audit.details}
        if args.keep_workdir:
            report["workdir"] = str(workspace)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"success": report["success"], "errors": audit.errors,
                          "warnings": audit.warnings, "report": str(args.report) if args.report else None,
                          "workdir": str(workspace) if args.keep_workdir else None}, ensure_ascii=False))
        if not args.keep_workdir:
            restore_temporary_permissions(workspace)
            shutil.rmtree(workspace)
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
