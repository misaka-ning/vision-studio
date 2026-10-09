#!/usr/bin/env python3
"""Build a root-owned Ubuntu 22.04 amd64 release package without sudo.

The build never edits its binary, Qt SDK, or Python environment inputs. Only
copies under a new staging directory are stripped or assigned new RUNPATHs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_metadata


PACKAGE = "vision-studio"
DEFAULT_ROOT = "/opt/VisionStudio"
WEBENGINE_RESOURCES = ("qtwebengine_resources.pak", "qtwebengine_devtools_resources.pak",
                       "qtwebengine_resources_100p.pak", "qtwebengine_resources_200p.pak",
                       "icudtl.dat", "v8_context_snapshot.bin")
PLUGIN_FILES = {
    "platforms": ("libqxcb.so", "libqoffscreen.so", "libqminimal.so",
                  "libqwayland-generic.so", "libqwayland-egl.so"),
    "imageformats": ("libqjpeg.so", "libqgif.so", "libqico.so", "libqsvg.so"),
    "iconengines": ("libqsvgicon.so",),
    "platforminputcontexts": ("libcomposeplatforminputcontextplugin.so", "libibusplatforminputcontextplugin.so"),
    "platformthemes": ("libqgtk3.so", "libqxdgdesktopportal.so"),
    "xcbglintegrations": ("libqxcb-glx-integration.so", "libqxcb-egl-integration.so"),
    "wayland-decoration-client": ("libbradient.so", "libadwaita.so"),
    "wayland-graphics-integration-client": ("libqt-plugin-wayland-egl.so",),
    "wayland-shell-integration": ("libxdg-shell.so",),
}


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, capture: bool = True) -> str:
    process_environment = dict(os.environ if env is None else env)
    process_environment["LC_ALL"] = "C"
    result = subprocess.run(command, cwd=cwd, env=process_environment, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail = ((result.stdout or "") + (result.stderr or ""))[-6000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(command[:5])}\n{detail}")
    return result.stdout or ""


def write(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755 if executable else 0o644)


def is_elf(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored_files(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {"__pycache__", ".git", ".cache", ".pytest_cache"}
            or name.endswith((".pyc", ".pyo"))}


def copy_tree(source: Path, target: Path, *, runtime: bool = False) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = ignored_files(directory, names)
        if runtime:
            skipped.update(name for name in names if name == "pip" or name.startswith("pip-")
                           and name.endswith(".dist-info"))
        return skipped
    shutil.copytree(source, target, symlinks=True, ignore=ignore)


def check_inputs(arguments: argparse.Namespace) -> None:
    for tool in ("dpkg-deb", "dpkg-shlibdeps", "patchelf", "strip", "readelf", "ldd"):
        if not shutil.which(tool):
            raise RuntimeError(f"Missing build tool: {tool}")
    if run(["dpkg", "--print-architecture"]).strip() != "amd64":
        raise RuntimeError("This release targets Ubuntu 22.04 amd64 only.")
    release = dict(re.findall(r'^([A-Z_]+)="?([^"\n]+)"?$',
                             Path("/etc/os-release").read_text(), flags=re.MULTILINE))
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "22.04":
        raise RuntimeError("Build on Ubuntu 22.04 to derive valid Jammy library dependencies.")
    if not is_elf(arguments.binary):
        raise RuntimeError(f"Release binary not found or not ELF: {arguments.binary}")
    if "Advanced Micro Devices X86-64" not in run(["readelf", "-h", str(arguments.binary)]):
        raise RuntimeError("Release binary is not x86_64.")
    if not (arguments.qt_prefix / "lib/libQt6Core.so.6").is_file():
        raise RuntimeError("--qt-prefix must point to the Qt 6.8.3 gcc_64 runtime/SDK directory.")
    for path in arguments.extra_qt_plugins:
        if not path.is_dir():
            raise RuntimeError(f"Extra Qt plugin directory not found: {path}")
    qt_version_file = arguments.qt_prefix / "lib/cmake/Qt6Core/Qt6CoreConfigVersionImpl.cmake"
    if qt_version_file.exists() and '"6.8.3"' not in qt_version_file.read_text():
        raise RuntimeError("The release bundles Qt 6.8.3; the selected SDK differs.")
    site_packages = arguments.runtime / "lib/python3.10/site-packages"
    if not site_packages.is_dir():
        raise RuntimeError("--runtime must contain Python 3.10 site-packages.")
    metadata = json.loads(run([str(arguments.runtime / "bin/python"), "-c",
                              "import json,sys,importlib.metadata as m;"
                              "print(json.dumps({'python':list(sys.version_info[:2]),"
                              "'torch':m.version('torch'),'ultralytics':m.version('ultralytics'),"
                              "'distributions':{d.metadata['Name']:d.version for d in m.distributions()"
                              " if d.metadata['Name'].lower()!='pip'}}))"]))
    if metadata["python"] != [3, 10] or "+cpu" not in metadata["torch"]:
        raise RuntimeError("The package requires the tested Python 3.10 CPU PyTorch runtime.")
    if "opencv-python-headless" not in metadata["distributions"] or "opencv-python" in metadata["distributions"]:
        raise RuntimeError("Use the release headless OpenCV runtime; the GUI OpenCV wheel adds an unwanted Qt5 runtime.")
    if metadata["distributions"].get("netron") != "9.3.1":
        raise RuntimeError("The model structure viewer requires the pinned Netron 9.3.1 distribution.")
    if metadata["distributions"].get("onnx") != "1.17.0" or metadata["distributions"].get("protobuf") != "6.33.0":
        raise RuntimeError("The model converter requires the tested ONNX 1.17.0 / protobuf 6.33.0 runtime.")
    if not (arguments.qt_prefix / "libexec/QtWebEngineProcess").is_file():
        raise RuntimeError("The Qt 6.8.3 SDK must include WebEngine and QtWebEngineProcess.")
    arguments.runtime_metadata = metadata
    if not arguments.copyright.is_file():
        raise RuntimeError("Missing final copyright file; supply --copyright or finish packaging/copyright.")
    root = PurePosixPath(arguments.install_root)
    if not root.is_absolute() or str(root) != arguments.install_root or ".." in root.parts \
            or str(root) in {"/", "/opt", "/usr", "/usr/bin", "/bin", "/etc", "/lib", "/var"} \
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(root)):
        raise RuntimeError("--install-root must be a dedicated absolute application directory.")
    if arguments.stage.exists():
        raise RuntimeError(f"Refusing to overwrite staging directory: {arguments.stage}")
    if arguments.output.exists():
        raise RuntimeError(f"Refusing to overwrite release file: {arguments.output}")
    try:
        release_metadata.application_version_from_debian(arguments.version)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def bundle_qt(arguments: argparse.Namespace, app: Path) -> dict[str, str]:
    library_root = (arguments.qt_prefix / "lib").resolve()
    library_target = app / "lib/qt"
    library_target.mkdir(parents=True)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(library_root)
    helper = arguments.qt_prefix / "libexec/QtWebEngineProcess"
    (app / "libexec").mkdir(parents=True)
    shutil.copy2(helper, app / "libexec/QtWebEngineProcess")
    seeds = [arguments.binary, helper]
    copied: dict[str, str] = {}
    for category, names in PLUGIN_FILES.items():
        for name in names:
            source = arguments.qt_prefix / "plugins" / category / name
            if not source.is_file():
                raise RuntimeError(f"Missing Qt plugin: {source}")
            target = app / "plugins" / category / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            seeds.append(source)
    for directory in arguments.extra_qt_plugins:
        extra_files = sorted(path for path in directory.rglob("*.so") if path.is_file())
        if not extra_files:
            raise RuntimeError(f"Extra Qt plugin directory contains no plugins: {directory}")
        for source in extra_files:
            relative = source.relative_to(directory)
            if len(relative.parts) < 2 or not is_elf(source):
                raise RuntimeError(f"Expected a categorized Qt ELF plugin: {source}")
            target = app / "plugins" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != sha256(source):
                raise RuntimeError(f"Extra Qt plugin conflicts with the selected SDK: {relative}")
            if not target.exists():
                shutil.copy2(source, target)
            seeds.append(source)
    for source in seeds:
        dependencies = run(["ldd", str(source)], env=environment)
        if "not found" in dependencies:
            raise RuntimeError(f"Unresolved dependency for {source.name}:\n{dependencies}")
        for name, path in re.findall(r"^\s*(\S+) => (\S+) \(", dependencies, flags=re.MULTILINE):
            candidate = Path(path).resolve()
            if not candidate.is_relative_to(library_root):
                continue
            target = library_target / candidate.name
            if not target.exists():
                shutil.copy2(candidate, target)
            if name != candidate.name:
                link = library_target / name
                if not link.exists():
                    link.symlink_to(candidate.name)
            copied[name] = candidate.name
    # QtSvg can otherwise be loaded solely through an image/icon plugin.
    if not all(name in copied for name in
               ("libQt6Core.so.6", "libQt6Svg.so.6", "libQt6WebEngineCore.so.6",
                "libQt6WebEngineWidgets.so.6", "libQt6WebChannel.so.6")):
        raise RuntimeError("Incomplete Qt dependency closure.")
    resources = app / "resources"
    resources.mkdir()
    for name in WEBENGINE_RESOURCES:
        original = arguments.qt_prefix / "resources" / name
        if not original.is_file():
            raise RuntimeError("Missing Qt WebEngine deployment resource: " + name)
        shutil.copy2(original, resources / name)
    locales = arguments.qt_prefix / "translations/qtwebengine_locales"
    if not (locales / "en-US.pak").is_file():
        raise RuntimeError("Missing Qt WebEngine Chromium locale resources.")
    copy_tree(locales, app / "translations/qtwebengine_locales")
    for translation in (arguments.qt_prefix / "translations").glob("qtwebengine_*.qm"):
        shutil.copy2(translation, app / "translations" / translation.name)
    write(app / "build/bin/qt.conf", "[Paths]\nPrefix=../..\nLibraries=lib/qt\nPlugins=plugins\n"
          "LibraryExecutables=libexec\nData=.\nTranslations=translations\n")
    write(app / "libexec/qt.conf", "[Paths]\nPrefix=..\nLibraries=lib/qt\nPlugins=plugins\n"
          "LibraryExecutables=libexec\nData=.\nTranslations=translations\n")
    return copied


def bundle_python(arguments: argparse.Namespace, app: Path) -> None:
    target = app / "runtime"
    copy_tree(arguments.runtime / "lib", target / "lib", runtime=True)
    (target / "bin").mkdir()
    # Console entry points and activation scripts are unnecessary for inference
    # and contain the original venv path. Use an explicitly versioned OS Python.
    (target / "bin/python").symlink_to("python3")
    (target / "bin/python3").symlink_to("/usr/bin/python3.10")
    (target / "bin/python3.10").symlink_to("python3")
    (target / "lib64").symlink_to("lib")
    write(target / "pyvenv.cfg", "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.10.12\n")


def required_versions(path: Path) -> dict[str, set[str]]:
    text = run(["readelf", "--version-info", str(path)])
    if "Version needs section" not in text:
        return {}
    text = text.split("Version needs section", 1)[1]
    result: dict[str, set[str]] = {}
    current = None
    for line in text.splitlines():
        file_match = re.search(r"File:\s+(\S+)", line)
        if file_match:
            current = file_match[1]
            result.setdefault(current, set())
        name_match = re.search(r"Name:\s+(\S+)", line)
        if current and name_match:
            result[current].add(name_match[1])
    return result


def defined_versions(path: Path) -> set[str]:
    text = run(["readelf", "--version-info", str(path)])
    if "Version definition section" not in text:
        return set()
    text = text.split("Version definition section", 1)[1].split("Version needs section", 1)[0]
    return set(re.findall(r"Name:\s+(\S+)", text))


def dynamic_symbols(path: Path, undefined: bool = False) -> set[str]:
    symbols = set()
    for line in run(["readelf", "--dyn-syms", "--wide", str(path)]).splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].rstrip(":").isdigit() and (parts[6] == "UND") == undefined:
            symbols.add(parts[7].split("@")[0])
    return symbols


def use_system_opencv_blas(app: Path) -> dict[str, str]:
    """Replace the headless wheel's old LP64 BLAS with Jammy BLAS/LAPACK.

    OpenCV's wheel BLAS can depend on the obsolete GFORTRAN_1.0 ABI, which
    cannot be redirected to libgfortran5. Its public LP64 BLAS/LAPACK entry
    points are provided by Jammy; NumPy's separate ILP64 BLAS stays bundled.
    """
    cv2 = app / "runtime/lib/python3.10/site-packages/cv2/cv2.abi3.so"
    old_names = [name for name in run(["patchelf", "--print-needed", str(cv2)]).splitlines()
                 if name.startswith("libopenblas")]
    if not old_names:
        return {}
    if len(old_names) != 1:
        raise RuntimeError("OpenCV has an unexpected BLAS dependency layout.")
    old = old_names[0]
    old_library = cv2.parent.parent / "opencv_python_headless.libs" / old
    if not old_library.is_file() or old in required_versions(cv2):
        raise RuntimeError("OpenCV's BLAS dependency is missing or has unsupported versioned entry points.")
    # Ubuntu 22.04 provides matching 32-bit-integer (LP64) standard interfaces.
    providers = {"libblas.so.3": Path("/lib/x86_64-linux-gnu/libblas.so.3"),
                 "liblapack.so.3": Path("/lib/x86_64-linux-gnu/liblapack.so.3")}
    provided = set()
    for path in providers.values():
        if not path.is_file():
            raise RuntimeError(f"Missing system BLAS/LAPACK library: {path.name}")
        provided.update(dynamic_symbols(path))
    used = dynamic_symbols(cv2, undefined=True) & dynamic_symbols(old_library)
    missing = used - provided
    if missing:
        raise RuntimeError("OpenCV requires BLAS symbols absent from Jammy BLAS/LAPACK: " + ", ".join(sorted(missing)))
    if not {"cblas_sgemm", "cblas_dgemm"}.issubset(used):
        raise RuntimeError("Unexpected OpenCV BLAS ABI; expected standard LP64 GEMM interfaces.")
    run(["patchelf", "--replace-needed", old, "libblas.so.3", str(cv2)])
    run(["patchelf", "--add-needed", "liblapack.so.3", str(cv2)])
    # No other extension may rely on the removed OpenCV-specific OpenBLAS.
    for path in (app / "runtime").rglob("*"):
        if is_elf(path) and path != old_library and old in run(["patchelf", "--print-needed", str(path)]).splitlines():
            raise RuntimeError(f"Another runtime extension still depends on the obsolete OpenCV BLAS: {path.name}")
    old_library.unlink()
    log("Replaced OpenCV's obsolete bundled BLAS with ABI-checked system BLAS/LAPACK.")
    return {old: "libblas.so.3 + liblapack.so.3"}


def use_system_gnu_libraries(app: Path) -> dict[str, str]:
    """Bind wheel GNU runtimes to Ubuntu libraries after checking their ABI.

    These libraries are already provided by Jammy. Using the system copies keeps
    the package's corresponding-source scope focused on its own bundled code.
    Only staged consumers have their DT_NEEDED entries changed; wheel libraries
    are never stripped or otherwise rewritten by binutils.
    """
    system_sonames = {"libgfortran": "libgfortran.so.5", "libquadmath": "libquadmath.so.0",
                      "libgomp": "libgomp.so.1"}
    native_files = sorted(path for path in (app / "runtime").rglob("*") if is_elf(path))
    bundled: dict[str, str] = {}
    removed = []
    for path in native_files:
        for prefix, soname in system_sonames.items():
            if path.name.startswith(prefix) and ".so" in path.name:
                old_soname = run(["patchelf", "--print-soname", str(path)]).strip() or path.name
                # Do not pretend libgfortran3 and libgfortran5 have the same ABI.
                if prefix == "libgfortran" and ".so.3" in old_soname:
                    soname = "libgfortran.so.3"
                bundled[old_soname] = soname
                bundled[path.name] = soname
                removed.append(path)
                break
    if not removed:
        return {}
    available = run(["ldconfig", "-p"])
    consumer_files = [path for path in native_files if path not in removed]
    needed_by_file = {path: run(["patchelf", "--print-needed", str(path)]).splitlines() for path in consumer_files}
    used_old_names = {name for names in needed_by_file.values() for name in names if name in bundled}
    system_files = {}
    provided = {}
    for soname in {bundled[name] for name in used_old_names}:
        candidates = re.findall(r"^\s*" + re.escape(soname)
                                + r"\s+\([^\n]*x86-64[^\n]*\) => (\S+)", available,
                                flags=re.MULTILINE)
        if not candidates:
            raise RuntimeError(f"Missing Ubuntu system library: {soname}")
        system_files[soname] = Path(candidates[0])
        provided[soname] = defined_versions(system_files[soname])
    rewritten = 0
    for path in consumer_files:
        needed = needed_by_file[path]
        substitutions = {old: bundled[old] for old in needed if old in bundled}
        if not substitutions:
            continue
        versions = required_versions(path)
        for old, new in substitutions.items():
            missing = versions.get(old, set()) - provided[new]
            if missing:
                raise RuntimeError(f"{path.name} requires {old} ABI versions absent from {new}: "
                                   + ", ".join(sorted(missing)))
            if old != new:
                run(["patchelf", "--replace-needed", old, new, str(path)])
            # Confirm loader metadata names changed along with DT_NEEDED.
            after = required_versions(path)
            if old != new and old in after:
                raise RuntimeError(f"ELF version dependency was not updated for {path.name}: {old}")
        rewritten += 1
    for path in removed:
        path.unlink()
    for path in (app / "runtime").rglob("*"):
        if path.is_symlink() and any(path.name.startswith(prefix) for prefix in system_sonames):
            path.unlink()
    log(f"Bound {rewritten} wheel native libraries to system GNU runtimes; removed {len(removed)} bundled copies.")
    return {old: bundled[old] for old in sorted(used_old_names)}


def patch_and_strip(arguments: argparse.Namespace, app: Path) -> list[Path]:
    native_files = sorted(path for path in app.rglob("*") if is_elf(path))
    log(f"Preparing {len(native_files)} staged ELF files (inputs stay unchanged)…")
    for path in native_files:
        relative = path.relative_to(app)
        if relative.parts[0] == "runtime" and path.name.startswith(("libQt5", "libgfortran", "libquadmath", "libgomp")):
            raise RuntimeError(f"Unexpected bundled Qt5/GNU runtime remains: {relative}")
        if relative == Path("build/bin/vision-studio"):
            run(["patchelf", "--set-rpath", "$ORIGIN/../../lib/qt", str(path)])
        elif relative.parts[:2] == ("lib", "qt"):
            run(["patchelf", "--set-rpath", "$ORIGIN", str(path)])
        elif relative.parts[0] == "plugins":
            run(["patchelf", "--set-rpath", "$ORIGIN/../../lib/qt", str(path)])
        elif relative == Path("libexec/QtWebEngineProcess"):
            run(["patchelf", "--set-rpath", "$ORIGIN/../lib/qt", str(path)])
        else:
            old_rpath = run(["patchelf", "--print-rpath", str(path)]).strip()
            if "/home/" in old_rpath:
                raise RuntimeError(f"Python wheel contains a non-relocatable RUNPATH: {relative}")
        # Auditwheel can add load segments with unusual alignment. GNU strip on
        # those already prepared wheels can corrupt their segment alignment.
        # Preserve wheel native binaries byte-for-byte; strip only our app/Qt.
        if not arguments.no_strip and relative.parts[0] != "runtime":
            run(["strip", "--strip-unneeded", str(path)])
        rpath = run(["patchelf", "--print-rpath", str(path)]).strip()
        if "/home/" in rpath:
            raise RuntimeError(f"Home directory leaked into RUNPATH: {relative}")
    binary_bytes = (app / "build/bin/vision-studio").read_bytes()
    for forbidden in (str(arguments.source_root).encode(), str(arguments.qt_prefix).encode(),
                      str(arguments.runtime).encode()):
        if forbidden.startswith(b"/home/") and forbidden in binary_bytes:
            raise RuntimeError("Release binary embeds a build-machine home path; rebuild with the release CMake paths.")
    return native_files


def dependencies(arguments: argparse.Namespace, app: Path, native_files: list[Path]) -> str:
    log("Deriving Ubuntu 22.04 native library dependencies…")
    with tempfile.TemporaryDirectory(prefix="vision-deb-deps-") as temporary:
        work = Path(temporary)
        write(work / "debian/control", "Source: vision-studio\nSection: graphics\nPriority: optional\n"
              "Maintainer: Vision Studio maintainers <noreply@localhost>\nStandards-Version: 4.6.0\n\n"
              "Package: vision-studio\nArchitecture: amd64\nDescription: Local vision model workspace\n")
        private_directories = sorted({path.parent for path in native_files})
        command = ["dpkg-shlibdeps", "-O", "--ignore-missing-info", "--warnings=0", f"-S{arguments.stage}"]
        command.extend(f"-l{directory}" for directory in private_directories)
        command.extend(f"-e{path}" for path in native_files)
        result = run(command, cwd=work)
        matches = re.findall(r"^shlibs:Depends=(.*)$", result, flags=re.MULTILINE)
        if len(matches) != 1 or not matches[0]:
            raise RuntimeError("dpkg-shlibdeps did not produce system dependencies.")
        native = matches[0]
        if "libopencv-core4.5d" not in native or "libopencv-dnn4.5d" not in native:
            raise RuntimeError("Native dependency analysis did not include OpenCV 4.5.")
        return native + ", python3 (>= 3.10), python3 (<< 3.11), python3.10, python3.10-venv, fontconfig, fonts-noto-cjk"


def normalize_permissions(stage: Path) -> None:
    for path in stage.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.stat().st_mode & 0o111:
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def stage_payload(arguments: argparse.Namespace) -> tuple[Path, dict[str, str]]:
    app = arguments.stage / arguments.install_root.lstrip("/")
    (app / "build/bin").mkdir(parents=True)
    shutil.copy2(arguments.binary, app / "build/bin/vision-studio")
    for name in ("assets", "models", "vendor"):
        copy_tree(arguments.source_root / name, app / name)
    # Omit training export inputs; keep the tested runnable models and provenance.
    if (app / "models/source").exists():
        shutil.rmtree(app / "models/source")
    (app / "scripts").mkdir()
    shutil.copy2(arguments.source_root / "scripts/pt_worker.py", app / "scripts/pt_worker.py")
    shutil.copy2(arguments.source_root / "scripts/netron_server.py", app / "scripts/netron_server.py")
    shutil.copy2(arguments.source_root / "scripts/model_convert.py", app / "scripts/model_convert.py")
    for name in ("gpu_setup.py", "gpu_probe.py", "setup_gpu.sh"):
        shutil.copy2(arguments.source_root / "scripts" / name, app / "scripts" / name)
    for name in ("requirements-gpu.txt", "requirements-gpu.lock.txt"):
        shutil.copy2(arguments.source_root / name, app / name)
    packages = arguments.runtime_metadata["distributions"]
    locked = "\n".join(f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda item: item[0].lower()))
    write(app / "requirements-pt.lock.txt", "# Actual distributions bundled in this release (pip excluded).\n" + locked + "\n")
    # Reproduce the headless release with --no-deps for Ultralytics, whose default
    # dependency name would reinstall the GUI OpenCV wheel beside headless cv2.
    write(app / "requirements-pt.txt", "# Release runtime: see lock for every distribution.\n"
          "# Install this complete lock with pip --no-deps to retain headless OpenCV.\n"
          "--extra-index-url https://download.pytorch.org/whl/cpu\n"
          "-r requirements-pt.lock.txt\n")
    runtime_info = json.loads((arguments.source_root / "runtime-info.json").read_text())
    runtime_info = {key: value for key, value in runtime_info.items() if not key.startswith("verified_")}
    runtime_info.update({"interpreter": arguments.install_root + "/runtime/bin/python",
                         "base_interpreter": "/usr/bin/python3.10",
                         "runtime_directory": arguments.install_root + "/runtime",
                         "opencv_python": ".".join(packages["opencv-python-headless"].split(".")[:3]),
                         "opencv_python_package": packages["opencv-python-headless"],
                         "opencv_python_distribution": "opencv-python-headless",
                         "resolved_distributions": packages})
    write(app / "runtime-info.json", json.dumps(runtime_info, ensure_ascii=False, indent=2) + "\n")
    docs = app / "docs"
    docs.mkdir()
    for source in (arguments.source_root / "docs").glob("*.md"):
        # Development verification logs describe the original home-directory
        # delivery. Release QA is published separately with the release bundle.
        if source.name != "验证报告.md":
            shutil.copy2(source, docs / source.name)
    if (arguments.source_root / "docs/release").is_dir():
        copy_tree(arguments.source_root / "docs/release", docs / "release")
        # Raw build/test logs can expose build-machine paths. They remain in the
        # separate source/verification release bundle, not in the installed app.
        for path in (docs / "release").rglob("*"):
            if path.is_file() and (path.suffix == ".log" or ".log.tmp" in path.name):
                path.unlink()
    shutil.copy2(arguments.source_root / "packaging/run-installed.sh", app / "run-installed.sh")
    (app / "run-installed.sh").chmod(0o755)
    (app / "run.sh").symlink_to("run-installed.sh")
    qt_libraries = bundle_qt(arguments, app)
    bundle_python(arguments, app)
    write(arguments.stage / "usr/bin/vision-studio", "#!/bin/sh\nexec "
          + shlex.quote(arguments.install_root + "/run-installed.sh") + ' "$@"\n', executable=True)
    write(arguments.stage / "usr/share/applications/vision-studio.desktop",
          "[Desktop Entry]\nType=Application\nName=Vision Studio\n"
          "Name[zh_CN]=Vision Studio 视觉模型工作台\n"
          "Comment=Local YOLO model inference workspace\n"
          "Comment[zh_CN]=本地 YOLO 目标检测与视觉模型工具\n"
          "Exec=vision-studio\nIcon=vision-studio\nTerminal=false\n"
          "Categories=Graphics;Science;\nStartupNotify=true\n")
    icon = arguments.stage / "usr/share/icons/hicolor/scalable/apps/vision-studio.svg"
    icon.parent.mkdir(parents=True)
    shutil.copy2(arguments.source_root / "assets/app-icon.svg", icon)
    package_docs = arguments.stage / "usr/share/doc/vision-studio"
    package_docs.mkdir(parents=True)
    shutil.copy2(arguments.copyright, package_docs / "copyright")
    if (arguments.source_root / "LICENSE").is_file():
        shutil.copy2(arguments.source_root / "LICENSE", package_docs / "LICENSE")
    shutil.copy2(arguments.source_root / "packaging/README.md", package_docs / "README.Debian")
    if (arguments.source_root / "packaging/licenses").is_dir():
        copy_tree(arguments.source_root / "packaging/licenses", package_docs / "licenses")
    return app, qt_libraries


def build(arguments: argparse.Namespace) -> None:
    check_inputs(arguments)
    log(f"Staging {PACKAGE} {arguments.version} at {arguments.stage}…")
    app, qt_libraries = stage_payload(arguments)
    system_blas_libraries = use_system_opencv_blas(app)
    system_gnu_libraries = use_system_gnu_libraries(app)
    native_files = patch_and_strip(arguments, app)
    depends = dependencies(arguments, app, native_files)
    installed_kib = (sum(path.stat().st_size for path in arguments.stage.rglob("*")
                         if path.is_file() and not path.is_symlink()) + 1023) // 1024
    control = (f"Package: {PACKAGE}\nVersion: {arguments.version}\nArchitecture: amd64\n"
               "Section: graphics\nPriority: optional\n"
               f"Maintainer: {arguments.maintainer}\nInstalled-Size: {installed_kib}\n"
               f"Depends: {depends}\n"
               "Description: Professional local YOLO vision model workspace\n"
               " C++ and Qt 6.8.3 desktop application with bundled CPU PyTorch runtime.\n"
               " Includes a local Netron model structure viewer embedded in Qt WebEngine.\n"
               " Optional NVIDIA CUDA support is prepared separately per user on request.\n"
               " Supports YOLO .pt and ONNX image, video, camera inference and result exports.\n"
               " Grayscale input adapts to one or three model channels; side-by-side streams\n"
               " can preview and infer the left or right eye independently.\n"
               " Built for Ubuntu 22.04 LTS amd64. Runtime assets reside in /opt/VisionStudio\n"
               " by default; results and settings are stored separately for each user.\n")
    write(arguments.stage / "DEBIAN/control", control)
    provenance = {"package": PACKAGE, "version": arguments.version, "architecture": "amd64",
                  "baseline": "Ubuntu 22.04 LTS", "install_root": arguments.install_root,
                  "qt_version": "6.8.3", "qt_libraries": qt_libraries,
                  "system_gnu_libraries": system_gnu_libraries,
                  "system_blas_libraries": system_blas_libraries,
                  "python_runtime": arguments.runtime_metadata, "depends": depends,
                  "binary_sha256": sha256(app / "build/bin/vision-studio")}
    write(app / "release-info.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    normalize_permissions(arguments.stage)
    # dpkg itself assigns root ownership in both archives; no chown or sudo.
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("DPKG_DEB_THREADS_MAX", "4")
    log(f"Compressing the package with {arguments.compression}…")
    run(["dpkg-deb", "--root-owner-group", "--uniform-compression",
         f"-Z{arguments.compression}", f"-z{arguments.compression_level}",
         "--build", str(arguments.stage), str(arguments.output)], env=environment, capture=False)
    checksum = sha256(arguments.output)
    write(arguments.output.with_suffix(arguments.output.suffix + ".sha256"),
          f"{checksum}  {arguments.output.name}\n")
    write(arguments.output.with_suffix(arguments.output.suffix + ".manifest.json"),
          json.dumps({**provenance, "sha256": checksum, "bytes": arguments.output.stat().st_size},
                     ensure_ascii=False, indent=2) + "\n")
    log(f"Built {arguments.output} ({arguments.output.stat().st_size / 1024**2:.1f} MiB)\n"
        f"SHA256 {checksum}\nStaged application: {app}")


def main() -> int:
    source_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=source_root)
    parser.add_argument("--binary", type=Path, default=source_root / "build/release/bin/vision-studio")
    parser.add_argument("--qt-prefix", type=Path, required=True)
    parser.add_argument("--extra-qt-plugins", type=Path, action="append", default=[],
                        help="Additional Qt 6.8.3 plugins directory, e.g. a supplemental QtImageFormats install.")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--install-root", default=DEFAULT_ROOT)
    parser.add_argument("--version", help="Debian Control version derived from CMake by default; beta uses ~beta.N")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--copyright", type=Path)
    parser.add_argument("--maintainer", default="misaka_ning <1468549029@qq.com>")
    parser.add_argument("--compression", choices=("xz", "zstd"), default="xz")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--no-strip", action="store_true", help="Retain staged symbols for diagnostics.")
    arguments = parser.parse_args()
    arguments.source_root = arguments.source_root.resolve()
    try:
        source_version = release_metadata.cmake_application_version(
            (arguments.source_root / "CMakeLists.txt").read_text(encoding="utf-8"))
        arguments.version = arguments.version or release_metadata.debian_version(source_version)
        application_version = release_metadata.application_version_from_debian(arguments.version)
        if application_version != source_version:
            raise ValueError("Debian package and source application versions differ")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for name in ("binary", "qt_prefix", "runtime"):
        setattr(arguments, name, getattr(arguments, name).resolve())
    arguments.extra_qt_plugins = [path.resolve() for path in arguments.extra_qt_plugins]
    arguments.copyright = (arguments.copyright or arguments.source_root / "packaging/copyright").resolve()
    arguments.stage = (arguments.stage or arguments.source_root / "output/deb-stage" / arguments.version).resolve()
    arguments.output = (arguments.output or arguments.source_root / "output/releases" / application_version
                        / release_metadata.deb_filename(arguments.version)).resolve()
    if not 0 <= arguments.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    if "\n" in arguments.maintainer or "\r" in arguments.maintainer:
        parser.error("--maintainer must be a single line")
    try:
        build(arguments)
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
