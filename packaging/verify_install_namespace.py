#!/usr/bin/env python3
"""Install, launch and purge a DEB inside private bubblewrap namespaces.

The host filesystem is read-only. Only a newly-created temporary directory,
private dpkg database, /opt, /usr/bin integration and desktop/icon/doc targets
are writable. Namespace UID0 maps to this invoking unprivileged user; inference
uses a separate namespace with the user's original nonzero UID. No host sudo,
system package installation, maintainer scripts or trigger execution is used.
Requires verify_deb.py beside this script and a host permitting bubblewrap user
namespaces. It deliberately fails instead of falling back to a host install.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

from verify_deb import Audit, archive_members, deb_fields


MOUNT = "/tmp/vision-install-qa"


def execute(arguments, *, timeout=120):
    # dpkg checks root administrative helpers even for packages without scripts.
    environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    return subprocess.run(arguments, env=environment, timeout=timeout, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_database(target):
    # copy2 preserves mode/mtime, but not ownership: every private copy belongs
    # to this user, which becomes UID0 in the installation namespace.
    def ignored(directory, names):
        return {name for name in names if name in {"lock", "lock-frontend", "Lock"}
                or (Path(directory).name == "info" and name.startswith("vision-studio."))}

    shutil.copytree("/var/lib/dpkg", target, symlinks=True, ignore=ignored)
    status = target / "status"
    content = status.read_text(encoding="utf-8")
    paragraphs = [paragraph for paragraph in content.split("\n\n")
                  if deb_fields(paragraph).get("Package") != "vision-studio"]
    status.write_text("\n\n".join(paragraphs).rstrip() + "\n", encoding="utf-8")


class Namespace:
    def __init__(self, workspace):
        self.workspace = workspace
        candidates = sorted(Path("/usr/bin").iterdir())
        # Some development machines contain dangling tool aliases. They are
        # irrelevant to installation and cannot be used as bind sources.
        self.original_bins = [path for path in candidates if path.exists()]
        self.skipped_broken_bins = [str(path) for path in candidates if not path.exists()]
        self.bindings = {
            "opt": "/opt",
            "dpkg": "/var/lib/dpkg",
            "usr-bin": "/usr/bin",
            "applications": "/usr/share/applications",
            "icons": "/usr/share/icons/hicolor/scalable/apps",
            "documentation": "/usr/share/doc",
        }

    def command(self, arguments, *, root=False, readonly=False):
        uid, gid = (0, 0) if root else (os.getuid(), os.getgid())
        command = ["bwrap", "--ro-bind", "/", "/", "--unshare-user", "--uid", str(uid),
                   "--gid", str(gid), "--unshare-pid", "--unshare-net", "--die-with-parent", "--new-session",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--bind", str(self.workspace), MOUNT]
        for local, destination in self.bindings.items():
            command.extend(["--ro-bind" if readonly else "--bind",
                            str(self.workspace / local), destination])
        # Keep host tools available without copying their binaries. Mountpoint
        # placeholders persist privately; a newly installed wrapper is kept.
        for source in self.original_bins:
            if source.name != "vision-studio":
                command.extend(["--ro-bind", str(source), "/usr/bin/" + source.name])
        for key, value in {
            "HOME": MOUNT + "/user/home", "TMPDIR": MOUNT + "/tmp",
            "XDG_CONFIG_HOME": MOUNT + "/user/config", "XDG_DATA_HOME": MOUNT + "/user/data",
            "XDG_CACHE_HOME": MOUNT + "/user/cache", "XDG_STATE_HOME": MOUNT + "/user/state",
            "XDG_RUNTIME_DIR": MOUNT + "/user/runtime", "YOLO_CONFIG_DIR": MOUNT + "/user/ultralytics",
            "YOLOV5_CONFIG_DIR": MOUNT + "/user/yolov5", "MPLCONFIGDIR": MOUNT + "/user/matplotlib",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "QT_QPA_PLATFORM": "offscreen",
            # Headless software rendering only; preserve the browser sandbox.
            "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu", "QT_QUICK_BACKEND": "software",
        }.items():
            command.extend(["--setenv", key, value])
        command.extend(["--chdir", MOUNT, "--", *arguments])
        return command

    def run(self, arguments, name, *, root=False, readonly=False, timeout=120):
        started = time.monotonic()
        result = execute(self.command(arguments, root=root, readonly=readonly), timeout=timeout)
        (self.workspace / "logs" / (name + "-stdout.log")).write_text(result.stdout, encoding="utf-8")
        (self.workspace / "logs" / (name + "-stderr.log")).write_text(result.stderr, encoding="utf-8")
        return result, round(time.monotonic() - started, 2)


def prepare(workspace):
    for name in ("opt", "usr-bin", "applications", "icons", "documentation", "logs", "tmp"):
        (workspace / name).mkdir(mode=0o700)
    # /usr/bin and installed integration targets must be traversable by a
    # separate non-root namespace, regardless of package-created subdirs.
    for name in ("opt", "usr-bin", "applications", "icons", "documentation"):
        (workspace / name).chmod(0o755)
    for name in ("home", "config", "data", "cache", "state", "runtime", "ultralytics", "yolov5", "matplotlib"):
        (workspace / "user" / name).mkdir(parents=True, mode=0o700)
    copy_database(workspace / "dpkg")


def uninstall_checks(workspace, audit, user_snapshot):
    for relative in ("opt/VisionStudio", "usr-bin/vision-studio", "applications/vision-studio.desktop",
                     "icons/vision-studio.svg", "documentation/vision-studio"):
        path = workspace / relative
        audit.require(not path.exists() and not path.is_symlink(), "Payload remains after private dpkg purge: " + relative)
    current = {str(path.relative_to(workspace / "user")): sha256(path)
               for path in (workspace / "user").rglob("*") if path.is_file() and not path.is_symlink()}
    audit.require(current == user_snapshot, "Private dpkg purge changed or deleted per-user data")


def verify(args, workspace, audit):
    if os.geteuid() == 0:
        raise RuntimeError("Run this verifier as an unprivileged user; namespace root is created privately")
    for tool in ("bwrap", "dpkg", "dpkg-deb", "dpkg-query"):
        if not shutil.which(tool):
            raise RuntimeError("Required tool unavailable: " + tool)
    if not args.deb.is_file():
        raise RuntimeError("DEB does not exist: " + str(args.deb))
    architecture = execute(["dpkg", "--print-architecture"]).stdout.strip()
    if architecture != "amd64":
        raise RuntimeError("Installation QA requires the target amd64 distribution")
    result = execute(["dpkg-deb", "-f", str(args.deb)])
    if result.returncode:
        raise RuntimeError(result.stderr)
    control = deb_fields(result.stdout)
    audit.require(control.get("Package") == "vision-studio" and control.get("Architecture") == "amd64",
                  "Unexpected package or architecture")
    archive_members(args.deb, audit)
    # List control members directly, never execute a package's script. An
    # application requiring maintainer scripts needs a separately reviewed QA.
    control_dir = workspace / "control"
    result = execute(["dpkg-deb", "-e", str(args.deb), str(control_dir)])
    if result.returncode:
        raise RuntimeError(result.stderr)
    for name in ("preinst", "postinst", "prerm", "postrm", "config"):
        audit.require(not (control_dir / name).exists(), "Package contains a maintainer script: " + name)
    if audit.errors:
        raise RuntimeError("Package archive/control safety audit failed; installation was not attempted")
    prepare(workspace)
    shutil.copyfile(args.deb, workspace / "input.deb")
    namespace = Namespace(workspace)
    probe, _ = namespace.run(["/usr/bin/id", "-u"], "namespace-root", root=True)
    if probe.returncode or probe.stdout.strip() != "0":
        raise RuntimeError("Private UID0 namespace unavailable; host installation was not attempted: " + probe.stderr[-1500:])
    probe, _ = namespace.run(["/usr/bin/id", "-u"], "namespace-user", readonly=True)
    if probe.returncode or probe.stdout.strip() != str(os.geteuid()):
        raise RuntimeError("Cannot guarantee non-root inference namespace: " + probe.stderr[-1500:])
    audit.details["namespace"] = {"host_uid": os.geteuid(), "installer_namespace_uid": 0,
                                   "inference_namespace_uid": os.geteuid(), "host_filesystem": "read-only",
                                   "network": "isolated/offline", "maintainer_scripts": False, "triggers": False,
                                   "skipped_broken_host_tool_aliases": namespace.skipped_broken_bins}
    result, elapsed = namespace.run(["/usr/bin/dpkg", "--no-triggers", "--log=" + MOUNT + "/logs/dpkg.log",
                                      "--install", MOUNT + "/input.deb"], "install", root=True, timeout=600)
    audit.details["installation"] = {"exit_code": result.returncode, "elapsed_seconds": elapsed,
                                       "stdout": result.stdout, "stderr": result.stderr}
    if not audit.require(result.returncode == 0, "Private dpkg installation failed: " + result.stderr[-2000:]):
        return
    for relative in ("opt/VisionStudio/build/bin/vision-studio", "usr-bin/vision-studio",
                     "applications/vision-studio.desktop", "icons/vision-studio.svg", "documentation/vision-studio/copyright"):
        audit.require((workspace / relative).is_file(), "Actual installation did not create required payload: " + relative)
    result, _ = namespace.run(["/usr/bin/dpkg-query", "-W", "-f=${Status}\\n", "vision-studio"],
                              "installed-status", readonly=True)
    audit.require(result.returncode == 0 and result.stdout.strip() == "install ok installed",
                  "Private dpkg database does not mark the application installed")
    result, _ = namespace.run(["/usr/bin/python3", "-c",
                              "import errno,os; assert os.geteuid()!=0; "
                              "\ntry: open('/opt/VisionStudio/.qa-write-probe','w').close()"
                              "\nexcept OSError as e: assert e.errno==errno.EROFS"
                              "\nelse: raise RuntimeError('Installed /opt tree is writable')"],
                              "read-only-user", readonly=True)
    audit.require(result.returncode == 0, "Non-root read-only /opt assertion failed: " + result.stderr[-1200:])
    # Test the actual fixed /opt path and actual /usr/bin launcher. The whole
    # installed tree is mounted read-only, including when owned by this user.
    smoke = {}
    for name, flag in (("onnx", "--smoke"), ("pt", "--smoke-pt")):
        result, elapsed = namespace.run(["/usr/bin/vision-studio", flag, MOUNT + "/user/" + name],
                                        name, readonly=True, timeout=180)
        path = workspace / "user" / name / "smoke-report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        passed = result.returncode == 0 and report.get("success") is True and report.get("predictions", 0) > 0
        audit.require(passed, f"Installed {name} inference failed as non-root: " + result.stderr[-1500:])
        smoke[name] = {"success": passed, "exit_code": result.returncode, "elapsed_seconds": elapsed, "report": report}
    result, elapsed = namespace.run(["/usr/bin/vision-studio", "--smoke-model", MOUNT + "/user/model-display"],
                                    "model-display", readonly=True, timeout=180)
    display_root = workspace / "user/model-display"
    path = display_root / "model-display-report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    nodes = report.get("nodes", 0)
    screenshot = display_root / "model-display.png"
    passed = (result.returncode == 0 and report.get("success") is True
              and isinstance(nodes, int) and not isinstance(nodes, bool) and nodes > 0
              and screenshot.is_file() and screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
    audit.require(passed, "Installed offline non-root model structure viewer failed: " + result.stderr[-1800:])
    smoke["model_display"] = {"success": passed, "exit_code": result.returncode,
                               "elapsed_seconds": elapsed, "report": report,
                               "rendering_flags": "--disable-gpu", "sandbox_disabled_by_test": False}
    audit.details["inference"] = smoke
    audit.require(bool(list((workspace / "user").rglob("preferences.ini"))), "Installed application did not persist isolated preferences")
    audit.require(bool(list((workspace / "user").rglob("history.json"))), "Installed application did not persist isolated history")
    snapshot = {str(path.relative_to(workspace / "user")): sha256(path)
                for path in (workspace / "user").rglob("*") if path.is_file() and not path.is_symlink()}
    result, elapsed = namespace.run(["/usr/bin/dpkg", "--no-triggers", "--log=" + MOUNT + "/logs/dpkg.log",
                                      "--purge", "vision-studio"], "purge", root=True, timeout=300)
    audit.details["purge"] = {"exit_code": result.returncode, "elapsed_seconds": elapsed,
                              "stdout": result.stdout, "stderr": result.stderr}
    audit.require(result.returncode == 0, "Private dpkg purge failed: " + result.stderr[-1800:])
    uninstall_checks(workspace, audit, snapshot)
    audit.details["user_data_preserved_after_purge"] = bool(snapshot) and all(
        (workspace / "user" / name).is_file() and sha256(workspace / "user" / name) == digest
        for name, digest in snapshot.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deb", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    workspace = Path(tempfile.mkdtemp(prefix="vision-studio-install-qa-"))
    audit = Audit()
    audit.details["deb"] = str(args.deb.resolve())
    host_status = Path("/var/lib/dpkg/status")
    original_host_status = sha256(host_status)
    integration = [Path("/usr/bin/vision-studio"), Path("/usr/share/applications/vision-studio.desktop"),
                   Path("/usr/share/icons/hicolor/scalable/apps/vision-studio.svg")]
    original_integration = {str(path): sha256(path) if path.is_file() else None for path in integration}
    try:
        audit.details["deb_sha256"] = sha256(args.deb)
        verify(args, workspace, audit)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError) as error:
        audit.errors.append(str(error))
    finally:
        audit.details["host_dpkg_status_unchanged"] = sha256(host_status) == original_host_status
        audit.require(audit.details["host_dpkg_status_unchanged"], "Host dpkg status changed during namespace QA")
        current_integration = {str(path): sha256(path) if path.is_file() else None for path in integration}
        audit.details["host_integration_unchanged"] = current_integration == original_integration
        audit.require(audit.details["host_integration_unchanged"], "Host launcher/desktop/icon changed during namespace QA")
        report = {"success": not audit.errors, "errors": audit.errors, "warnings": audit.warnings, "details": audit.details}
        if args.keep_workdir:
            report["workdir"] = str(workspace)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"success": report["success"], "errors": audit.errors,
                          "report": str(args.report) if args.report else None,
                          "workdir": str(workspace) if args.keep_workdir else None}, ensure_ascii=False))
        if not args.keep_workdir:
            shutil.rmtree(workspace)
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
