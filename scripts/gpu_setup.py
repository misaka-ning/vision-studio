#!/usr/bin/env python3
"""Install a pinned, per-user NVIDIA runtime without altering CPU/driver files."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from gpu_probe import INSTALLER, PINS, default_runtime, owned_runtime, probe, terminate_private_group

# Publisher SHA256 from the fixed official CUDA 12.8 wheel index. Use canonical
# CloudFront links; the index's R2 endpoint can return HTTP 403 in some networks.
TORCH_WHEELS = (
    ("torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", "c8d670aa0be6fbecd2b0e7b7d514a104dbdefcc3786ca446cf0c3415043ea40a"),
    ("torch-2.9.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl", "2a1da940f0757621d098c9755f7504d791a72a40920ec85a4fd98b20253fca4e"),
    ("torchvision-0.24.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl", "87f6e32d2d98d2779f9fa58aa93cf6af049e3f87681344a2088e2fc7d9dd3e00"),
    ("torchvision-0.24.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl", "1b44f67cbd8f36e2a58bfaa3176d35b37df55604adf5929e89006e531f849faa"),
)


def canonical_links(lock_text: str) -> str:
    allowed = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", lock_text))
    if any(digest not in allowed for _, digest in TORCH_WHEELS):
        raise RuntimeError("官方 PyTorch 下载链接与 GPU 哈希锁不一致")
    return "<!doctype html><html><body>\n" + "\n".join(
        '<a href="https://download.pytorch.org/whl/cu128/' + name.replace("+", "%2B") + '#sha256=' + digest + '">' + name + '</a>'
        for name, digest in TORCH_WHEELS) + "\n</body></html>\n"


def complete_wheelhouse(directory: Path | None, pins: dict, version: list[int], lock_text: str) -> bool:
    if directory is None:
        return False
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    required = {normalize(name): value for name, value in pins.items()}
    allowed = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", lock_text))
    available = set()
    tag = "cp" + str(version[0]) + str(version[1])
    for path in directory.glob("*.whl"):
        if not path.is_file():
            continue
        try:
            name, distribution_version, python, abi, platform = path.name[:-4].rsplit("-", 4)
        except ValueError:
            continue
        name = normalize(name)
        if required.get(name) != distribution_version or name in available:
            continue
        if not (platform == "any" or "x86_64" in platform and ("manylinux" in platform or platform.startswith("linux_"))):
            continue
        if any(int(item) > 35 for item in re.findall(r"manylinux_2_(\d+)", platform)):
            continue
        if not ("py3" in python or tag in python.split(".") or abi == "abi3" and any(int(item) <= int(tag[2:]) for item in re.findall(r"cp(\d+)", python))):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() in allowed:
            available.add(name)
    return available == set(required)


class Cancelled(Exception):
    pass


def event(name: str, **values) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def cancelled(signum, frame):
    raise Cancelled("GPU 环境准备已取消")


def run(command: list[str], *, env: dict | None = None) -> None:
    # The private group contains only children started by this setup process.
    process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr,
                               env=env, start_new_session=True)
    try:
        code = process.wait()
    except BaseException:
        terminate_private_group(process)
        raise
    if code:
        raise RuntimeError(f"依赖配置步骤失败（退出码 {code}），请查看安装日志后重试")


def read_pins(lock: Path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock.read_text(encoding="utf-8"), re.M))


def managed(path: Path) -> bool:
    if not owned_runtime(path):
        return False
    try:
        marker = path / "ready.json"
        if marker.is_symlink() or marker.stat().st_size > 65536:
            return False
        ready = json.loads(marker.read_text(encoding="utf-8"))
        return ready.get("schema") == 1 and ready.get("installer") == INSTALLER and all(ready.get(key) == value for key, value in PINS.items())
    except (OSError, ValueError, TypeError):
        return False


def validate_destination(runtime: Path) -> None:
    if not runtime.is_absolute() or runtime.name in {"", ".", ".."} or len(runtime.parts) < 4:
        raise RuntimeError("请选择专用的用户 GPU 环境目录")
    if runtime.parts[1] in {"opt", "usr", "etc", "bin", "sbin", "lib", "lib64", "var", "proc", "sys", "dev"}:
        raise RuntimeError("GPU 环境只能写入用户目录，不能修改系统或 CPU 安装目录")
    if any(path.is_symlink() for path in [runtime, *runtime.parents]):
        raise RuntimeError("GPU 环境目录及父目录不能是符号链接")
    runtime.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if runtime.parent.stat().st_uid != os.getuid() or not os.access(runtime.parent, os.W_OK):
        raise RuntimeError("GPU 环境父目录不属于当前用户或不可写")
    if runtime.exists() and not managed(runtime):
        raise RuntimeError("目标目录已有未知内容；为保护数据，请选择新的专用目录")


def atomic_rename(first: Path, second: Path, flags: int) -> None:
    """Linux atomic exchange preserves the old runtime if any publication fails."""
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RuntimeError("当前系统不支持安全原子替换；原 GPU 环境已保留")
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(first), -100, os.fsencode(second), flags):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def exchange(first: Path, second: Path) -> None:
    atomic_rename(first, second, 2)


def publish_checked(stage: Path, runtime: Path, private_stages: set[Path]) -> None:
    replacing = runtime.exists()
    changed = False
    try:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            if replacing:
                if not managed(runtime):
                    raise RuntimeError("原 GPU 环境发生变化，已保留原文件")
                exchange(stage, runtime)
            else:
                atomic_rename(stage, runtime, 1)
            changed = True
        finally:
            # A pending cancellation is delivered inside the rollback guard,
            # after publication state was recorded, never between those steps.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        final = probe(runtime)
        if not final["ok"]:
            raise RuntimeError(final["reason"])
    except BaseException:
        if changed:
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            if replacing:
                # If rollback itself fails, this is the unique old environment
                # and must not be removed by staging cleanup.
                private_stages.discard(stage)
            try:
                if replacing:
                    exchange(stage, runtime)
                    private_stages.add(stage)
                else:
                    atomic_rename(runtime, stage, 1)
            except OSError as error:
                event("recovery", message="GPU 发布回滚失败，已保留环境备份", backup_dir=str(stage if replacing else runtime))
                raise RuntimeError("GPU 发布回滚失败，环境已保留：" + str(stage if replacing else runtime)) from error
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        raise


def relocate_entries(stage: Path, destination: Path) -> None:
    # Only venv-generated textual entry points/activation files change. Wheels,
    # including every NVIDIA binary, remain byte-for-byte as downloaded.
    original = str(stage).encode()
    replacement = str(destination).encode()
    for path in (stage / "bin").iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            continue
        content = path.read_bytes()
        if original in content and b"\0" not in content:
            path.write_bytes(content.replace(original, replacement))


INVENTORY = r'''
import importlib.metadata as metadata, json, pathlib, re, shutil, sys
root=pathlib.Path(sys.argv[1]); pins=json.loads(sys.argv[2])
normalize=lambda name: re.sub(r'[-_.]+','-',name).lower()
installed={normalize(d.metadata['Name']):d.version for d in metadata.distributions()}
expected={normalize(name):version for name,version in pins.items()}
if installed!=expected: raise RuntimeError('Installed GPU packages differ from the complete lock: '+str(set(installed.items())^set(expected.items())))
records=[]
for distribution in metadata.distributions():
    name=distribution.metadata['Name']
    copied=[]
    for item in distribution.files or []:
        parts=pathlib.PurePosixPath(str(item)).parts
        base=pathlib.PurePosixPath(str(item)).name.lower()
        if not (base.startswith(('license','licence','copying','notice','thirdpartynotice','third_party_notice')) or any(p.lower()=='licenses' for p in parts)): continue
        if '..' in parts or pathlib.PurePosixPath(str(item)).is_absolute(): continue
        source=pathlib.Path(distribution.locate_file(item))
        if not source.is_file() or source.is_symlink(): continue
        target=root/'licenses'/normalize(name)/pathlib.Path(*parts)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        copied.append(str(target.relative_to(root)))
    records.append({'name':name,'version':distribution.version,'license_expression':distribution.metadata.get('License-Expression',''),
                    'license':distribution.metadata.get('License',''),'license_files':copied})
(root/'installed-packages.txt').write_text(''.join(name+'=='+version+'\n' for name,version in sorted(installed.items())))
(root/'gpu-license-inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
(root/'GPU-THIRD-PARTY-NOTICES.md').write_text('# Optional NVIDIA runtime components\n\nThis user-installed environment is separate from the AGPL application and bundled CPU runtime.\nEach dependency retains its original license; NVIDIA CUDA/cuDNN redistributables are proprietary, not relicensed under AGPL.\nPublisher license texts supplied by the wheels are preserved under licenses/. Exact versions and metadata are in gpu-license-inventory.json.\nOfficial sources: https://download.pytorch.org/whl/cu128/ and https://pypi.org/ .\nCUDA terms: https://docs.nvidia.com/cuda/eula/index.html\ncuDNN terms: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html\n')
'''


def setup(runtime: Path, base_python: Path, wheelhouse: Path | None = None) -> None:
    validate_destination(runtime)
    lock_path = runtime.parent / ("." + runtime.name + ".setup.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage: Path | None = None
    private_stages: set[Path] = set()
    with os.fdopen(descriptor, "w") as guard:
        if os.fstat(guard.fileno()).st_uid != os.getuid():
            raise RuntimeError("GPU 配置锁不属于当前用户")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("另一个 GPU 环境准备任务正在运行") from None
        validate_destination(runtime)
        if runtime.exists():
            ready = probe(runtime)
            if ready["ok"]:
                event("ready", runtime_dir=str(runtime), reused=True)
                return
        project = Path(__file__).resolve().parent.parent
        lock = project / "requirements-gpu.lock.txt"
        requirements = project / "requirements-gpu.txt"
        if not lock.is_file() or not requirements.is_file():
            raise RuntimeError("缺少发行版 GPU 依赖锁")
        pins = read_pins(lock)
        if pins.get("torch") != PINS["torch_version"] or pins.get("torchvision") != "0.24.1+cu128" or pins.get("onnxruntime-gpu") != PINS["ort_version"]:
            raise RuntimeError("GPU 依赖锁不匹配")
        if not base_python.is_absolute() or not base_python.is_file():
            raise RuntimeError("请使用系统 Python 3.10 或 3.11 的绝对路径")
        check = subprocess.run([str(base_python), "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"], capture_output=True, text=True, timeout=10, check=True)
        version = json.loads(check.stdout)
        if version not in ([3, 10], [3, 11]):
            raise RuntimeError("GPU 与旧 YOLOv5 环境需要 Python 3.10 / 3.11；不能使用 Conda 3.14")
        env = dict(os.environ)
        for key in list(env):
            if key not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH") and not key.startswith("PIP_"):
                continue
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE=os.devnull,
                   PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
        local_wheels = []
        if wheelhouse is not None:
            if not wheelhouse.is_absolute() or not wheelhouse.is_dir():
                raise RuntimeError("本地官方 wheel 缓存目录不存在")
            # Every candidate still has to match a publisher hash in the lock.
            local_wheels = ["--find-links", str(wheelhouse)]
        offline = complete_wheelhouse(wheelhouse, pins, version, lock.read_text(encoding="utf-8"))
        if offline:
            local_wheels.insert(0, "--no-index")
        stage = Path(tempfile.mkdtemp(prefix="vision-studio-gpu-", dir="/tmp"))
        private_stages.add(stage)
        try:
            event("progress", phase="environment", message="创建独立 GPU 环境", percent=5)
            run([str(base_python), "-I", "-m", "venv", str(stage)], env=env)
            links = stage / "torch-wheel-links.html"
            links.write_text(canonical_links(lock.read_text(encoding="utf-8")), encoding="utf-8")
            inference_wheels = local_wheels if offline else local_wheels + ["--find-links", str(links)]
            # Ubuntu's ensurepip predates recoverable downloads. Bootstrap a
            # hash-pinned pip from this same lock, only inside the staging venv.
            bootstrap = stage / "bootstrap-pip.txt"
            block = re.search(r"(?m)^pip==[^\n]+(?:\n[ \t]+--hash=sha256:[^\n]+)+", lock.read_text(encoding="utf-8"))
            if not block:
                raise RuntimeError("GPU 锁缺少可恢复下载所需的 pip 校验值")
            bootstrap.write_text("--index-url https://pypi.org/simple\n" + block.group(0) + "\n", encoding="utf-8")
            run([str(stage / "bin/python"), "-I", "-m", "pip", "install", "--no-deps", "--require-hashes",
                 "--only-binary=:all:", "--timeout", "120", "--retries", "10", *local_wheels, "-r", str(bootstrap)], env=env)
            bootstrap.unlink()
            event("progress", phase="dependencies", message="从官方源下载并校验 GPU 依赖；首次下载约 4–5 GB", percent=15)
            run([str(stage / "bin/python"), "-I", "-m", "pip", "install", "--no-deps", "--require-hashes",
                 "--only-binary=:all:", "--progress-bar", "off", "--timeout", "120", "--retries", "10",
                 "--resume-retries", "10", *inference_wheels, "-r", str(requirements)], env=env)
            event("progress", phase="verify", message="核对依赖、PyTorch CUDA 和 ONNX Runtime", percent=85)
            run([str(stage / "bin/python"), "-I", "-c", INVENTORY, str(stage), json.dumps(pins)], env=env)
            ready = probe(stage, candidate=True)
            if not ready["ok"]:
                raise RuntimeError(ready["reason"] or "GPU 实际计算自检未通过")
            metadata = {"schema": 1, "installer": INSTALLER, **PINS, "pins": pins}
            (stage / "ready.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            shutil.copy2(lock, stage / "requirements-gpu.lock.txt")
            relocate_entries(stage, runtime)
            # Cross-filesystem installation first copies into a private sibling;
            # the final publish still uses atomic rename on one filesystem.
            if stage.stat().st_dev != runtime.parent.stat().st_dev:
                sibling = Path(tempfile.mkdtemp(prefix="." + runtime.name + ".staging-", dir=runtime.parent))
                private_stages.add(sibling)
                sibling.rmdir()
                try:
                    shutil.copytree(stage, sibling, symlinks=True)
                except BaseException:
                    shutil.rmtree(sibling, ignore_errors=True)
                    raise
                shutil.rmtree(stage)
                stage = sibling
            event("progress", phase="publish", message="发布已通过自检的 GPU 环境", percent=95)
            replacing = runtime.exists()
            publish_checked(stage, runtime, private_stages)
            if replacing:
                # stage now contains only the old environment we validated.
                shutil.rmtree(stage)
                stage = None
            else:
                stage = None
            event("ready", runtime_dir=str(runtime))
        finally:
            for private in private_stages:
                if private.exists():
                    shutil.rmtree(private, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=default_runtime())
    parser.add_argument("--base-python", type=Path, default=Path("/usr/bin/python3.10"))
    parser.add_argument("--wheelhouse", type=Path, help="Optional local official wheels, still verified against lock hashes")
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        setup(args.runtime_dir.absolute(), args.base_python, args.wheelhouse)
        return 0
    except Cancelled as error:
        event("cancelled", message=str(error))
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        event("error", message=str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
