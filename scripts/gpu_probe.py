#!/usr/bin/env python3
"""Read-only NVIDIA runtime probe; stdout is exactly one JSON object."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

PINS = {"torch_version": "2.9.1+cu128", "torch_cuda": "12.8", "ort_version": "1.23.2"}
INSTALLER = "vision-studio-gpu"


class ProbeCancelled(BaseException):
    pass


def terminate_private_group(process):
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Retain the unreaped group leader during this short grace period so
        # its PID cannot be reused. Kill remaining grandchildren even if their
        # leader exited and they redirected all inherited pipes.
        time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=2)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def managed_command(command, *, env=None, timeout=120):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        terminate_private_group(process)
        raise


def default_runtime() -> Path:
    if os.environ.get("VISION_STUDIO_GPU_RUNTIME_DIR"):
        return Path(os.environ["VISION_STUDIO_GPU_RUNTIME_DIR"])
    data = os.environ.get("VISION_STUDIO_DATA_DIR")
    if not data:
        data = str(Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "vision-studio")
    return Path(data) / "gpu-runtime"


def hardware() -> tuple[list[dict], str]:
    try:
        result = managed_command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                  "--format=csv,noheader,nounits"], timeout=8)
        if result.returncode:
            return [], ""
        devices, driver = [], ""
        for line in result.stdout.splitlines():
            index, name, memory, driver = [value.strip() for value in line.split(",", 3)]
            devices.append({"index": int(index), "name": name, "total_memory_mb": int(float(memory))})
        return devices, driver
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], ""


def owned_runtime(path: Path) -> bool:
    """Do not follow a user-supplied directory symlink or run another user's venv."""
    absolute = path.absolute()
    if any(part.is_symlink() for part in [absolute, *absolute.parents]):
        return False
    try:
        return absolute.is_dir() and absolute.stat().st_uid == os.getuid() and (absolute / "pyvenv.cfg").is_file()
    except OSError:
        return False


# A small opset-13 MatMul graph, encoded using protobuf wire format. Keeping
# this self-contained avoids adding the onnx authoring package to inference.
CHILD_PROBE = r'''
import json, sys, importlib.metadata as metadata
def varint(n):
    result=bytearray()
    while n>127: result.append((n & 127)|128); n >>= 7
    result.append(n); return bytes(result)
def field(n,value):
    if isinstance(value,int): return varint(n<<3)+varint(value)
    if isinstance(value,str): value=value.encode()
    return varint((n<<3)|2)+varint(len(value))+value
def tensor_info(name):
    dimension=field(1,2)
    shape=field(1,dimension)+field(1,dimension)
    tensor=field(1,1)+field(2,shape)
    return field(1,name)+field(2,field(1,tensor))
def model():
    node=field(1,'x')+field(1,'x')+field(2,'y')+field(3,'gpu-probe')+field(4,'MatMul')
    graph=field(1,node)+field(2,'vision-studio-gpu-probe')+field(11,tensor_info('x'))+field(12,tensor_info('y'))
    return field(1,8)+field(2,'Vision Studio')+field(7,graph)+field(8,field(2,13))
result={'ok':False,'prepared':False,'cuda_available':False,'reason':'','devices':[],
        'torch_version':'','torch_cuda':'','ort_version':'','driver_version':''}
try:
    import torch, torchvision, numpy as np
    import onnxruntime as ort
    result.update(torch_version=torch.__version__,torch_cuda=torch.version.cuda or '',ort_version=ort.__version__)
    if (torch.__version__,torchvision.__version__,torch.version.cuda,ort.__version__) != ('2.9.1+cu128','0.24.1+cu128','12.8','1.23.2'):
        raise RuntimeError('GPU 依赖版本与发行锁不一致')
    # All three PT loaders need their original CPU runtime dependencies.
    import ultralytics, pandas, seaborn, IPython, pkg_resources, cv2
    if metadata.version('ultralytics')!='8.4.173' or metadata.version('opencv-python-headless')!='4.11.0.86':
        raise RuntimeError('YOLO / OpenCV 依赖版本不匹配')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA 不可用；请检查已安装的 NVIDIA 驱动和显卡权限')
    for index in range(torch.cuda.device_count()):
        properties=torch.cuda.get_device_properties(index)
        result['devices'].append({'index':index,'name':properties.name,'total_memory_mb':properties.total_memory//(1024*1024)})
        value=torch.ones((2,2),device='cuda:'+str(index))
        assert (value @ value).sum().item()==8
        torch.cuda.synchronize(index)
    ort.preload_dlls(cuda=True,cudnn=True)
    options=ort.SessionOptions(); options.log_severity_level=3
    session=ort.InferenceSession(model(),sess_options=options,providers=[('CUDAExecutionProvider',{'device_id':0})])
    if not session.get_providers() or session.get_providers()[0]!='CUDAExecutionProvider':
        raise RuntimeError('ONNX Runtime CUDA provider 无法加载')
    session.disable_fallback()
    np.testing.assert_array_equal(session.run(None,{'x':np.ones((2,2),dtype=np.float32)})[0],np.full((2,2),2,dtype=np.float32))
    result.update(ok=True,prepared=True,cuda_available=True)
except Exception as error:
    result['reason']=str(error)
print(json.dumps(result,ensure_ascii=False))
sys.exit(0 if result['ok'] else 1)
'''


def probe(runtime: Path, *, candidate: bool = False) -> dict:
    devices, driver = hardware()
    result = {"ok": False, "prepared": False, "cuda_available": False,
              "reason": "GPU 环境尚未准备", "devices": devices, "torch_version": "",
              "torch_cuda": "", "ort_version": "", "driver_version": driver}
    if not owned_runtime(runtime):
        if runtime.exists() or runtime.is_symlink():
            result["reason"] = "GPU 环境目录不可信或缺少独立 Python 环境"
        return result
    if not candidate:
        try:
            path = runtime / "ready.json"
            if path.is_symlink() or path.stat().st_size > 65536:
                raise ValueError("invalid ready marker")
            ready = json.loads(path.read_text(encoding="utf-8"))
            if ready.get("schema") != 1 or ready.get("installer") != INSTALLER or any(ready.get(key) != value for key, value in PINS.items()):
                raise ValueError("invalid ready marker")
        except (OSError, ValueError, TypeError):
            result["reason"] = "GPU 环境尚未准备或版本记录不匹配"
            return result
    executable = runtime / "bin/python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        result["reason"] = "GPU 环境缺少可运行的 Python"
        return result
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        process = managed_command([str(executable), "-I", "-c", CHILD_PROBE], env=env, timeout=120)
        child = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(child, dict):
            raise ValueError("invalid child response")
        result.update(child)
        result["driver_version"] = driver
        if not result["devices"]:
            result["devices"] = devices
        if process.returncode and not result["reason"]:
            result["reason"] = "GPU 环境自检失败"
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        result["reason"] = "GPU 环境自检失败：" + str(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=default_runtime())
    parser.add_argument("--candidate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    def cancel(signum, frame):
        raise ProbeCancelled()
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        result = probe(args.runtime_dir.absolute(), candidate=args.candidate)
    except ProbeCancelled:
        result = {"ok": False, "prepared": False, "cuda_available": False, "reason": "GPU 检查已取消",
                  "devices": [], "torch_version": "", "torch_cuda": "", "ort_version": "", "driver_version": ""}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 130
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
