#!/usr/bin/env python3
"""Optional Windows CUDA runtime, isolated from the offline CPU installation."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

PINS = {'torch_version':'2.9.1+cu128', 'torch_cuda':'12.8', 'ort_version':'1.23.2'}
INSTALLER = 'vision-studio-gpu-windows'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def emit(event, **fields):
    print(json.dumps({'event':event, **fields}, ensure_ascii=False), flush=True)


def default_runtime():
    data = os.environ.get('VISION_STUDIO_DATA_DIR') or str(Path(os.environ['LOCALAPPDATA']) / 'vision-studio')
    return Path(os.environ.get('VISION_STUDIO_GPU_RUNTIME_DIR', str(Path(data) / 'gpu-runtime')))


def real_path(path):
    return not any(p.exists() and p.lstat().st_file_attributes & 0x400 for p in [path, *path.parents])


def managed(path):
    try:
        marker = json.loads((path / 'ready.json').read_text(encoding='utf-8'))
        return real_path(path) and marker.get('installer') == INSTALLER and all(marker.get(k) == v for k,v in PINS.items())
    except (OSError, ValueError):
        return False


def hardware():
    from gpu_probe import hardware as inspect
    return inspect()


def probe(runtime, candidate=False):
    from gpu_probe import CHILD_PROBE
    devices, driver = hardware()
    result = dict(ok=False, prepared=False, cuda_available=False, reason='GPU 环境尚未准备',
                  devices=devices, driver_version=driver, torch_version='', torch_cuda='', ort_version='')
    try:
        if not real_path(runtime) or not (runtime / 'python.exe').is_file() or not (candidate or managed(runtime)):
            return result
    except OSError as error:
        result['reason'] = 'GPU 环境无法读取，请检查所选目录的访问权限：' + str(error)
        return result
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    config = Path(os.environ.get('VISION_STUDIO_DATA_DIR', runtime.parent)) / 'gpu-probe-config'
    config.mkdir(parents=True, exist_ok=True)
    env.update(YOLO_CONFIG_DIR=str(config), MPLCONFIGDIR=str(config))
    try:
        child = subprocess.run([str(runtime / 'python.exe'), '-I', '-c', CHILD_PROBE], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
            timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(json.loads(child.stdout.strip().splitlines()[-1]))
        result['driver_version'] = driver
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        result['reason'] = 'GPU 自检失败：' + str(error)
    return result


def contain_children():
    # Closing this job when QProcess kills the preparer also kills pip workers.
    class Basic(ctypes.Structure):
        _fields_ = [('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),
                    ('min_work',ctypes.c_size_t),('max_work',ctypes.c_size_t),('active',wintypes.DWORD),
                    ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(n,ctypes.c_uint64) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Limits(ctypes.Structure):
        _fields_ = [('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                    ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None,None)
    limits = Limits(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job,kernel.GetCurrentProcess()):
        raise RuntimeError('无法建立 GPU 安装进程保护，已停止准备。')
    return job


def inventory(runtime):
    code = r'''
import importlib.metadata as m,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); records=[]
for d in sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()):
 records.append({'name':d.metadata['Name'],'version':d.version,'license':d.metadata.get('License-Expression',d.metadata.get('License',''))})
(root/'gpu-license-inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'installed-packages.txt').write_text(''.join(d['name']+'=='+d['version']+'\n' for d in records),encoding='utf-8')
'''
    subprocess.run([str(runtime/'python.exe'),'-I','-c',code,str(runtime)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    (runtime/'GPU-THIRD-PARTY-NOTICES.txt').write_text(
        'Optional Windows NVIDIA runtime. Original wheel licenses remain in Lib/site-packages.\n'
        'CUDA/cuDNN retain NVIDIA proprietary terms and are not relicensed under AGPL.\n'
        'Sources: https://download.pytorch.org/whl/cu128/ and https://pypi.org/\n',encoding='utf-8')


def setup(runtime, wheelhouse=None):
    runtime = runtime.absolute()
    source = Path(sys.executable).resolve().parent
    if len(runtime.parts) < 4 or not real_path(runtime) or runtime == source or source in runtime.parents:
        raise RuntimeError('请选择独立、可写的用户 GPU 目录。')
    runtime.parent.mkdir(parents=True, exist_ok=True)
    lockfile = runtime.parent / ('.' + runtime.name + '.setup.lock')
    with lockfile.open('a+b') as guard:
        guard.seek(0); guard.write(b'0'); guard.flush(); guard.seek(0)
        try:
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('另一个 GPU 准备任务正在运行。')
        recovery = runtime.parent / ('.' + runtime.name + '.recovery.json')
        if recovery.exists():
            backup = Path(json.loads(recovery.read_text(encoding='utf-8'))['backup'])
            if backup.parent == runtime.parent and backup.name.startswith('.'+runtime.name+'.previous-') and managed(backup) and not runtime.exists():
                backup.rename(runtime)
            if runtime.exists(): recovery.unlink()
        if runtime.exists():
            if not managed(runtime): raise RuntimeError('目标目录含有未知内容，已保留。请选择新的 GPU 目录。')
            ready = probe(runtime)
            if ready['ok']:
                emit('ready',message='GPU 环境已准备完成。',runtime_dir=str(runtime)); return
        job = contain_children()
        stage = Path(tempfile.mkdtemp(prefix='.'+runtime.name+'.staging-',dir=runtime.parent))
        identity = stage.stat().st_ino
        backup = None
        try:
            emit('progress',message='复制独立基础环境；保留原 CPU 环境。',percent=5)
            def omit(directory, names):
                result = {n for n in names if n in {'__pycache__','Scripts','include','libs'} or n.endswith(('.pyc','.pdb'))}
                if Path(directory).name == 'site-packages':
                    result.update(n for n in names if n in {'torch','torchgen','functorch','torchvision','onnxruntime'} or n.startswith(('torch-','torchvision-','onnxruntime','coloredlogs-','flatbuffers-','humanfriendly-')))
                return result
            shutil.copytree(source,stage,dirs_exist_ok=True,ignore=omit)
            lock = Path(__file__).resolve().parent.parent / 'requirements-gpu-windows.lock.txt'
            if not lock.is_file(): raise RuntimeError('缺少 Windows GPU 哈希锁。')
            command = [str(stage/'python.exe'),'-I','-m','pip','install','--no-deps','--require-hashes',
                       '--only-binary=:all:','--disable-pip-version-check','-r',str(lock)]
            if wheelhouse:
                command += ['--no-index','--find-links',str(wheelhouse)]
            else:
                command += ['--extra-index-url','https://download.pytorch.org/whl/cu128']
            emit('progress',message='下载并核验 Windows CUDA 依赖（首次约 3 GB）。',percent=15)
            subprocess.run(command,check=True,stdout=sys.stderr,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            emit('progress',message='执行真实 PyTorch 和 ONNX CUDA 自检。',percent=85)
            ready = probe(stage,candidate=True)
            if not ready['ok']: raise RuntimeError(ready['reason'])
            inventory(stage)
            (stage/'ready.json').write_text(json.dumps({'schema':1,'installer':INSTALLER,**PINS},indent=2)+'\n',encoding='utf-8')
            shutil.copy2(lock,stage/'requirements-gpu-windows.lock.txt')
            if runtime.exists():
                backup = runtime.with_name('.'+runtime.name+'.previous-'+uuid.uuid4().hex)
                recovery.write_text(json.dumps({'backup':str(backup)}),encoding='utf-8')
                runtime.rename(backup)
            stage.rename(runtime)
            final = probe(runtime)
            if not final['ok']:
                runtime.rename(stage)
                if backup: backup.rename(runtime)
                raise RuntimeError(final['reason'])
            if recovery.exists(): recovery.unlink()
            # Keep the known previous environment as an explicit recovery copy.
            emit('ready',message='GPU 环境自检通过。',runtime_dir=str(runtime),backup_dir=str(backup or ''))
        finally:
            if stage.exists() and real_path(stage) and stage.stat().st_ino == identity:
                shutil.rmtree(stage)


def setup_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir',type=Path,default=default_runtime())
    parser.add_argument('--wheelhouse',type=Path)
    args=parser.parse_args()
    try:
        setup(args.runtime_dir,args.wheelhouse);return 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        emit('error',message=str(error));return 1


def probe_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir',type=Path,default=default_runtime())
    parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args();result=probe(args.runtime_dir.absolute(),args.candidate)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['ok'] else 1
