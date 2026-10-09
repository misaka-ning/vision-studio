#!/usr/bin/env python3
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[2]
evidence = root/'docs/releases/evidence/2.1.0/conversion-portability-proof'
evidence.mkdir(exist_ok=True)
log = evidence/'ctest-full-model-conversion.log'
if log.exists():
    raise SystemExit('Preserve existing actual run instead of overwriting it')
state = root/'output/v2.1.0/full-model-conversion-qa-state'
env = os.environ.copy()
env.update(json.loads((root/'docs/releases/evidence/2.1.0/ctest-all.log.execution.json').read_text())['environment'])
for key, directory in {
    'HOME':'home','XDG_CONFIG_HOME':'config','XDG_DATA_HOME':'data','XDG_CACHE_HOME':'cache',
    'XDG_STATE_HOME':'state','XDG_RUNTIME_DIR':'runtime','VISION_STUDIO_DATA_DIR':'app-data',
    'YOLO_CONFIG_DIR':'ultralytics','YOLOV5_CONFIG_DIR':'yolov5','MPLCONFIGDIR':'matplotlib',
    'TORCH_HOME':'torch','CUDA_CACHE_PATH':'cuda-cache','TMPDIR':'tmp'}.items():
    p = state/directory
    p.mkdir(parents=True,exist_ok=True)
    p.chmod(0o700)
    env[key] = str(p)
paths = ['scripts/model_convert.py','tests/model_conversion_tests.py','tests/conversion_inference_tests.cpp',
         'scripts/pt_worker.py','src/ui/mainwindow.cpp','src/ui/mainwindow.h',
         'src/ui/modelconversionpage.cpp','src/ui/modelconversionpage.h',
         'build-2.1.0/bin/vision-studio','build-2.1.0/libvision-core.a',
         'build-2.1.0/vision-conversion-inference-tests']
def files():
    return {p:{'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest(),'bytes':(root/p).stat().st_size} for p in paths}
record = {'version':'2.1.0','stage':'Full YOLOv8/v5 input640 ONNX CPU/CUDA and TorchScript CUDA actual conversion regressions',
          'started_at':datetime.datetime.now().astimezone().isoformat(),
          'command':['ctest','--test-dir',str(root/'build-2.1.0'),'-V','-j','1','--no-tests=error','-R','^vision-conversion-inference-tests$'],
          'environment':{k:env[k] for k in env if k.startswith(('VISION_STUDIO_', 'XDG_', 'YOLO','QT_')) or k in
                         ['HOME','TMPDIR','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE']},
          'files_before':files()}
t = time.monotonic()
with log.open('w') as stream:
    p = subprocess.run(record['command'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
record.update({'exit_code':p.returncode,'elapsed_seconds':round(time.monotonic()-t,3),
               'finished_at':datetime.datetime.now().astimezone().isoformat(),'files_after':files(),
               'log':{'file':str(log.relative_to(root)),'bytes':log.stat().st_size,'sha256':hashlib.sha256(log.read_bytes()).hexdigest()}})
record['unchanged'] = record['files_before'] == record['files_after']
record['success'] = p.returncode == 0 and record['unchanged']
(evidence/'ctest-full-model-conversion.log.execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'success':record['success'],'exit_code':p.returncode,'elapsed_seconds':record['elapsed_seconds'],
                  'unchanged':record['unchanged'],'log':str(log)},ensure_ascii=False))
raise SystemExit(0 if record['success'] else 1)
