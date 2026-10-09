#!/usr/bin/env python3
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/v2.1.0/portability-fix3-proof'
OUT.mkdir(exist_ok=True)
ART = ROOT / 'output/v2.1.0-conversion/portability-fix3'
PROBE = ROOT / 'output/v2.1.0/converted_gpu_probe_v2'
EVID = ROOT / 'docs/releases/evidence/2.1.0/conversion-portability-proof'
EVID.mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

record = {'schema_version': 1, 'version': '2.1.0', 'success': False,
          'started_at': datetime.datetime.now().astimezone().isoformat(),
          'source_sha256': {p: sha(ROOT/p) for p in
              ['scripts/model_convert.py', 'scripts/pt_worker.py', 'tests/model_conversion_tests.py',
               'tests/conversion_inference_tests.cpp', 'build-2.1.0/bin/vision-studio',
               'build-2.1.0/libvision-core.a']},
          'runs': [], 'comparisons': [], 'errors': []}
env = os.environ.copy()
env.update(json.loads((ROOT/'docs/releases/evidence/2.1.0/ctest-all.log.execution.json').read_text())['environment'])
for family in ['yolov8n', 'yolov5n']:
    events = [json.loads(x) for x in (ART/f'{family}-onnx.jsonl').read_text().splitlines() if x.startswith('{')]
    result = next(x for x in events if x.get('type') == 'result')
    meta = OUT/f'{family}-metadata.json'
    meta.write_text(json.dumps({'conversion':result}, ensure_ascii=False, indent=2)+'\n')
    data = {}
    for kind, path, device in [('onnx-cpu', ART/f'{family}-640.onnx', 'cpu'),
                               ('onnx-cuda', ART/f'{family}-640.onnx', 'cuda'),
                               ('original-pt-cpu', ROOT/f'models/{family}.pt', 'cpu')]:
        name = f'{family}-{kind}'
        command = [str(PROBE), str(path), str(meta), str(ROOT/'assets/bus.jpg'), device]
        runenv = env.copy()
        if device == 'cuda':
            runenv['VISION_STUDIO_ORT_PROFILE_PREFIX'] = str(OUT/f'{name}-nodes')
        started = time.monotonic()
        run = subprocess.run(command, cwd=ROOT, env=runenv, capture_output=True, text=True, timeout=90)
        stdout, stderr = OUT/f'{name}.stdout.log', OUT/f'{name}.stderr.log'
        stdout.write_text(run.stdout)
        stderr.write_text(run.stderr)
        parsed = next((json.loads(x) for x in reversed(run.stdout.splitlines()) if x.startswith('{')), {})
        row = {'name':name, 'command':command, 'exit_code':run.returncode,
               'elapsed_seconds':round(time.monotonic()-started,3), 'model_sha256':sha(path),
               'stdout':str(stdout.relative_to(ROOT)), 'stderr':str(stderr.relative_to(ROOT)), 'result':parsed}
        record['runs'].append(row)
        data[kind] = parsed
        if run.returncode != 0 or not parsed.get('success') or parsed.get('actual_device') != device:
            record['errors'].append(f'{name}: actual inference/device failed')
        if device == 'cuda':
            traces = sorted(OUT.glob(f'{name}-nodes*.json'))
            nodes = []
            for trace in traces:
                contents = json.loads(trace.read_text())
                nodes.extend(e for e in contents if e.get('args',{}).get('provider') == 'CUDAExecutionProvider')
                row.setdefault('cuda_traces', []).append({'file':str(trace.relative_to(ROOT)), 'sha256':sha(trace), 'bytes':trace.stat().st_size})
            row['cuda_node_count'] = len(nodes)
            if not nodes:
                record['errors'].append(f'{name}: no actual CUDAExecutionProvider events')
    reference = data['original-pt-cpu'].get('predictions', [])
    for kind in ['onnx-cpu', 'onnx-cuda']:
        actual = data[kind].get('predictions', [])
        differences = []
        valid = bool(reference) and len(actual) == len(reference)
        for p,q in zip(reference,actual):
            valid &= (p['class_id'],p['label']) == (q['class_id'],q['label'])
            differences.append({'confidence':abs(p['confidence']-q['confidence']),
                                'box':max(abs(p[k]-q[k]) for k in ['x','y','width','height'])})
        confidence = max((x['confidence'] for x in differences), default=float('inf'))
        box = max((x['box'] for x in differences), default=float('inf'))
        valid &= confidence < .002 and box < .05
        comparison = {'family':family, 'export':kind, 'reference':'original-pt-cpu',
                      'same_count_classes_labels':bool(reference) and len(actual)==len(reference) and all(
                          (p['class_id'],p['label'])==(q['class_id'],q['label']) for p,q in zip(reference,actual)),
                      'max_confidence_absolute_error':confidence, 'max_box_absolute_error_pixels':box,
                      'tolerances':{'confidence':.002,'box_pixels':.05}, 'passed':bool(valid)}
        record['comparisons'].append(comparison)
        if not valid:
            record['errors'].append(f'{family}-{kind}: original PT prediction comparison failed')
record['success'] = not record['errors']
record['finished_at'] = datetime.datetime.now().astimezone().isoformat()
report = OUT/'portability-onnx-full-model-qa.json'
report.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
# Keep all real raw logs and traces with the successful or failed report.
import shutil
for path in OUT.iterdir():
    if path.is_file():
        shutil.copy2(path,EVID/path.name)
print(json.dumps({'success':record['success'],'errors':record['errors'],'comparisons':record['comparisons'],
                  'report':str(report),'runs':[(r['name'],r['exit_code'],len(r['result'].get('predictions',[])),r.get('cuda_node_count')) for r in record['runs']]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if record['success'] else 1)
