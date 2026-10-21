#!/usr/bin/env python3
"""Retain small public final-package QA files; perform no inference."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from PIL import Image

ROOT = Path.cwd()
RELEASE = ROOT / 'output/releases/2.5.0'
VERSION = '2.5.0'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def bundle(name, entries, metadata):
    entries = sorted(set((str(arc), Path(path)) for arc, path in entries))
    require(len({arc for arc, _ in entries}) == len(entries), 'Duplicate archive members')
    archive = RELEASE / (name + '-artifacts.zip')
    require(not archive.exists(), 'Refusing to overwrite retained artifact archive')
    files, pngs = [], []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for arc, path in entries:
            require(path.is_file() and not path.is_symlink(), 'Selected artifact must be an ordinary file')
            require(path.stat().st_size < 32 * 1024 * 1024, 'Unexpectedly large raw evidence file')
            require('..' not in Path(arc).parts and not Path(arc).is_absolute(), 'Unsafe archive path')
            value = sha(path)
            if path.suffix.lower() == '.png':
                with Image.open(path) as image:
                    image.load()
                    pngs.append({'file': arc, 'width': image.width, 'height': image.height, 'mode': image.mode})
            out.write(path, arc)
            files.append({'file': arc, 'bytes': path.stat().st_size, 'sha256': value})
    with zipfile.ZipFile(archive) as verify:
        require(verify.testzip() is None, 'ZIP CRC failure')
        require(verify.namelist() == [row['file'] for row in files], 'ZIP member list changed')
        for row in files:
            data = verify.read(row['file'])
            require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'ZIP member SHA failure')
            if row['file'].endswith('.png'):
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
    report = {'schema_version': 1, 'version': VERSION, 'success': True, 'errors': [], **metadata,
              'archive': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive)},
              'files': files, 'file_count': len(files), 'uncompressed_bytes': sum(row['bytes'] for row in files),
              'png_fully_decoded_count': len(pngs), 'png_complete_decode': pngs,
              'zip_crc_and_member_sha256_verified': True}
    (RELEASE / (name + '-artifacts.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'name': name, 'files': len(files), 'pngs': len(pngs), 'archive_bytes': archive.stat().st_size}, ensure_ascii=False))

matrix_path = RELEASE / 'packaged-evaluation-qa.json'
matrix = read(matrix_path)
require(matrix['success'] and matrix['eligible_for_release'] and matrix['version'] == VERSION, 'Final matrix did not pass')
work = Path(matrix['workdir'])
entries = []
for case in matrix['cases']:
    state = work / case
    for p in [state / 'request.json', state / 'application.stdout.log', state / 'application.stderr.log']:
        entries.append((str(p.relative_to(work)), p))
    for folder in [state / 'data/evaluations', state / 'ui-smoke']:
        for p in folder.rglob('*'):
            if p.is_file() and 'plot-config' not in p.parts:
                entries.append((str(p.relative_to(work)), p))
    for p in state.glob('ort-evaluation*.json'):
        entries.append((str(p.relative_to(work)), p))
for p in (work / 'torchscript-export').glob('conversion.*.log'):
    entries.append((str(p.relative_to(work)), p))
for key, path in [('dataset/provenance.json', Path('/tmp/vision-studio-evaluation-public-230/provenance.json')),
                  ('dataset/coco8-qa.yaml', Path('/tmp/vision-studio-evaluation-public-230/coco8-qa.yaml'))]:
    entries.append((key, path))
bundle('evaluation', entries, {
    'scope': 'Small original raw final-DEB public COCO8 evaluation artifacts; six CPU/CUDA cases and three same-weight comparisons. Archiving performs no inference.',
    'deb_sha256': matrix['deb_sha256'], 'packaged_binary_sha256': matrix['binary_sha256'],
    'original_matrix_report': str(matrix_path.relative_to(ROOT)), 'original_matrix_report_sha256': sha(matrix_path),
    'source_workdir': str(work), 'dataset_provenance': matrix['dataset']['provenance'],
    'dataset_note': 'Four public COCO8 Val images verify pipeline integration, not football accuracy or deployment/generalization quality. No private dataset is included.',
    'case_names': sorted(matrix['cases']), 'comparisons': matrix['comparisons'],
    'actual_cuda_profiles': matrix['cases']['onnx_cuda']['cuda_profiles'],
    'torchscript_export_result': matrix['torchscript_export'],
    'excluded': ['Extracted application/runtime/source tree', 'Models, including exported TorchScript', 'Runtime/cache directories', 'Private user inputs and reports'],
    'limitations': ['This is preservation of existing final-package raw evidence, not another executed test suite.', 'Timing and tiny-set metrics are not advertised performance or model-quality guarantees.'],
    'visual_inspection': [{'file': 'onnx_cuda/ui-smoke/evaluation-analysis.png', 'result': 'Reviewed actual 2.5.0 CUDA analysis page: full-width metrics and visible threshold curves; lower analysis content remains available through its scrollable viewport, without overlapping controls.'},
                          {'file': 'pt_cpu/ui-smoke/evaluation-preview.png', 'result': 'Reviewed actual PT CPU preview: source image, GT and predicted box, target list and compact top controls visible.'}]
})

folder_path = RELEASE / 'folder-package-qa.json'
folder = read(folder_path)
require(folder['success'] and folder['version'] == VERSION, 'Folder package regression did not pass')
work = Path(folder['workdir'])
entries = []
for p in [work / 'application/request.json', work / 'application/application.stdout.log', work / 'application/application.stderr.log']:
    entries.append((str(p.relative_to(work)), p))
for directory in ['application/data/evaluations', 'application/ui-smoke']:
    for p in (work / directory).rglob('*'):
        if p.is_file() and 'plot-config' not in p.parts:
            entries.append((str(p.relative_to(work)), p))
for relative in ['migrated-dataset/data.yaml', 'migrated-dataset/lists/val.txt', 'migrated-dataset/train_from_scratch.example.yaml']:
    entries.append((relative, work / relative))
bundle('folder', entries, {
    'scope': 'Small original public folder-migration ONNX CPU regression evidence. Four copied public COCO8 Val images/labels were byte-verified by the original check; source JPEGs are not duplicated here.',
    'deb_sha256': folder['deb_sha256'], 'source_workdir': str(work),
    'source_report': {'file': str(folder_path.relative_to(ROOT)), 'sha256': sha(folder_path)},
    'excluded': ['Extracted application/runtime', 'Public source JPEG copies', 'Models', 'Cache folders', 'Private inputs'],
    'limitations': ['Original 10-check regression, not an additional suite or football accuracy benchmark.']
})

namespace_path = RELEASE / 'install-qa.json'
namespace = read(namespace_path)
require(namespace['success'] and namespace['details']['application_version'] == VERSION, 'Namespace install QA did not pass')
work = Path(namespace['workdir'])
entries = [(str(p.relative_to(work)), p) for p in sorted((work / 'logs').glob('*.log'))]
for directory in ['onnx', 'pt', 'model_display', 'model_display_pt', 'conversion_onnx', 'conversion_torchscript']:
    for p in (work / 'user' / directory).iterdir():
        if p.is_file() and p.suffix.lower() in {'.json', '.png', '.csv'}:
            entries.append((str(p.relative_to(work)), p))
bundle('namespace', entries, {
    'scope': 'Small original isolated dpkg install/run/purge evidence: raw logs, reports and PNGs only. No new inference is performed by archiving.',
    'source_workdir': str(work), 'source_report': {'file': str(namespace_path.relative_to(ROOT)), 'sha256': sha(namespace_path)},
    'deb_sha256': namespace['details']['deb_sha256'],
    'data_provenance': 'Public packaged bus sample used by explicit internal smoke checks; no private user images, models, settings, history or datasets.',
    'excluded': ['Readonly /opt/VisionStudio runtime/source/model tree', 'usr-bin/system bindings', 'Exported ONNX/TorchScript models', 'User config/cache/history'],
    'limitations': ['Only proves private namespace installation, execution and purge. It does not assert host installation or model quality.'],
    'visual_inspection': [{'file': 'user/onnx/workbench.png', 'result': 'Actual isolated CPU smoke workbench inspected; image, five detections and readable UI visible.'},
                          {'file': 'user/model_display/model-hierarchy.png', 'result': 'Actual isolated native model hierarchy inspected; module names and details are readable in the dark theme.'}]
})
