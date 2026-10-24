#!/usr/bin/env python3
"""Build a self-contained Windows x64 installer from tested local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]


def run(command, **kwargs):
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def base_commit():
    if (ROOT / '.git').exists():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    snapshot = ROOT / 'WINDOWS-SOURCE-SNAPSHOT.json'
    return json.loads(snapshot.read_text(encoding='utf-8')).get('base_commit', 'source-snapshot') if snapshot.exists() else 'unversioned-source'


def copy_tree(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '*.pdb', '*.exp', '*.lib', '.git', '.cache'))


def write_manifest(stage):
    paths = [p for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'WINDOWS-FILES.json']
    def record(p):
        return {'path':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(record, paths))
    (stage / 'WINDOWS-FILES.json').write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')
    return files


def make_source_snapshot(destination, ffmpeg_source_dir=None):
    version = re.search(r'set\(VISION_STUDIO_APP_VERSION "([^"]+)"\)',
                        (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')).group(1)
    if (ROOT / '.git').exists():
        tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    else:
        snapshot = json.loads((ROOT / 'WINDOWS-SOURCE-SNAPSHOT.json').read_text(encoding='utf-8'))
        tracked = [entry['path'] for entry in snapshot['files']]
    names = {name for name in tracked if name and (ROOT / name).is_file()}
    # Acceptance reports bind this ZIP's checksum and therefore stay outside it.
    names = {name for name in names if not name.startswith(f'docs/releases/evidence/{version}/windows/')}
    for directory in (ROOT / 'packaging/windows',):
        if directory.exists():
            names.update(p.relative_to(ROOT).as_posix() for p in directory.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    names.update({'scripts/gpu_windows.py', 'requirements-gpu-windows.lock.txt'})
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'vision-studio-{version}-windows-sources/'
    records = []
    temporary = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            source = ROOT / name
            archive.write(source, prefix + name)
            records.append({'path':name, 'sha256':digest(source)})
        companion = []
        if ffmpeg_source_dir:
            native = json.loads((ROOT / 'packaging/windows/windows-native-sources.json').read_text(encoding='utf-8'))
            for entry in native['ffmpeg']:
                original = Path(ffmpeg_source_dir) / entry['archive']
                if digest(original) != entry['sha256']:
                    raise RuntimeError('FFmpeg corresponding source checksum mismatch')
                path = 'third-party/ffmpeg/' + entry['archive']
                archive.write(original, prefix + path)
                companion.append({'path': path, 'sha256': entry['sha256'], 'bytes': original.stat().st_size})
        archive.writestr(prefix + 'WINDOWS-SOURCE-SNAPSHOT.json' , json.dumps({
            'source_state':'Working-tree snapshot including Windows adaptations',
            'base_commit':base_commit(),
            'files':records, 'third_party_files':companion}, indent=2)+'\n')
    temporary.replace(destination)
    return destination


PYTHON_INVENTORY = "\nimport importlib.metadata as m,json,pathlib,shutil,sys\nroot=pathlib.Path(sys.argv[1]); records=[]\nfor d in sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()):\n files=[]\n for item in d.files or []:\n  p=pathlib.PurePosixPath(str(item)); low=p.name.lower()\n  if '..' in p.parts or p.is_absolute(): continue\n  if not (low.startswith(('license','licence','copying','notice','thirdpartynotice','third_party_notice')) or 'licenses' in [x.lower() for x in p.parts]): continue\n  source=pathlib.Path(d.locate_file(item))\n  if source.is_file():\n   target=root/'licenses/python'/d.metadata['Name']/pathlib.Path(*p.parts)\n   target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target);files.append(str(target.relative_to(root)))\n records.append({'name':d.metadata['Name'],'version':d.version,'license':d.metadata.get('License-Expression',d.metadata.get('License','')),'files':files})\n(root/'windows-python-distributions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')\n(root/'requirements-windows.lock.txt').write_text(''.join(d['name']+'=='+d['version']+'\\n' for d in records),encoding='utf-8')\n"

RUNTIME_SCRIPTS = ('pt_worker.py', 'netron_server.py', 'model_convert.py', 'gpu_probe.py',
    'gpu_setup.py', 'gpu_windows.py', 'evaluation_worker.py', 'evaluation_data.py',
    'evaluation_metrics.py', 'evaluation_reporting.py', 'evaluation_import.py',
    'evaluation_snapshot.py', 'prepare_cpu_runtime.ps1')
DEV_ROOTS = ('include', 'libs', 'Scripts', 'Lib/test', 'Lib/site-packages/torch/include',
             'Lib/site-packages/numpy/_core/include', 'Lib/site-packages/numpy/core/include')
TEST_PACKAGES = {'numpy', 'pandas', 'onnx', 'matplotlib', 'sympy', 'networkx', 'PIL'}


def runtime_excluded(relative):
    path = Path(relative)
    posix = path.as_posix()
    if any(posix == item or posix.startswith(item + '/') for item in DEV_ROOTS): return True
    if any(part in {'__pycache__', '.git', '.cache'} for part in path.parts): return True
    if path.suffix.lower() in {'.pyc', '.pyo', '.pdb', '.exp', '.lib'}: return True
    if len(path.parts) >= 4 and path.parts[:2] == ('Lib', 'site-packages'):
        if path.parts[2] in TEST_PACKAGES and any(p in {'tests', 'test'} for p in path.parts[3:]): return True
    return False


def prepare_runtime(source, output, version, msvc_redist):
    directory = output / 'components/cpu'
    destination = output / f'vision-studio-{version}-windows-x64-cpu-runtime.zip'
    crt = {p.name: digest(p) for p in sorted(msvc_redist.glob('*.dll'))}
    if not crt: raise RuntimeError('The native VC runtime DLL directory is empty')
    if destination.exists():
        inventory = json.loads((directory / 'CPU-FILES.json').read_text(encoding='utf-8'))
        if inventory['requirements_sha256'] != digest(ROOT / 'packaging/windows/requirements-cpu.lock.txt'):
            raise RuntimeError('Existing CPU runtime has a different dependency lock; choose new output directory')
        if inventory.get('vc_runtime') != crt:
            raise RuntimeError('Existing CPU runtime has different or missing VC DLLs; choose new output directory')
        with zipfile.ZipFile(destination) as archive:
            if archive.read('CPU-FILES.json') != (directory / 'CPU-FILES.json').read_bytes():
                raise RuntimeError('Existing runtime ZIP and component stage differ')
        return directory, destination
    if directory.exists(): raise RuntimeError(f'Use a new component stage: {directory}')
    omitted = []
    def ignore(parent, names):
        result = []
        for name in names:
            item = Path(parent) / name
            if runtime_excluded(item.relative_to(source)):
                result.append(name)
                omitted.append(item.relative_to(source).as_posix())
        return result
    shutil.copytree(source, directory, ignore=ignore)
    # Python children cannot inherit the GUI process's app-local DLL directory.
    # Keep the component independently usable on machines without a VC SDK.
    for dll in sorted(msvc_redist.glob('*.dll')): shutil.copy2(dll, directory/dll.name)
    run([directory / 'python.exe', '-I', '-B', '-c', PYTHON_INVENTORY, directory])
    probe = output/'builder-probe-config'; probe.mkdir(exist_ok=True)
    probe_env = dict(os.environ, YOLO_CONFIG_DIR=str(probe), MPLCONFIGDIR=str(probe), PYTHONDONTWRITEBYTECODE='1')
    run([directory / 'python.exe', '-I', '-B', '-c',
         'import torch,torchvision,onnx,cv2,netron,ultralytics,numpy,PIL,yaml,matplotlib; assert not torch.version.cuda'], env=probe_env)
    records = write_manifest(directory)
    (directory / 'WINDOWS-FILES.json').unlink()
    inventory = {'schema_version': 1, 'requirements_sha256': digest(ROOT / 'packaging/windows/requirements-cpu.lock.txt'),
                 'omitted_development_paths': omitted, 'vc_runtime': crt, 'files': records}
    (directory / 'CPU-FILES.json').write_text(json.dumps(inventory, indent=2)+'\n', encoding='utf-8')
    partial = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob('*')):
            if not path.is_file(): continue
            info = zipfile.ZipInfo(path.relative_to(directory).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open('rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                shutil.copyfileobj(src, dst)
    partial.replace(destination)
    return directory, destination


def source_notes(stage, version):
    (stage / 'SOURCE-DOWNLOAD.txt').write_text(
        f'Vision Studio {version}\nApplication license: AGPL-3.0-only.\n'
        'Application and Windows FFmpeg corresponding sources, licenses, dependency locks and rebuild instructions:\n'
        f'https://github.com/misaka-ning/vision-studio/releases/download/v{version}/vision-studio-{version}-windows-sources.zip\n'
        'Shared upstream corresponding sources and original notices:\n'
        f'https://github.com/misaka-ning/vision-studio/releases/download/v{version}/vision-studio-{version}-complete-source.tar.xz\n'
        'These archives are independent downloads; the installer does not download or install them.\n', encoding='utf-8')


def refresh_checksums(output, version):
    names = [f'vision-studio-{version}-windows-x64-{profile}-setup.exe' for profile in ('offline', 'online')]
    names += [f'vision-studio-{version}-windows-x64-cpu-runtime.zip', f'vision-studio-{version}-windows-sources.zip']
    if all((output/name).is_file() for name in names):
        (output/'SHA256SUMS-windows-x64').write_text(''.join(digest(output/name)+'  '+name+'\n' for name in names), encoding='ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--qt-prefix', type=Path, required=True)
    parser.add_argument('--opencv-bin', type=Path, required=True)
    parser.add_argument('--msvc-redist', type=Path, required=True)
    parser.add_argument('--iscc', type=Path, required=True)
    parser.add_argument('--ffmpeg-source-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('offline', 'online'), required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--skip-source', action='store_true', help='Freeze source ZIP separately after acceptance evidence is committed')
    args = parser.parse_args()
    native = json.loads(Path(__file__).with_name('windows-native-sources.json').read_text(encoding='utf-8'))
    for item in native['ffmpeg']:
        source = args.ffmpeg_source_dir / item['archive']
        if not source.is_file() or digest(source) != item['sha256']:
            raise RuntimeError(f'Missing or incorrect FFmpeg corresponding source: {source}')
    version = re.search(r'set\(VISION_STUDIO_APP_VERSION "([^"]+)"\)', (ROOT/'CMakeLists.txt').read_text(encoding='utf-8')).group(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    cpu, cpu_zip = prepare_runtime(args.runtime.resolve(), output, version, args.msvc_redist)
    stage = args.stage.resolve()
    if stage.exists(): raise RuntimeError(f'Stage already exists; choose new directory: {stage}')
    stage.mkdir(parents=True)
    binary = stage/'bin'; binary.mkdir()
    shutil.copy2(args.binary, binary/'vision-studio.exe')
    run([args.qt_prefix/'bin/windeployqt.exe', '--release', '--no-compiler-runtime', '--no-opengl-sw',
         '--no-system-d3d-compiler', '--skip-plugin-types', 'platforminputcontexts,qmltooling,position', binary/'vision-studio.exe'])
    for pattern in ('opencv_world*.dll', 'opencv_videoio_ffmpeg*.dll'):
        files = [p for p in args.opencv_bin.glob(pattern) if not p.stem.endswith('d')]
        if len(files) != 1: raise RuntimeError(f'Expected one release DLL: {pattern}')
        shutil.copy2(files[0], binary)
    for dll in args.msvc_redist.glob('*.dll'): shutil.copy2(dll, binary)
    (binary/'qt.conf').write_text('[Paths]\nPrefix=.\nPlugins=.\n', encoding='utf-8')
    if args.profile == 'offline':
        def stage_link(source, target):
            try: os.link(source, target)
            except OSError: shutil.copy2(source, target)
            return target
        # Immutable local build stages may share files; installers contain bytes,
        # and no link or source-machine path is distributed to the recipient.
        shutil.copytree(cpu, stage/'runtime', copy_function=stage_link)
    scripts = stage/'scripts'; scripts.mkdir()
    for name in RUNTIME_SCRIPTS: shutil.copy2(ROOT/'scripts'/name, scripts)
    shutil.copy2(Path(__file__).with_name('cleanup_install.ps1'), scripts/'cleanup_install.ps1')
    # Runtime Python modules and architecture YAMLs are required for legacy PT.
    vendor = stage/'vendor/yolov5'
    vendor.mkdir(parents=True)
    for folder in ('models', 'utils', 'classify', 'segment', 'data'):
        shutil.copytree(ROOT/'vendor/yolov5'/folder, vendor/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'images', '*.ipynb', '.github'))
    for name in ('LICENSE', 'requirements.txt', 'export.py', 'val.py'):
        shutil.copy2(ROOT/'vendor/yolov5'/name, vendor/name)
    # All GUI resources are compiled into resources.qrc; no example asset folder.
    for name in ('LICENSE', 'requirements-gpu-windows.lock.txt'):
        shutil.copy2(ROOT/name, stage/name)
    for name in ('gpu-wheel-sources.json', 'windows-native-sources.json'):
        shutil.copy2(Path(__file__).with_name(name), stage/name)
    notices = stage/'licenses'
    copy_tree(ROOT/'packaging/licenses/texts', notices/'texts')
    for folder in ('extracted', 'protocol-notices', 'source-notices'):
        copy_tree(ROOT/'packaging/licenses/qt'/folder, notices/'qt'/folder)
    shutil.copy2(ROOT/'packaging/licenses/qt/ICU-73.2-LICENSE.txt', notices/'qt')
    for folder in ('netron', 'model-conversion', 'fluent-qt-design-reference'):
        copy_tree(ROOT/'packaging/licenses'/folder, notices/folder)
    copy_tree(Path(__file__).with_name('licenses'), notices/'windows-opencv')
    source_notes(stage, version)
    descriptor = {'schema_version': 1, 'version': version, 'platform': 'windows-x64',
                  'file': cpu_zip.name, 'bytes': cpu_zip.stat().st_size, 'sha256': digest(cpu_zip),
                  'uncompressed_bytes': sum(p.stat().st_size for p in cpu.rglob('*') if p.is_file()),
                  'url': f'https://github.com/misaka-ning/vision-studio/releases/download/v{version}/{cpu_zip.name}',
                  'inventory_sha256': digest(cpu/'CPU-FILES.json')}
    (stage/'cpu-runtime.json').write_text(json.dumps(descriptor, indent=2)+'\n', encoding='utf-8')
    marker = {'version': version, 'platform': 'windows-x64', 'profile': args.profile, 'qt': '6.8.3',
              'python': 'runtime/python.exe' if args.profile == 'offline' else 'user-cache',
              'binary': 'bin/vision-studio.exe', 'default_install_directory': r'C:\Users\<username>\VisionStudio',
              'user_data': '%LOCALAPPDATA%/vision-studio', 'build_commit': base_commit(),
              'runtime_sha256': descriptor['sha256'], 'package_revision': 'windows.1'}
    (stage/'windows-install.json').write_text(json.dumps(marker, indent=2)+'\n', encoding='utf-8')
    write_manifest(stage)
    if not args.skip_source: make_source_snapshot(output/f'vision-studio-{version}-windows-sources.zip', args.ffmpeg_source_dir)
    if not args.stage_only:
        run([args.iscc, '/Qp', f'/DAppVersion={version}', f'/DInstallProfile={args.profile}',
             f'/DStageDir={stage}', f'/DOutputDir={output}', Path(__file__).with_name('vision-studio.iss')])
        print(output/f'vision-studio-{version}-windows-x64-{args.profile}-setup.exe')
    refresh_checksums(output, version)


if __name__ == '__main__': main()
