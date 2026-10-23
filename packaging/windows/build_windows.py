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
import zipfile

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
    files = [{'path':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
             for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'WINDOWS-FILES.json']
    (stage / 'WINDOWS-FILES.json').write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')
    return files


def make_source_snapshot(destination):
    version = re.search(r'set\(VISION_STUDIO_APP_VERSION "([^"]+)"\)',
                        (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')).group(1)
    if (ROOT / '.git').exists():
        tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    else:
        snapshot = json.loads((ROOT / 'WINDOWS-SOURCE-SNAPSHOT.json').read_text(encoding='utf-8'))
        tracked = [entry['path'] for entry in snapshot['files']]
    names = {name for name in tracked if name and (ROOT / name).is_file()}
    for directory in (ROOT / 'packaging/windows', ROOT / 'docs/releases/evidence' / version / 'windows'):
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
        archive.writestr(prefix + 'WINDOWS-SOURCE-SNAPSHOT.json', json.dumps({
            'source_state':'Working-tree snapshot including Windows adaptations',
            'base_commit':base_commit(),
            'files':records}, indent=2)+'\n')
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--qt-prefix', type=Path, required=True)
    parser.add_argument('--opencv-bin', type=Path, required=True)
    parser.add_argument('--msvc-redist', type=Path, required=True)
    parser.add_argument('--iscc', type=Path, required=True)
    parser.add_argument('--ffmpeg-source-dir', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    native_sources = json.loads(Path(__file__).with_name('windows-native-sources.json').read_text(encoding='utf-8'))
    for item in native_sources['ffmpeg']:
        source = args.ffmpeg_source_dir / item['archive']
        if not source.is_file() or digest(source) != item['sha256']:
            raise RuntimeError(f'Missing or incorrect corresponding FFmpeg source: {source}')
    version = re.search(r'set\(VISION_STUDIO_APP_VERSION "([^"]+)"\)',
                        (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')).group(1)
    stage = args.stage.resolve()
    if stage.exists():
        raise SystemExit(f'Stage already exists; choose a new empty directory: {stage}')
    stage.mkdir(parents=True)
    binary = stage / 'bin'
    binary.mkdir()
    shutil.copy2(args.binary, binary / 'vision-studio.exe')
    run([args.qt_prefix / 'bin/windeployqt.exe', '--release', '--no-compiler-runtime',
         '--no-opengl-sw', '--no-system-d3d-compiler',
         '--skip-plugin-types', 'platforminputcontexts,qmltooling,position',
         binary / 'vision-studio.exe'])
    for pattern in ('opencv_world*.dll', 'opencv_videoio_ffmpeg*.dll'):
        files = [p for p in args.opencv_bin.glob(pattern) if not p.stem.endswith('d')]
        if len(files) != 1:
            raise RuntimeError(f'Expected one release DLL for {pattern}: {files}')
        shutil.copy2(files[0], binary)
    for dll in args.msvc_redist.glob('*.dll'):
        shutil.copy2(dll, binary)
    (binary / 'qt.conf').write_text('[Paths]\nPrefix=.\nPlugins=.\n', encoding='utf-8')
    copy_tree(args.runtime, stage / 'runtime')
    # Console-script executables embed the build-machine interpreter path.
    # The application and GPU preparer use python.exe -m <module> exclusively.
    shutil.rmtree(stage / 'runtime/Scripts', ignore_errors=True)
    for folder in ('assets', 'models', 'vendor'):
        copy_tree(ROOT / folder, stage / folder)
    shutil.rmtree(stage / 'models/source', ignore_errors=True)
    scripts = stage / 'scripts'
    scripts.mkdir()
    for name in ('pt_worker.py', 'netron_server.py', 'model_convert.py',
                 'evaluation_data.py', 'evaluation_metrics.py', 'evaluation_worker.py', 'evaluation_reporting.py', 'evaluation_snapshot.py', 'evaluation_import.py',
                 'gpu_probe.py', 'gpu_setup.py', 'gpu_windows.py'):
        source = ROOT / 'scripts' / name
        shutil.copy2(source, scripts)  # Every deployed helper is required; fail rather than silently omit one.
    for name in ('LICENSE', 'README.md', 'CHANGELOG.md', 'requirements-gpu.lock.txt', 'requirements-gpu-windows.lock.txt'):
        if not (ROOT / name).is_file():
            continue
        shutil.copy2(ROOT / name, stage)
    for name in ('BUILD-INPUTS.json', 'gpu-wheel-sources.json', 'upstream-source-reference.json', 'windows-native-sources.json'):
        shutil.copy2(Path(__file__).with_name(name), stage)
    for item in native_sources['ffmpeg']:
        target = stage / 'sources/ffmpeg' / item['archive']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.ffmpeg_source_dir / item['archive'], target)
    copy_tree(ROOT / 'docs', stage / 'docs')
    copy_tree(ROOT / 'packaging/licenses', stage / 'licenses/linux-release-reference')
    native_licenses = stage / 'licenses/windows-opencv'
    native_licenses.mkdir(parents=True)
    for name in ('LICENSE.txt', 'LICENSE_FFMPEG.txt'):
        source = args.opencv_bin.parents[3] / name
        if not source.is_file():
            raise RuntimeError(f'Missing original Windows OpenCV notice: {source}')
        shutil.copy2(source, native_licenses)
    copy_tree(args.opencv_bin.parents[2] / 'etc/licenses', native_licenses / 'third-party')
    # Preserve every original wheel license as shipped, plus a readable inventory.
    inventory_code = r'''
import importlib.metadata as m,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); records=[]
for d in sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()):
 files=[]
 for item in d.files or []:
  p=pathlib.PurePosixPath(str(item)); low=p.name.lower()
  if '..' in p.parts or p.is_absolute(): continue
  if not (low.startswith(('license','licence','copying','notice','thirdpartynotice','third_party_notice')) or 'licenses' in [x.lower() for x in p.parts]): continue
  source=pathlib.Path(d.locate_file(item))
  if source.is_file():
   target=root/'licenses/python'/d.metadata['Name']/pathlib.Path(*p.parts)
   target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target);files.append(str(target.relative_to(root)))
 records.append({'name':d.metadata['Name'],'version':d.version,'license':d.metadata.get('License-Expression',d.metadata.get('License','')),'files':files})
(root/'windows-python-distributions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'requirements-windows.lock.txt').write_text(''.join(d['name']+'=='+d['version']+'\n' for d in records),encoding='utf-8')
'''
    run([stage / 'runtime/python.exe', '-I', '-c', inventory_code, stage])
    marker = {'version': version, 'platform': 'windows-x64', 'qt': '6.8.3',
              'python': 'runtime/python.exe', 'binary': 'bin/vision-studio.exe',
              'default_install_directory': r'C:\Users\<username>\VisionStudio',
              'user_data': '%LOCALAPPDATA%/vision-studio',
              'build_commit': base_commit(),
              'source_state': 'Windows working-tree snapshot; see accompanying Windows source archive'}
    (stage / 'windows-install.json').write_text(json.dumps(marker, indent=2)+'\n', encoding='utf-8')
    make_source_snapshot(stage / 'sources' / f'vision-studio-{version}-windows-sources.zip')
    write_manifest(stage)
    if not args.stage_only:
        args.output.mkdir(parents=True, exist_ok=True)
        run([args.iscc, '/Qp', f'/DAppVersion={version}', f'/DStageDir={stage}',
             f'/DOutputDir={args.output.resolve()}', Path(__file__).with_name('vision-studio.iss')])
        installer = args.output / f'vision-studio-{version}-windows-x64-setup.exe'
        (args.output / 'SHA256SUMS').write_text(digest(installer)+'  '+installer.name+'\n', encoding='ascii')
        print(installer.resolve())


if __name__ == '__main__':
    main()
