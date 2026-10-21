#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Preview first: remove only the reviewed 2.4.0 source/build/stage copies.

Release directories are EXCLUDED; use scripts/github_release.py prune separately.
No network, installation, application tests, SDK mutation or historical tag changes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/misaka/.codex/worktrees/ef0b/yolo')
PRIMARY = Path('/home/misaka/文档/ChatGPT/yolo')
HELPER = ROOT / 'docs/releases/evidence/2.5.0/cleanup_verified_copies.py'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('historical_cleanup_utilities', HELPER)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
# Only the pure filesystem/hash/manifest helpers are reused. In particular the
# old PID exceptions, active_references(), proof functions and main() are NOT used.
require, digest, read = u.require, u.digest, u.read
ALLOWED = {'source': 'output/v2.4.0/clean-source', 'build': 'output/build/2.4.0-release',
           'stage': 'output/deb-stage/2.4.0-1'}
USER_FILES = ('docs/足球模型微调.md', 'scripts/evaluate_ball.py',
              'scripts/finetune_ball.py', 'scripts/run_finetune_ball.sh')


def identities():
    raw = subprocess.check_output(['git', '-C', str(PRIMARY), 'status', '--porcelain=v1', '-z', '--untracked-files=all'])
    untracked = {line[3:].decode('utf-8') for line in raw.split(b'\0') if line.startswith(b'?? ')}
    require(untracked == set(USER_FILES), 'Primary untracked files changed: stop and review; never delete or reset them')
    result = {}
    for name in USER_FILES:
        p = PRIMARY / name
        require(p.is_file() and not p.is_symlink(), 'Protected user file is not an ordinary file: ' + str(p))
        result[name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    return result


def assets(project, version):
    return u.local_assets(project, version)


def receipts(paths, threshold):
    proof = {}
    for version, path in paths.items():
        value = read(path)
        require(value.get('success') is True and value.get('repository') == 'misaka-ning/vision-studio'
                and value.get('tag') == 'v' + version and value.get('prerelease') is False
                and not value.get('draft', False), 'Not an official verified formal release: ' + version)
        require(value.get('url') == 'https://github.com/misaka-ning/vision-studio/releases/tag/v' + version,
                'Unexpected release URL')
        actual = u.asset_map(value.get('verified_assets', []))
        require(len(actual) == 3 and actual == assets(ROOT, version), 'Remote asset bytes/SHA256 differ: ' + version)
        require(actual == assets(PRIMARY, version), 'Primary retained release copy is missing/different: ' + version)
        require(0 <= time.time() - path.stat().st_mtime < 86400 and path.stat().st_mtime >= threshold - 2,
                'Run official remote verify again after this preview: ' + version)
        tag = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', f'v{version}^{{commit}}'], text=True).strip()
        require(tag == value.get('commit'), 'Receipt and immutable tag commit differ: ' + version)
        proof[version] = {'receipt': str(path), 'sha256': digest(path), 'commit': tag, 'assets': actual}
    expected_old = read(ROOT / 'docs/releases/evidence/2.4.0/final-assets.json')['assets']
    require(proof['2.4.0']['assets'] == expected_old, 'Historical 2.4.0 published assets changed')
    return proof


def source_proof(path, rows):
    freeze = read(ROOT / 'docs/releases/evidence/2.4.0/source-freeze.json')
    require(freeze['version'] == '2.4.0' and freeze['clean_export'] == str(path), 'Wrong historical source freeze')
    actual = {row['path']: row for row in rows if row['type'] != 'directory'}
    expected = {}
    process = subprocess.Popen(['git', '-C', str(ROOT), 'archive', freeze['source_commit']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for item in archive:
                if item.isdir(): continue
                require(item.isfile(), 'Historical source archive has non-regular files')
                data = archive.extractfile(item).read()
                expected[item.name] = (item.size, hashlib.sha256(data).hexdigest())
        require(process.wait(timeout=30) == 0, 'Cannot read historical Git source archive')
    finally:
        if process.poll() is None: process.kill(); process.wait()
    require(set(actual) == set(expected) | {'runtime-info.json'}, 'Source copy contains missing or unique extra files')
    for name, (size, sha) in expected.items():
        require(actual[name]['type'] == 'file' and actual[name]['size'] == size and actual[name]['sha256'] == sha,
                'Source differs from historical Git snapshot: ' + name)
    require(actual['runtime-info.json']['sha256'] == freeze['runtime_info_sha256'], 'Source runtime provenance differs')
    tree = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', freeze['source_commit'] + '^{tree}'], text=True).strip()
    require(tree == freeze['source_tree'], 'Historical Git source tree differs')
    return {'source_commit': freeze['source_commit'], 'source_tree': tree, 'git_files_verified': len(expected)}


def build_proof(path, rows):
    old = read(ROOT / 'docs/releases/evidence/2.4.0/production-inputs-after-fixture.json')
    require(digest(path / 'bin/vision-studio') == old['production_binary_sha256'], 'Historical application binary differs')
    require(digest(path / 'CMakeCache.txt') == old['cmake_cache_sha256'], 'Historical CMake cache differs')
    for name, row in old['current_test_binary_sha256'].items():
        require(digest(path / name) == row['sha256'], 'Historical test binary differs: ' + name)
    cache = (path / 'CMakeCache.txt').read_text()
    require('CMAKE_PROJECT_VERSION:STATIC=2.4.0' in cache and 'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(ROOT) in cache,
            'Build is not the exact old CMake project')
    allowed_top = {'.ninja_deps', '.ninja_log', '.qt', 'CMakeCache.txt', 'CMakeFiles', 'CTestTestfile.cmake',
                   'DartConfiguration.tcl', 'Testing', 'bin', 'build.ninja', 'cmake_install.cmake', 'libvision-core.a'}
    for row in rows:
        top = row['path'].split('/')[0]
        require(top in allowed_top or (top.startswith('vision-') and (top.endswith('-tests') or top.endswith('_autogen'))),
                'Unclassified build member: ' + row['path'])
        require(row['type'] != 'symlink', 'Unexpected historical build symlink; review manually')
        require(Path(row['path']).suffix.lower() not in {'.pt', '.onnx', '.png', '.jpg', '.mp4', '.deb', '.xz', '.pem', '.key'},
                'Possible unique user material in build: ' + row['path'])
    return {'production_binary_sha256': old['production_binary_sha256'],
            'cmake_cache_sha256': old['cmake_cache_sha256'], 'historical_test_binaries_verified': len(old['current_test_binary_sha256'])}


def stage_proof(path, rows):
    deb = ROOT / 'output/releases/2.4.0/vision-studio_2.4.0-1_amd64.deb'
    actual = {row['path']: row for row in rows}
    expected = {'DEBIAN'}
    require(actual['DEBIAN']['type'] == 'directory' and actual['DEBIAN']['mode'] == 0o755, 'Unexpected stage control parent')
    for flag, prefix in (('--fsys-tarfile', ''), ('--ctrl-tarfile', 'DEBIAN/')):
        process = subprocess.Popen(['dpkg-deb', flag, str(deb)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    name = member.name.removeprefix('./').rstrip('/')
                    if not name or name == '.': continue
                    key = prefix + name; expected.add(key)
                    require(key in actual, 'Stage lacks DEB member: ' + key)
                    row = actual[key]
                    if member.isdir(): require(row['type'] == 'directory', 'Stage directory differs: ' + key)
                    elif member.issym(): require(row['type'] == 'symlink' and row['target'] == member.linkname, 'Stage symlink differs: ' + key)
                    elif member.islnk():
                        target = prefix + member.linkname.removeprefix('./')
                        require(row['type'] == 'file' and target in actual and row['sha256'] == actual[target]['sha256'], 'Stage hardlink differs: ' + key)
                    else:
                        require(member.isfile() and row['type'] == 'file', 'Special stage member: ' + key)
                        stream = archive.extractfile(member); h = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
                        require(row['size'] == member.size and row['sha256'] == h.hexdigest(), 'Stage payload differs: ' + key)
                    require(row['mode'] == stat.S_IMODE(member.mode), 'Stage mode differs: ' + key)
            require(process.wait(timeout=30) == 0, 'Cannot stream historical DEB')
        finally:
            if process.poll() is None: process.kill(); process.wait()
    require(set(actual) == expected, 'Stage contains unbacked extra files')
    return {'deb_sha256': digest(deb), 'payload_and_control_members_verified': len(expected)}


def process_identity(pid):
    p = Path('/proc') / str(pid)
    status = dict(line.split(':', 1) for line in (p / 'status').read_text().splitlines() if ':' in line)
    raw = (p / 'stat').read_text(); fields = raw[raw.rfind(')') + 1:].split()
    return {'pid': pid, 'comm': (p / 'comm').read_text().strip(), 'starttime_ticks': int(fields[19]),
            'uid': list(map(int, status['Uid'].split())), 'gid': list(map(int, status['Gid'].split())),
            'cmdline_sha256': hashlib.sha256((p / 'cmdline').read_bytes()).hexdigest()}


def process_guard(paths, review_path):
    reviewed = {}
    if review_path:
        audit = read(review_path)
        require(audit.get('review_kind') == 'manual-readonly-identity-review' and 0 <= time.time() - audit['created_at_epoch'] < 86400,
                'Opaque process identity review is missing/stale')
        reviewed = {row['pid']: row for row in audit['identities']}
        for pid, expected in reviewed.items(): require(process_identity(pid) == expected, 'Opaque process identity changed')
    references, unreadable, exceptions = [], [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid != os.getuid(): continue
            identity = process_identity(int(p.name))
            denied = []
            try: links = [p / 'cwd', p / 'exe', *(p / 'fd').iterdir()]
            except PermissionError: denied.append('fd'); links = [p / 'cwd', p / 'exe']
            for link in links:
                try:
                    target = os.readlink(link).removesuffix(' (deleted)')
                    if any(target == str(x) or target.startswith(str(x) + '/') for x in paths): references.append({'pid': int(p.name), 'link': str(link), 'target': target})
                except PermissionError: denied.append(link.name)
                except FileNotFoundError: pass
            try:
                for line in (p / 'maps').read_text().splitlines():
                    if any(str(x) + '/' in line for x in paths): references.append({'pid': int(p.name), 'mapped_candidate': True})
            except PermissionError: denied.append('maps')
            if denied:
                if reviewed.get(int(p.name)) == identity: exceptions.append({'identity': identity, 'denied_components': denied})
                else: unreadable.append({'identity': identity, 'denied_components': denied})
        except FileNotFoundError: continue
        except PermissionError: unreadable.append({'pid': int(p.name), 'denied_components': ['identity']})
    require(not references and not unreadable, 'Active paths or unaudited unreadable same-user processes: ' + json.dumps(references + unreadable))
    return {'active_references': references, 'unreviewed_unreadable_processes': unreadable, 'exact_fresh_identity_exceptions': exceptions}


def preserve(path, rows, destination):
    preserved = []
    for row in rows:
        name = row['path']
        if row['type'] != 'file' or not (name == 'runtime-info.json' or name.startswith('Testing/') or
            name.endswith(('.log', '.txt', '.ninja')) or name in ('.ninja_log', '.ninja_deps')): continue
        require(row['size'] <= 16 * 1024 * 1024, 'Large unique build log requires separate review: ' + name)
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): require(digest(target) == row['sha256'], 'Preservation destination differs')
        else: shutil.copyfile(path / name, target)
        require(digest(target) == row['sha256'], 'Preservation copy failed')
        preserved.append({'path': str(target), 'sha256': row['sha256']})
    return preserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--verify-260', type=Path, required=True)
    parser.add_argument('--verify-250', type=Path, required=True)
    parser.add_argument('--verify-240', type=Path, required=True)
    parser.add_argument('--opaque-process-audit', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'output/v2.6.0'; out.mkdir(parents=True, exist_ok=True)
    for name in (args.plan, args.report):
        require(name.absolute().is_relative_to(out) and name.absolute().resolve() == name.absolute(), 'Reports must be inside exact current-version output')
    require(not args.report.exists(), 'Never overwrite prior cleanup report')
    report = {'schema_version': 1, 'version': '2.6.0', 'old_version': '2.4.0', 'created_at_epoch': time.time(),
              'mode': 'execute' if args.execute else 'preview', 'success': False, 'errors': [], 'deletions_performed': [],
              'release_roots_excluded': True, 'retained_versions': ['2.6.0', '2.5.0'], 'not_application_qa': True}
    try:
        require(ROOT.resolve() == ROOT and Path(__file__).resolve() == ROOT / 'docs/releases/evidence/2.6.0/cleanup_verified_240_copies.py', 'Wrong project/tool identity')
        allowed = {label: u.exact_path(ROOT, relative) for label, relative in ALLOWED.items()}
        old = read(args.plan) if args.execute else None
        if old:
            require(old['success'] and old['mode'] == 'preview' and not old['errors'] and not old['deletions_performed'], 'A completed preview is required')
            require(old['script_sha256'] == digest(Path(__file__)) and old['helper_sha256'] == digest(HELPER), 'Reviewed tool changed')
        report['remote_proof'] = receipts({'2.6.0': args.verify_260, '2.5.0': args.verify_250, '2.4.0': args.verify_240}, old['created_at_epoch'] if old else 0)
        if old:
            for version, row in report['remote_proof'].items():
                require(row['assets'] == old['remote_proof'][version]['assets'] and row['commit'] == old['remote_proof'][version]['commit'], 'Release identity changed since preview: ' + version)
        report['protected_user_files'] = identities()
        if old: require(report['protected_user_files'] == old['protected_user_files'], 'Protected user files changed')
        report['candidates'] = {}; proofs = {'source': source_proof, 'build': build_proof, 'stage': stage_proof}
        for label, path in allowed.items():
            rows = u.manifest(path); identity = u.root_identity(path)
            for row in rows:
                require((path / row['path']).lstat().st_dev == identity['device'], 'Candidate crosses a filesystem')
            for line in Path('/proc/self/mountinfo').read_text().splitlines():
                mount = line.split()[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
                require(not (mount == str(path) or mount.startswith(str(path) + '/')), 'Candidate contains a mount point')
            if old:
                require(identity == old['candidates'][label]['identity'] and rows == u.load_manifest(old['candidates'][label]['manifest']), 'Candidate changed after preview')
                for row in old['candidates'][label]['preserved']:
                    require(digest(Path(row['path'])) == row['sha256'], 'Preserved unique log/provenance changed')
                item = old['candidates'][label]
            else:
                proof = proofs[label](path, rows)
                item = {'path': str(path), 'identity': identity, 'local_proof': proof,
                        'manifest': u.preserve_manifest(out / f'cleanup-240-{label}-members.jsonl.gz', rows),
                        'preserved': preserve(path, rows, out / 'cleanup-240-unique-local-only' / label)}
            report['candidates'][label] = item
        report['process_guard'] = process_guard(list(allowed.values()), args.opaque_process_audit)
        report['script_sha256'] = digest(Path(__file__)); report['helper_sha256'] = digest(HELPER)
        if args.execute:
            require(getattr(shutil.rmtree, 'avoids_symlink_attacks', False), 'Descriptor-safe rmtree is required')
            require(identities() == report['protected_user_files'], 'User files changed immediately before deletion')
            for label, path in allowed.items():
                require(u.root_identity(path) == old['candidates'][label]['identity'] and u.manifest(path) == u.load_manifest(old['candidates'][label]['manifest']), 'Candidate changed at final deletion guard')
            for path in allowed.values():
                shutil.rmtree(path); require(not path.exists(), 'Candidate still exists'); report['deletions_performed'].append(str(path))
            require(identities() == report['protected_user_files'], 'Protected user file verification failed after cleanup')
        report['success'] = True
    except Exception as error: report['errors'].append(str(error))
    u.write_new(args.report, report)
    print(json.dumps({'mode': report['mode'], 'success': report['success'], 'deleted': report['deletions_performed'], 'errors': report['errors'], 'report': str(args.report)}, ensure_ascii=False))
    return 0 if report['success'] else 1

if __name__ == '__main__': raise SystemExit(main())
