#!/usr/bin/env python3
"""Exact, two-phase cleanup of four audited Vision Studio 2.0 build/source directories.

Default: preview only. --prepare re-hashes the original source archive and DEB,
then freezes all directory/file identities in --plan. --execute requires that
plan, successful 2.2 release verification and verified 2.0 remote prune evidence.
It preserves 48 unique prepublication files before any descriptor-based deletion.
It never deletes the old release directory, its parents, or any other QA tree.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

OLD = Path('/home/misaka/文档/ChatGPT/yolo')
TARGETS = (
    OLD / 'output/deb-stage/2.0.0-1',
    OLD / 'output/v2.0-stable/source-inputs',
    OLD / 'output/v2.0-stable/review/delivery-build',
    OLD / 'output/v2.0-stable/package-source-inputs',
)
PRESERVE = OLD / 'output/legacy-qa-preserved/2.0.0-prepublication'
PROTECTED = {'.git', '.agents', '.codex', '.aws', '.ssh'}
FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest_stream(stream):
    h = hashlib.sha256()
    while data := stream.read(4 * 1024 * 1024):
        h.update(data)
    return h.hexdigest()


def digest(path):
    with os.fdopen(os.open(path, FLAGS), 'rb') as f:
        return digest_stream(f)


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()


def load(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), f'Missing/unsafe JSON: {path}')
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f'JSON must be an object: {path}')
    return value


def no_symlink_components(path):
    require(path.is_absolute(), f'Absolute path required: {path}')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f'Symlink path component: {current}')


def identity(info):
    return {'dev': info.st_dev, 'ino': info.st_ino, 'mode': stat.S_IMODE(info.st_mode),
            'mtime_ns': info.st_mtime_ns}


def scan(root):
    no_symlink_components(root)
    root_info = root.lstat()
    require(stat.S_ISDIR(root_info.st_mode), f'Not a directory: {root}')
    files, directories = {}, {'': identity(root_info)}
    for base, subdirs, names in os.walk(root, followlinks=False):
        for name in list(subdirs) + names:
            require(name not in PROTECTED, f'Protected path: {Path(base) / name}')
            path = Path(base) / name
            key = path.relative_to(root).as_posix()
            info = path.lstat()
            require(info.st_dev == root_info.st_dev, f'Cannot cross filesystem: {path}')
            row = identity(info)
            row.update({'blocks': info.st_blocks, 'nlink': info.st_nlink})
            if stat.S_ISDIR(info.st_mode):
                directories[key] = row
            elif stat.S_ISLNK(info.st_mode):
                row.update({'kind': 'symlink', 'target': os.readlink(path)})
                files[key] = row
                if name in subdirs:
                    subdirs.remove(name)
            else:
                require(stat.S_ISREG(info.st_mode), f'Special file: {path}')
                row.update({'kind': 'file', 'bytes': info.st_size, 'sha256': digest(path)})
                require(identity(path.lstat()) == identity(info), f'File changed while hashing: {path}')
                files[key] = row
    # Root identity needs the same bookkeeping as child directories.
    directories[''].update({'blocks': root_info.st_blocks, 'nlink': root_info.st_nlink})
    return {'root_identity': identity(root_info), 'files': files, 'directories': directories}


def content(rows):
    return {k: {n: v[n] for n in ('kind', 'sha256', 'bytes', 'target') if n in v}
            for k, v in rows.items()}


def tar_rows(path=None, stream=None, strip=False):
    rows, prefix = {}, None
    with tarfile.open(path, fileobj=stream, mode='r|xz' if path else 'r|') as archive:
        for member in archive:
            name = member.name.removeprefix('./')
            require(not name.startswith('/') and '..' not in Path(name).parts, 'Unsafe tar member')
            if strip:
                if prefix is None:
                    prefix = name.split('/')[0]
                if name == prefix:
                    continue
                require(name.startswith(prefix + '/'), 'Unexpected tar root')
                name = name[len(prefix) + 1:]
            if member.isfile():
                rows[name] = {'kind': 'file', 'sha256': digest_stream(archive.extractfile(member)), 'bytes': member.size}
            elif member.issym():
                rows[name] = {'kind': 'symlink', 'target': member.linkname}
            elif member.islnk():
                rows[name] = {'kind': 'hardlink', 'target': member.linkname.removeprefix('./')}
            else:
                require(member.isdir(), 'Special tar member')
    for key, row in rows.items():
        if row['kind'] == 'hardlink':
            target = row['target']
            if strip and target.startswith(prefix + '/'):
                target = target[len(prefix) + 1:]
            require(target in rows and rows[target]['kind'] == 'file', 'Unsupported tar hardlink')
            rows[key] = rows[target].copy()
    return rows


def prepare(a):
    audit = load(a.audit)
    require(audit.get('release_version') == '2.0.0', 'Wrong audit version')
    for path in TARGETS:
        require(path.exists(), f'Prepare must precede release prune: {path}')
    snapshots = {str(path): scan(path) for path in TARGETS}
    archive = Path(audit['source_archive']['path'])
    require(digest(archive) == audit['source_archive']['sha256'], 'Old source archive changed')
    archived = tar_rows(path=archive, strip=True)
    require(len(archived) == 3229, 'Expected exactly 3229 original source members')
    src = content(snapshots[str(TARGETS[1])]['files'])
    extra = 'scripts/__pycache__/release_metadata.cpython-314.pyc'
    proof = audit['directories']['source-inputs']['reproducible_generated_extras'][0]
    require(set(src) == set(archived) | {extra}, 'Unexpected source-inputs paths')
    require(all(src[k] == v for k, v in archived.items()), 'source-inputs differs from original archive')
    require(src[extra]['sha256'] == proof['cached_sha256'] and proof['byte_for_byte_reconstructed'] is True,
            'Extra Python cache differs from proved reproducible cache')
    require(src['scripts/release_metadata.py']['sha256'] == proof['source_sha256'], 'Cache source differs')
    pkg = content(snapshots[str(TARGETS[3])]['files'])
    unique = []
    for entry in audit['directories']['package-source-inputs']['unresolved_local_content']:
        key = entry['path']
        require('..' not in Path(key).parts and not Path(key).is_absolute(), 'Unsafe unique path')
        expected = entry.get('actual', entry)
        expected = {n: expected[n] for n in ('kind', 'sha256', 'bytes')}
        require(pkg.get(key) == expected, f'Unique file changed: {key}')
        unique.append({'path': key, **expected})
    require(len(unique) == 48 and len({row['path'] for row in unique}) == 48, 'Expected exactly 48 unique files')
    same = {k: v for k, v in pkg.items() if k not in {row['path'] for row in unique}}
    require(len(same) == 3179 and all(archived.get(k) == v for k, v in same.items()), '3179 common package source files differ')
    stage_audit = audit['directories']['deb-stage/2.0.0-1']
    deb = Path(stage_audit['deb_path'])
    require(digest(deb) == stage_audit['deb_sha256'], 'Original DEB changed or already pruned')
    stage = content(snapshots[str(TARGETS[0])]['files'])
    for option, expected in (
        ('--fsys-tarfile', {k: v for k, v in stage.items() if not k.startswith('DEBIAN/')}),
        ('--ctrl-tarfile', {k[7:]: v for k, v in stage.items() if k.startswith('DEBIAN/')}),
    ):
        proc = subprocess.Popen(['dpkg-deb', option, str(deb)], stdout=subprocess.PIPE)
        reference = tar_rows(stream=proc.stdout)
        require(proc.wait() == 0 and reference == expected, f'DEB stage mismatch: {option}')
    require(re.search(r'^Version:\s*2\.0\.0-1\s*$', (TARGETS[0] / 'DEBIAN/control').read_text(), re.M), 'Wrong DEB stage Control version')
    build = audit['directories']['review/delivery-build']
    actual_build = snapshots[str(TARGETS[2])]['files']
    require(build['pure_generated_build_recreatable_from_archived_sources'] is True, 'Build not proved regenerable')
    require(set(build['manifest']) == set(actual_build), 'Old build acquired extra/missing files')
    require(all(actual_build[k]['sha256'] == v['sha256'] for k, v in build['manifest'].items()), 'Old build changed')
    require('CMAKE_PROJECT_VERSION:STATIC=2.0.0\n' in (TARGETS[2] / 'CMakeCache.txt').read_text(), 'Wrong build version')
    old_receipt = load(OLD / 'output/releases/2.0.0/publication-receipt.json')
    require(old_receipt.get('success') is True and old_receipt.get('tag') == 'v2.0.0', 'Wrong original receipt')
    return {'schema_version': 1, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'mode': 'exact-old-2.0-code-and-build-cleanup', 'audit': audit,
            'audit_sha256': digest(a.audit), 'audit_path': str(Path(a.audit).absolute()),
            'targets': snapshots, 'unique_files': unique, 'preservation_directory': str(PRESERVE),
            'original_release_assets': old_receipt['verified_assets'],
            'verification': {'source_members_matching': 3229, 'package_common_matching': 3179,
                             'unique_preserved_required': 48, 'deb_stage_payload_and_control_exact': True}}


def assets(rows, version):
    require(isinstance(rows, list), 'Asset list must be an array')
    expected = {f'vision-studio_{version}-1_amd64.deb', f'vision-studio-{version}-complete-source.tar.xz', 'SHA256SUMS'}
    require({x.get('name') for x in rows} == expected and len(rows) == 3, 'Exactly three release assets required')
    for row in rows:
        require(type(row.get('bytes')) is int and row['bytes'] > 0 and
                re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) and
                type(row.get('asset_id')) is int and row['asset_id'] > 0 and
                bool(row.get('verification')), 'Invalid verified asset data')
    return {x['name']: (x['bytes'], x['sha256']) for x in rows}


def gates(a, plan):
    verified, prune = load(a.release_verify), load(a.prune_receipt)
    require(verified.get('success') is True and verified.get('repository') == 'misaka-ning/vision-studio'
            and verified.get('tag') == 'v2.2.0' and verified.get('prerelease') is False
            and re.fullmatch('[0-9a-f]{40}', verified.get('commit', ''))
            and verified.get('url') == 'https://github.com/misaka-ning/vision-studio/releases/tag/v2.2.0',
            'Successful formal 2.2.0 verify JSON required')
    new_assets = assets(verified.get('verified_assets'), '2.2.0')
    require(prune.get('success') is True and prune.get('root') == str(OLD / 'output/releases'), 'Wrong prune receipt root/status')
    require({Path(x).name for x in prune.get('kept', [])} == {'2.2.0', '2.1.0'}, 'Prune must retain latest two formal releases')
    candidates = [x for x in prune.get('verified_candidates', []) if x.get('path') == str(OLD / 'output/releases/2.0.0')]
    require(len(candidates) == 1 and candidates[0].get('url') == 'https://github.com/misaka-ning/vision-studio/releases/tag/v2.0.0'
            and re.fullmatch('[0-9a-f]{40}', candidates[0].get('commit', '')), 'Verified 2.0.0 prune candidate required')
    require(assets(candidates[0].get('assets'), '2.0.0') == assets(plan['original_release_assets'], '2.0.0'), 'Old remote assets differ from prepared originals')
    kept = [x for x in prune.get('verified_kept', []) if Path(x.get('path', '')).name == '2.2.0']
    require(len(kept) == 1 and kept[0].get('commit') == verified['commit'] and
            assets(kept[0].get('assets'), '2.2.0') == new_assets, 'Retained 2.2.0 assets/commit mismatch')
    return {'release_verify': str(a.release_verify), 'release_verify_sha256': digest(a.release_verify),
            'prune_receipt': str(a.prune_receipt), 'prune_receipt_sha256': digest(a.prune_receipt)}


def stable(snapshot):
    # Link counts may drop when release prune removes other hard links; contents/inodes must stay fixed.
    def clean(rows):
        return {k: {n: v for n, v in row.items() if n not in ('nlink', 'blocks')}
                for k, row in rows.items()}
    return {'root_identity': snapshot['root_identity'], 'files': clean(snapshot['files']),
            'directories': clean(snapshot['directories'])}


def copy_unique(plan):
    no_symlink_components(PRESERVE)
    require(not PRESERVE.exists(), f'Preservation directory exists; inspect it manually before retry: {PRESERVE}')
    PRESERVE.mkdir(parents=True)
    for row in plan['unique_files']:
        source, target = TARGETS[3] / row['path'], PRESERVE / 'files' / row['path']
        require(digest(source) == row['sha256'], 'Unique source changed before copy')
        target.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(source, FLAGS), 'rb') as src, target.open('xb') as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        require(digest(target) == row['sha256'] and target.stat().st_size == row['bytes'], 'Unique backup hash failed')
    (PRESERVE / 'original-audit.json').write_bytes(encode(plan['audit']))
    (PRESERVE / 'prepared-cleanup-plan.json.gz').write_bytes(gzip.compress(encode(plan), mtime=0))
    preserved = {'schema_version': 1, 'success': True, 'source': str(TARGETS[3]), 'files': plan['unique_files'],
                 'count': 48, 'logical_bytes': sum(x['bytes'] for x in plan['unique_files'])}
    (PRESERVE / 'preservation-manifest.json').write_bytes(encode(preserved))
    require(load(PRESERVE / 'original-audit.json') == plan['audit'], 'Audit backup differs')
    require(digest(PRESERVE / 'original-audit.json') == plan['audit_sha256'], 'Original audit bytes differ')
    require(json.loads(gzip.decompress((PRESERVE / 'prepared-cleanup-plan.json.gz').read_bytes())) == plan, 'Prepared manifest backup differs')
    for row in plan['unique_files']:
        require(digest(PRESERVE / 'files' / row['path']) == row['sha256'], 'Final preservation check failed')
    return preserved


def same_inode(info, expected):
    return info.st_dev == expected['dev'] and info.st_ino == expected['ino']


def erase_exact(root, snapshot):
    # Descriptor-pinned recursion: never resolve a symlink or glob and never traverse an unlisted entry.
    children = {key: set() for key in snapshot['directories']}
    for key in list(snapshot['files']) + [k for k in snapshot['directories'] if k]:
        parent = str(Path(key).parent)
        children['' if parent == '.' else parent].add(Path(key).name)
    def descend(fd, prefix):
        entries = set(os.listdir(fd))
        require(entries == children[prefix], f'Unexpected entry during deletion: {root / prefix}')
        for name in sorted(entries):
            key = (Path(prefix) / name).as_posix()
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if key in snapshot['directories']:
                row = snapshot['directories'][key]
                require(same_inode(info, row) and stat.S_ISDIR(info.st_mode), 'Directory identity changed')
                child = os.open(name, FLAGS | os.O_DIRECTORY, dir_fd=fd)
                try:
                    require(same_inode(os.fstat(child), row), 'Directory swapped while opening')
                    descend(child, key)
                finally:
                    os.close(child)
                require(same_inode(os.stat(name, dir_fd=fd, follow_symlinks=False), row), 'Directory swapped before rmdir')
                os.rmdir(name, dir_fd=fd)
            else:
                row = snapshot['files'][key]
                require(identity(info) == {k: row[k] for k in ('dev', 'ino', 'mode', 'mtime_ns')},
                        'File identity changed before unlink')
                require((row['kind'] == 'symlink' and stat.S_ISLNK(info.st_mode)) or
                        (row['kind'] == 'file' and stat.S_ISREG(info.st_mode)), 'File type changed')
                require(row['kind'] != 'file' or info.st_size == row['bytes'], 'File size changed before unlink')
                require(row['kind'] != 'symlink' or os.readlink(name, dir_fd=fd) == row['target'],
                        'Symlink target changed before unlink')
                os.unlink(name, dir_fd=fd)
    parent = os.open(root.parent, FLAGS | os.O_DIRECTORY)
    try:
        fd = os.open(root.name, FLAGS | os.O_DIRECTORY, dir_fd=parent)
        try:
            require(same_inode(os.fstat(fd), snapshot['root_identity']), 'Root directory identity changed')
            descend(fd, '')
        finally:
            os.close(fd)
        require(same_inode(os.stat(root.name, dir_fd=parent, follow_symlinks=False), snapshot['root_identity']), 'Root swapped before rmdir')
        os.rmdir(root.name, dir_fd=parent)
    finally:
        os.close(parent)


def summary(plan):
    return {'success': True, 'dry_run': True, 'targets': [
        {'path': k, 'files_and_links': len(v['files']), 'logical_bytes': sum(x.get('bytes', 0) for x in v['files'].values()),
         'root_identity': v['root_identity']} for k, v in plan['targets'].items()],
        'unique_files_required': len(plan['unique_files']), 'preserve_to': str(PRESERVE),
        'verification': plan['verification']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--audit', type=Path, default=Path('/tmp/vision220-old-code-audit.json'))
    parser.add_argument('--plan', type=Path, default=Path('/tmp/vision220-old-code-cleanup-plan.json'))
    parser.add_argument('--release-verify', type=Path, required=True)
    parser.add_argument('--prune-receipt', type=Path, required=True)
    a = parser.parse_args()
    require(not a.execute or a.plan.exists(), '--execute requires an existing --prepare plan')
    if a.prepare or not a.plan.exists():
        plan = prepare(a)
        if a.prepare:
            a.plan.write_bytes(encode(plan))
    else:
        plan = load(a.plan)
    require(plan.get('schema_version') == 1 and plan.get('mode') == 'exact-old-2.0-code-and-build-cleanup'
            and set(plan.get('targets', {})) == {str(p) for p in TARGETS}
            and plan.get('preservation_directory') == str(PRESERVE), 'Wrong cleanup plan')
    gate = gates(a, plan)
    if not a.execute:
        result = summary(plan)
        result['remote_receipts'] = gate
        result['plan'] = str(a.plan) if a.plan.exists() else None
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    # Rehash all four exact targets before writing preservation or removing anything.
    current = {str(path): scan(path) for path in TARGETS}
    require(all(stable(current[k]) == stable(v) for k, v in plan['targets'].items()), 'Prepared tree changed; stop and prepare anew')
    preserved = copy_unique(plan)
    before = os.statvfs(OLD)
    result = summary(plan)
    result.update({'success': False, 'dry_run': False, 'remote_receipts': gate,
                   'preserved_unique': preserved, 'removed': [], 'in_progress': None})
    report = PRESERVE / 'cleanup-result.json'
    report.write_bytes(encode(result))
    unique_inodes = {}
    for tree in current.values():
        for is_dir, rows in ((False, tree['files']), (True, tree['directories'])):
            for row in rows.values():
                key = (row['dev'], row['ino'])
                value = unique_inodes.setdefault(key, {'inside_links': 0, 'links': row['nlink'], 'blocks': row['blocks'], 'dir': is_dir})
                value['inside_links'] += 1
    estimate = sum(v['blocks'] * 512 for v in unique_inodes.values() if v['dir'] or v['inside_links'] >= v['links'])
    try:
        for root in TARGETS:
            result['in_progress'] = str(root)
            report.write_bytes(encode(result))
            erase_exact(root, plan['targets'][str(root)])
            result['removed'].append(str(root))
            result['in_progress'] = None
            report.write_bytes(encode(result))
        after = os.statvfs(OLD)
        result.update({'success': True, 'estimated_released_inode_blocks_bytes': estimate,
                       'observed_filesystem_free_bytes_change': (after.f_bfree - before.f_bfree) * after.f_frsize,
                       'space_note': 'Inode-block estimate accounts for external hard links; filesystem free-space change may include concurrent activity and shared extents.',
                       'finished_at_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        report.write_bytes(encode(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'success': False, 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
