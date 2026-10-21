#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""V2 preview first; exact authenticated process exceptions and three proven old copies.

Release directories are deliberately excluded and must use github_release.py prune.
This script performs no network operations, installs, builds or runtime mutations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time

NEW = '2.5.0'
OLD = '2.3.1'
OFFICIAL = 'misaka-ning/vision-studio'
PROJECT = Path('/home/misaka/.codex/worktrees/ef0b/yolo')
ALLOWED_RELATIVE = {'source': 'output/v2.3.1/clean-source', 'build': 'output/build/2.3.1-release',
                    'stage': 'output/deb-stage/2.3.1-1'}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Not an ordinary evidence file: {path}')
    result = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(result, dict), f'Expected evidence object: {path}')
    return result

def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite evidence: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def root_identity(path):
    st = path.lstat()
    require(stat.S_ISDIR(st.st_mode) and not path.is_symlink(), f'Not an ordinary root directory: {path}')
    return {'device': st.st_dev, 'inode': st.st_ino, 'mode': stat.S_IMODE(st.st_mode)}

def exact_path(root, relative):
    candidate = root / relative
    require(candidate.is_absolute() and candidate.resolve() == candidate, f'Symlink/resolution mismatch: {candidate}')
    require(candidate not in (root, root.parent) and candidate.is_relative_to(root / 'output'), 'Unsafe candidate root')
    for parent in [candidate, *candidate.parents]:
        require(not parent.is_symlink(), f'Symlink in candidate ancestry: {parent}')
        if parent == root:
            break
    return candidate

def manifest(path):
    rows = []
    def visit(directory):
        with os.scandir(directory) as stream:
            entries = sorted(stream, key=lambda entry: entry.name)
        for entry in entries:
            p = Path(entry.path)
            require(entry.name != '.git', f'Embedded Git metadata: {p}')
            st = entry.stat(follow_symlinks=False)
            row = {'path': p.relative_to(path).as_posix(), 'mode': stat.S_IMODE(st.st_mode)}
            if stat.S_ISLNK(st.st_mode):
                row.update(type='symlink', target=os.readlink(p))
            elif stat.S_ISDIR(st.st_mode):
                row['type'] = 'directory'
            elif stat.S_ISREG(st.st_mode):
                row.update(type='file', size=st.st_size, sha256=digest(p))
            else:
                raise RuntimeError(f'Special/unclassified filesystem entry: {p}')
            rows.append(row)
            if row['type'] == 'directory':
                visit(p)
    visit(path)
    return sorted(rows, key=lambda row: row['path'])

def preserve_manifest(path, rows):
    raw = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in rows).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buffer, mtime=0) as stream:
        stream.write(raw)
    data = buffer.getvalue()
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == data, f'Manifest changed: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(path), 'members': len(rows),
            'logical_file_bytes': sum(row.get('size', 0) for row in rows)}

def load_manifest(row):
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Preserved manifest changed')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == row['members'] and len({item['path'] for item in rows}) == len(rows), 'Invalid preserved manifest')
    return rows

def preserve_unique(root, notes, directory):
    saved = []
    markers = ('/workspace/datasets/', '/workspace/script/model-evaluation', '/weights/', 'Authorization:',
               'BEGIN PRIVATE KEY', 'ghp_', 'github_pat_')
    for candidate in notes['candidates']:
        if not candidate.get('exists'):
            continue
        if candidate['kind'] == 'cmake_build':
            rows = candidate['proof']['unique_logs_need_preservation']
        elif candidate['kind'] == 'immutable_release_copy':
            rows = candidate.get('unarchived_metadata_need_preservation', [])
        else:
            continue
        for row in rows:
            source = Path(row['source'])
            require(source.is_file() and not source.is_symlink() and source.stat().st_size == row['bytes'] and digest(source) == row['sha256'], f'Unique source changed: {source}')
            relative = source.relative_to(Path(candidate['path']))
            destination = directory / candidate['label'] / relative
            require(destination.is_relative_to(directory) and not destination.is_symlink(), 'Unsafe preservation destination')
            text = source.read_text(encoding='utf-8', errors='replace')
            require(not any(marker in text for marker in markers), 'Potential private data in unique log; retain locally and review before public evidence')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                require(destination.is_file() and digest(destination) == row['sha256'], 'Existing unique copy differs')
            else:
                shutil.copyfile(source, destination)
            require(destination.stat().st_size == row['bytes'] and digest(destination) == row['sha256'], 'Preserved unique copy failed verification')
            saved.append({'source': str(source), 'preserved_copy': str(destination), 'bytes': row['bytes'],
                          'sha256': row['sha256'], 'private_or_secret_marker_scan_passed': True})
    return saved

def check_preserved(rows):
    for row in rows:
        p = Path(row['preserved_copy'])
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], 'A unique retained log/metadata is missing or changed')

def git_source_proof(root, path, rows, candidate):
    proof = candidate['git_and_corresponding_source_proof']
    require(proof['success'] and candidate['application_archive_proof']['success'], 'Read-only source proof was not successful')
    actual = {row['path']: row for row in rows if row['type'] != 'directory'}
    expected = {}
    process = subprocess.Popen(['git', '-C', str(root), 'archive', proof['source_commit']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if member.isdir():
                    continue
                require(member.isfile(), 'Unexpected non-file in historical Git export')
                data = archive.extractfile(member).read()
                expected[member.name] = (len(data), hashlib.sha256(data).hexdigest())
        error = process.stderr.read().decode(errors='replace')
        require(process.wait(timeout=30) == 0, 'Historical Git archive failed: ' + error)
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
    require(set(actual) == set(expected) | {'runtime-info.json'}, 'Frozen source has missing or unique extra files')
    for name, (size, value) in expected.items():
        require(actual[name]['type'] == 'file' and actual[name]['size'] == size and actual[name]['sha256'] == value, 'Frozen source differs from immutable Git: ' + name)
    require(actual['runtime-info.json']['sha256'] == proof['runtime_info_sha256'], 'Unique frozen runtime provenance differs')
    tree = subprocess.check_output(['git', '-C', str(root), 'rev-parse', proof['source_commit'] + '^{tree}'], text=True).strip()
    require(tree == proof['source_commit_tree'] == proof['archive_input_tree'], 'Historical source trees differ')
    return {'success': True, 'source_commit': proof['source_commit'], 'source_tree': tree,
            'git_files_verified': len(expected), 'runtime_info_sha256': proof['runtime_info_sha256']}

def build_proof(path, rows, candidate):
    expected = candidate['proof']; actual = {row['path']: row for row in rows}
    require(expected['binary_and_cache_proof_success'] and not expected['unclassified_files'], 'Build was not previously classified')
    cache = (path / 'CMakeCache.txt').read_text()
    for token in ('CMAKE_PROJECT_VERSION:STATIC=2.3.1', 'CMAKE_BUILD_TYPE:STRING=Release',
                  'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(PROJECT), 'VISION_RESOURCE_FALLBACK:STRING=/opt/VisionStudio'):
        require(token in cache, 'Historical CMake configuration differs: ' + token)
    require(actual['bin/vision-studio']['sha256'] == expected['application_sha256'], 'Historical application binary changed')
    for test in expected['test_binaries']:
        require(actual[test['name']]['sha256'] == test['sha256'], 'Historical test binary changed: ' + test['name'])
    library = expected['generated_static_library_proof']
    require(actual[library['path']]['sha256'] == library['sha256'], 'Generated static library changed')
    for obj in library['member_byte_checks']:
        key = Path(obj['object_path']).relative_to(path).as_posix()
        require(actual[key]['sha256'] == obj['sha256'] and actual[key]['size'] == obj['bytes'], 'Generated archive object changed')
    return {'success': True, 'application_sha256': expected['application_sha256'],
            'native_tests_verified': len(expected['test_binaries']), 'static_library_sha256': library['sha256']}

def stage_proof(root, path, rows, candidate):
    deb = root / 'output/releases/2.3.1/vision-studio_2.3.1-1_amd64.deb'
    require(deb.is_file() and digest(deb) == candidate['proof']['deb_sha256'], 'Historical immutable DEB is absent/changed before preservation')
    actual = {row['path']: row for row in rows}; expected = {'DEBIAN'}
    require(actual['DEBIAN']['type'] == 'directory' and actual['DEBIAN']['mode'] == 0o755, 'Unexpected generated control parent')
    for flag, prefix in (('--fsys-tarfile', ''), ('--ctrl-tarfile', 'DEBIAN/')):
        process = subprocess.Popen(['dpkg-deb', flag, str(deb)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    name = member.name.removeprefix('./').rstrip('/')
                    if not name or name == '.':
                        continue
                    key = prefix + name; expected.add(key)
                    require(key in actual, 'Stage missing DEB member: ' + key)
                    row = actual[key]
                    if member.isdir():
                        require(row['type'] == 'directory', 'Stage member kind differs: ' + key)
                    elif member.issym():
                        require(row['type'] == 'symlink' and row['target'] == member.linkname, 'Stage symlink differs: ' + key)
                    elif member.islnk():
                        target = prefix + member.linkname.removeprefix('./')
                        require(row['type'] == 'file' and target in actual and row['sha256'] == actual[target]['sha256'], 'Stage hardlink differs: ' + key)
                    else:
                        require(member.isfile() and row['type'] == 'file', 'Unexpected stage member type: ' + key)
                        h = hashlib.sha256(); stream = archive.extractfile(member)
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block)
                        require(row['size'] == member.size and row['sha256'] == h.hexdigest(), 'Stage file differs from DEB: ' + key)
                    require(row['mode'] == stat.S_IMODE(member.mode), 'Stage mode differs from DEB: ' + key)
            error = process.stderr.read().decode(errors='replace')
            require(process.wait(timeout=30) == 0, 'Cannot stream historical DEB: ' + error)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    require(set(actual) == expected, 'Stage contains extra members')
    return {'success': True, 'deb_sha256': candidate['proof']['deb_sha256'], 'all_payload_and_control_members_verified': len(expected)}

def asset_map(value):
    if isinstance(value, dict):
        return {name: {'bytes': row['bytes'], 'sha256': row['sha256']} for name, row in value.items()}
    return {row['name']: {'bytes': row['bytes'], 'sha256': row['sha256']} for row in value}

def local_assets(root, version):
    base = root / 'output/releases' / version
    names = (f'vision-studio_{version}-1_amd64.deb', f'vision-studio-{version}-complete-source.tar.xz', 'SHA256SUMS')
    result = {}
    for name in names:
        p = base / name
        require(p.is_file() and not p.is_symlink(), 'Missing final local asset: ' + str(p))
        result[name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    return result

def remote_receipts(root, plan, new_path, old_path, strict):
    result = {'success': False, 'fresh_remote_verification_required': True, 'receipts': [], 'pending': []}
    for version, path, expected in ((NEW, new_path, plan['new_assets']), (OLD, old_path, plan['old_assets'])):
        if not path.is_file():
            result['pending'].append('Missing independently generated official verify receipt: ' + str(path)); continue
        value = read(path)
        require(value.get('success') is True and value.get('repository') == OFFICIAL and value.get('tag') == 'v' + version and
                value.get('prerelease') is False and not value.get('draft', False), 'Remote receipt is not the expected published formal release')
        require(value.get('url') == f'https://github.com/{OFFICIAL}/releases/tag/v{version}', 'Remote release URL differs')
        require(len(value.get('verified_assets', [])) == 3 and asset_map(value['verified_assets']) == expected, 'Remote asset names/sizes/SHA256 differ')
        if path.stat().st_mtime < plan['created_at_epoch'] - 2 or time.time() - path.stat().st_mtime > 86400:
            result['pending'].append('Remote receipt is stale; run official verify after this preview: ' + str(path))
            continue
        tag = subprocess.check_output(['git', '-C', str(root), 'rev-parse', f'v{version}^{{commit}}'], text=True).strip()
        require(tag == value.get('commit'), 'Receipt commit differs from immutable local tag')
        result['receipts'].append({'version': version, 'path': str(path), 'sha256': digest(path), 'commit': tag,
                                   'assets': value['verified_assets'], 'mtime_epoch': path.stat().st_mtime})
    result['success'] = not result['pending'] and len(result['receipts']) == 2
    if strict:
        require(result['success'], '; '.join(result['pending']) or 'Remote receipts incomplete')
    return result

# These are the two actually inspected non-dumpable authentication identities.
# Never match by name alone, never read process environments or agent sockets,
# and never permit any other unreadable process through this exception.
OPAQUE_AUTHENTICATION_IDENTITIES = {
    1757: {'pid': 1757, 'comm': '(sd-pam)', 'starttime_ticks': 3985,
           'uid': [1000, 1000, 1000, 1000], 'gid': [1000, 1000, 1000, 1000],
           'cmdline_sha256': '971490059d839d27af3ded30a476216b92689d837b0236a700723fb13640e370'},
    521154: {'pid': 521154, 'comm': 'ssh-agent', 'starttime_ticks': 32536773,
             'uid': [1000, 1000, 1000, 1000], 'gid': [1000, 1000, 112, 1000],
             'cmdline_sha256': 'bda5e58c66fe7e615a147888edce38a29deb873439b821f8bc2e5e48875151bb'},
}


def process_identity(pid):
    process = Path('/proc') / str(pid)
    status = {}
    for line in (process / 'status').read_text().splitlines():
        if ':' in line:
            name, value = line.split(':', 1)
            status[name] = value.strip()
    raw_stat = (process / 'stat').read_text()
    close = raw_stat.rfind(')')
    require(close >= 0, 'Invalid process stat record')
    fields = raw_stat[close + 1:].split()
    require(len(fields) >= 20, 'Truncated process stat record')
    return {'pid': pid, 'comm': (process / 'comm').read_text().strip(),
            'starttime_ticks': int(fields[19]),
            'uid': [int(value) for value in status['Uid'].split()],
            'gid': [int(value) for value in status['Gid'].split()],
            'cmdline_sha256': hashlib.sha256((process / 'cmdline').read_bytes()).hexdigest()}


def active_references(paths):
    # Observe both reviewed host processes before checking. A restricted /proc
    # namespace must never produce a misleading successful preview.
    for pid, expected in OPAQUE_AUTHENTICATION_IDENTITIES.items():
        require(process_identity(pid) == expected, 'Reviewed authentication identity missing or changed; use the host namespace and re-audit PID ' + str(pid))
    found = []; unreadable = []; exceptions = {}
    def inaccessible(process, component):
        pid = int(process.name)
        expected = OPAQUE_AUTHENTICATION_IDENTITIES.get(pid)
        if expected is not None and process_identity(pid) == expected:
            if pid not in exceptions:
                exceptions[pid] = {**expected, 'inaccessible_components': [],
                    'identity_explanation': 'Previously inspected non-dumpable system authentication process; this exact PID, starttime, comm, full UID/GID and command-line SHA256 are pinned. No environment or agent socket is read.'}
            exceptions[pid]['inaccessible_components'].append(component)
        else:
            unreadable.append('/proc/' + process.name + '/' + component)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            links = [('cwd', process / 'cwd'), ('exe', process / 'exe')]
            try:
                links.extend(('fd', entry) for entry in (process / 'fd').iterdir())
            except FileNotFoundError:
                continue
            except PermissionError:
                inaccessible(process, 'fd')
            for component, link in links:
                try:
                    target = os.readlink(link).removesuffix(' (deleted)')
                except FileNotFoundError:
                    continue
                except PermissionError:
                    inaccessible(process, component); continue
                if any(target == str(path) or target.startswith(str(path) + '/') for path in paths):
                    found.append({'pid': int(process.name), 'link': str(link), 'target': target})
        except FileNotFoundError:
            continue
    require(not found and not unreadable, 'Active old paths or unreadable non-exempt same-user references: ' + json.dumps(found + unreadable))
    return {'same_user_active_references': found, 'unreadable_nonexempt_same_user_references': unreadable,
            'reviewed_host_authentication_identities_observed': True,
            'narrow_opaque_authentication_process_exceptions': [exceptions[pid] for pid in sorted(exceptions)],
            'exception_boundary': 'Only the two exact identity records above; any PID reuse, command/credential/starttime change or other unreadable process fails closed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--preview-plan', type=Path)
    parser.add_argument('--new-verify', type=Path)
    parser.add_argument('--old-verify', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--execute', action='store_true', help='Only after explicit approval and fresh official new/old verify receipts')
    args = parser.parse_args(); root = args.project.absolute()
    require(root == PROJECT and root.resolve() == root, 'This audit is fixed to the exact reviewed worktree')
    out = root / 'output/v2.5.0'; out.mkdir(parents=True, exist_ok=True)
    plan_path = args.preview_plan or out / 'cleanup-old-231-v2-preview.json'
    destination = args.report or (out / 'cleanup-old-231-v2-executed.json' if args.execute else plan_path)
    require(not destination.exists(), 'Refusing to overwrite a previous report')
    release = root / 'output/releases/2.5.0'
    new_verify = args.new_verify or release / 'release-verify.json'
    old_verify = args.old_verify or release / 'previous-2.3.1-verify.json'
    report = {'schema_version': 1, 'version': NEW, 'old_version': OLD, 'success': False, 'errors': [],
              'mode': 'execute' if args.execute else 'preview', 'deletions_performed': [],
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'created_at_epoch': time.time()}
    try:
        allowed = {label: exact_path(root, relative) for label, relative in ALLOWED_RELATIVE.items()}
        if args.execute:
            plan = read(plan_path)
            require(plan['success'] and plan['mode'] == 'preview' and not plan['errors'] and not plan['deletions_performed'], 'Preview has not passed')
            require(plan['reviewed_script_sha256'] == digest(Path(__file__)), 'Cleanup script changed since preview')
            require(plan['new_assets'] == local_assets(root, NEW), 'Final new assets changed after preview')
            check_preserved(plan['preserved_backups'])
            report['remote_verification'] = remote_receipts(root, plan, new_verify, old_verify, True)
            checked = []
            for label, path in allowed.items():
                old = plan['candidates'][label]
                require(old['path'] == str(path) and root_identity(path) == old['root_identity'], 'Candidate identity/path changed')
                expected = load_manifest(old['manifest']); actual = manifest(path)
                require(actual == expected, 'Candidate gained/changed members since preview: ' + label)
                checked.append({'label': label, 'path': str(path), 'members_rechecked': len(actual),
                                'logical_bytes': sum(row.get('size', 0) for row in actual), 'manifest_sha256': old['manifest']['sha256']})
            report['process_guard'] = active_references(list(allowed.values()))
            require(getattr(shutil.rmtree, 'avoids_symlink_attacks', False), 'A symlink-safe rmtree implementation is required')
            for label, path in allowed.items():
                require(root_identity(path) == plan['candidates'][label]['root_identity'], 'Root changed immediately before deletion')
                shutil.rmtree(path)
                require(not path.exists(), 'Old generated directory still exists after deletion')
                report['deletions_performed'].append(str(path))
            report.update(preview=str(plan_path), preview_sha256=digest(plan_path), verified_candidates=checked,
                          preserved_backups=plan['preserved_backups'], remote_new_and_old_verified=True)
        else:
            notes_path = out / 'cleanup-231-readonly-notes.json'; notes = read(notes_path)
            require(notes['version'] == NEW and notes['old_version'] == OLD and not notes['errors'] and
                    not notes['deletions_performed'] and notes['local_proof_summary']['staging_exactly_matches_immutable_deb'], 'Original read-only audit did not pass')
            report['readonly_audit'] = {'path': str(notes_path), 'sha256': digest(notes_path)}
            report['historical_evidence'] = notes['historical_evidence']
            for row in notes['historical_evidence'].values():
                require(digest(Path(row['path'])) == row['sha256'], 'Historical proof evidence changed')
            report['preserved_backups'] = preserve_unique(root, notes, out / 'cleanup-old-231-unique-logs')
            report['new_assets'] = local_assets(root, NEW)
            report['old_assets'] = asset_map(read(root / 'docs/releases/evidence/2.3.1/final-assets.json')['assets'])
            require(local_assets(root, OLD) == report['old_assets'], 'Historical local assets differ from published final hashes')
            candidates = {item['path']: item for item in notes['candidates']}
            report['candidates'] = {}
            for label, path in allowed.items():
                require(str(path) in candidates and candidates[str(path)]['exists'], 'Candidate was not part of the reviewed read-only audit')
                identity = root_identity(path); rows = manifest(path)
                item = candidates[str(path)]
                if label == 'source':
                    proof = git_source_proof(root, path, rows, item)
                elif label == 'build':
                    proof = build_proof(path, rows, item)
                else:
                    proof = stage_proof(root, path, rows, item)
                report['candidates'][label] = {'path': str(path), 'root_identity': identity, 'local_proof': proof,
                    'manifest': preserve_manifest(out / f'cleanup-old-231-{label}-members.jsonl.gz', rows)}
            report['process_guard'] = active_references(list(allowed.values()))
            report['reviewed_script_sha256'] = digest(Path(__file__))
            report['protected_paths'] = notes['protected_paths']
            report['protected_version_parent_children'] = notes['version_parent_preserved_children']
            report['retained_versions'] = ['2.5.0', '2.4.0']
            report['remote_verification'] = remote_receipts(root, report, new_verify, old_verify, False)
            report['ready_for_execute'] = report['remote_verification']['success']
            report['official_release_directories_excluded'] = True
        report['success'] = True
    except Exception as error:
        report['errors'].append(str(error))
    report['protected_sdk_runtime_and_user_data'] = True
    report['scope'] = 'Only exact old clean-source, CMake build and DEB stage; no release roots, user data, Git, SDK, runtime or other QA directories.'
    write_new(destination, report)
    print(json.dumps({'success': report['success'], 'mode': report['mode'], 'report': str(destination),
                      'deleted': report['deletions_performed'], 'ready_for_execute': report.get('ready_for_execute'),
                      'errors': report['errors']}, ensure_ascii=False))
    return 0 if report['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
