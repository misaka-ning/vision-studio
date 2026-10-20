#!/usr/bin/env python3
"""Read-only final-package dataset precheck; private manifests remain in /tmp."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def snapshot(root):
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()}

def aggregate(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--deb', type=Path, required=True)
    parser.add_argument('--matrix-report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve() / 'packaging'))
    import verify_evaluation_deb as qa
    report = {'schema_version': 1, 'version': '2.4.0', 'success': False, 'errors': [],
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Actual final extracted helper read-only private dataset precheck, no inference',
              'dataset_writes': False, 'metadata_selected': False, 'model_inference': False,
              'privacy_note': 'Private images, labels, filenames, source paths and complete manifests retained locally only; public evidence contains counts and hashes.'}
    started = time.monotonic()
    workspace = Path(tempfile.mkdtemp(prefix='vision240-private-precheck-'))
    try:
        matrix = qa.read_json(args.matrix_report.resolve())
        require(matrix.get('success') is True and matrix.get('eligible_for_release') is True, 'Final package matrix must pass first')
        require(matrix.get('version') == '2.4.0', 'Package matrix belongs to another version')
        app = Path(matrix['workdir']) / 'extracted/opt/VisionStudio'
        binding = qa.bind_package(args.deb.resolve(), app)
        require(binding['version'] == '2.4.0' and binding['deb_sha256'] == matrix['deb_sha256'], 'Final DEB binding differs')
        qa.require_readonly(app)
        report.update(deb_sha256=binding['deb_sha256'], package_bound=True, read_only_application=True)
        source = args.source.resolve()
        helpers = ('evaluation_worker.py', 'evaluation_data.py', 'evaluation_metrics.py')
        match = {name: digest(source / 'scripts' / name) == digest(app / 'scripts' / name) for name in helpers}
        report['source_helpers_match_staged'] = match
        require(all(match.values()), 'Frozen source and final package helper bytes differ')
        dataset = args.dataset.resolve()
        require(dataset.is_dir(), 'Private dataset directory is missing')
        before = snapshot(dataset)
        report['input_files'] = len(before)
        report['input_files_aggregate_sha256'] = aggregate(before)
        request = workspace / 'request.json'
        qa.atomic_json(request, {'dataset_yaml': str(dataset), 'split': 'val'})
        manifests = {}
        for name, python, helper in (('source', Path('/opt/VisionStudio/runtime/bin/python'), source / 'scripts/evaluation_worker.py'),
                                     ('packaged', app / 'runtime/bin/python', app / 'scripts/evaluation_worker.py')):
            state = workspace / name
            state.mkdir()
            output = state / 'manifest.json'
            env = qa.isolated_environment(state, app, python, None, None)
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [str(python), str(helper), 'prepare', '--request', str(request), '--output', str(output)]
            with (state / 'helper.stdout.log').open('wb') as out, (state / 'helper.stderr.log').open('wb') as err:
                code = subprocess.run(command, cwd=state, env=env, stdout=out, stderr=err, timeout=300).returncode
            require(code == 0, f'{name} private helper failed; private diagnostic logs retained locally')
            manifest = qa.read_json(output)
            require(manifest.get('split') == 'val' and not manifest.get('requires_attention'), f'{name} private dataset precheck requires attention')
            manifests[name] = manifest
        original, packaged = manifests['source'], manifests['packaged']
        report['manifest_matches_source_precheck'] = original == packaged
        require(original == packaged, 'Source and package prechecks returned different manifests')
        report.update(selected_images=packaged['selected_images'], available_images=packaged['available_images'],
                      class_count=len(packaged['names']), validation=packaged['validation'],
                      root_relocated=packaged.get('root_relocated') is True,
                      manifest_fingerprint=packaged['fingerprint'], requires_attention=packaged['requires_attention'])
        require(report['selected_images'] > 0 and report['root_relocated'], 'Private directory precheck was empty or relocation was not exercised')
        after = snapshot(dataset)
        report['input_files_unchanged'] = before == after
        yaml_path = Path(packaged['data_yaml'])
        report['original_yaml_sha256_unchanged'] = digest(yaml_path) == packaged['source_hashes'][str(yaml_path)]
        report['yaml_sha256'] = digest(yaml_path)
        require(before == after and report['original_yaml_sha256_unchanged'], 'Private dataset input bytes/list changed')
        require({name: qa.digest(app / name) for name in binding['code_sha256']} == binding['code_sha256'], 'Packaged code changed')
        report['packaged_application_unchanged'] = True
        qa.require_readonly(app)
        report['success'] = True
    except Exception as error:
        # Exception text intentionally excludes source image or label filenames.
        (workspace / 'private-failure.txt').write_text(str(error), encoding='utf-8')
        report['errors'].append('Private precheck failed; see private retained diagnostic directory')
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    qa.atomic_json(args.report.resolve(), report)
    print(json.dumps({'success': report['success'], 'selected_images': report.get('selected_images'),
                      'validation': report.get('validation'), 'input_files_unchanged': report.get('input_files_unchanged'),
                      'model_inference': False, 'dataset_writes': False}, ensure_ascii=False))
    return 0 if report['success'] else 1

if __name__ == '__main__':
    sys.exit(main())
