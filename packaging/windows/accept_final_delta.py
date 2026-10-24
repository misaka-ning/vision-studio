#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bind an unaffected full functional run to a final, narrowly changed installer.

Every payload file must be identical apart from the explicitly rechecked GPU
probe and the build provenance marker. The final installed payload is rehashed,
CUDA probes/inference are rerun, then uninstall/data retention is rechecked.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from accept_windows import Acceptance, Registration, dump, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('qa-root','installer','previous-report','previous-manifest','final-manifest',
                 'gpu-runtime','fixtures','runtime-archive','existing-data'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('offline','online'),required=True)
    a=p.parse_args()
    previous=json.loads(a.previous_report.read_text(encoding='utf-8'))
    if previous.get('success') is not True: raise RuntimeError('Previous complete acceptance failed')
    before={r['path']:r for r in json.loads(a.previous_manifest.read_text())}
    after={r['path']:r for r in json.loads(a.final_manifest.read_text())}
    if before.keys()!=after.keys(): raise RuntimeError('The final payload member set changed')
    changed={path for path in before if before[path]!=after[path]}
    if changed!={'scripts/gpu_probe.py','windows-install.json'}:
        raise RuntimeError('Unexpected payload difference: '+str(changed))
    job=Acceptance(SimpleNamespace(qa_root=a.qa_root,gpu_runtime=a.gpu_runtime,installer=a.installer,
                                   fixtures=a.fixtures,runtime_archive=a.runtime_archive))
    # Online cached CPU was genuinely downloaded and checked in the prior run.
    # The follow-up deliberately reuses the same user data to verify retention.
    job.env['VISION_STUDIO_DATA_DIR']=str(a.existing_data.resolve())
    job.env['YOLO_CONFIG_DIR']=str(a.existing_data.resolve()/'yolo-config')
    job.env['MPLCONFIGDIR']=str(a.existing_data.resolve()/'matplotlib')
    try:
        with Registration():
            job.setup('install-final-delta',job.install); job.payload()
            installed_bytes=sum(path.stat().st_size for path in job.install.rglob('*') if path.is_file())
            job.smoke('--smoke','final-onnx-cuda',('--device','cuda'))
            job.smoke('--smoke-pt','final-pt-cuda',('--device','cuda'))
            sentinel=job.install/'private-final-user-model.pt'; sentinel.write_bytes(b'private retained model')
            job.uninstall('uninstall-final-delta')
            if not sentinel.is_file() or sentinel.read_bytes()!=b'private retained model':
                raise RuntimeError('Final uninstall changed user model')
            job.check('final-uninstall-preserves-user-files')
        report=dict(previous,sha256=sha(a.installer),
            verification_method='Aggregate: complete functional acceptance plus exact file comparison and final installed affected-function retest',
            earlier_full_installer_sha256=previous['sha256'],
            earlier_full_report=str(a.previous_report),
            final_delta={'changed_payload_files':sorted(changed),'identical_payload_files':len(before)-len(changed),
                         'installed_bytes':installed_bytes,
                         'checks':job.checks,'gpu_runtime':str(a.gpu_runtime),'final_installer_sha256':sha(a.installer)})
        report['limitations'].append('Functional checks before the last GPU probe script correction are combined with final installer payload equality and affected CUDA/uninstall retests; see final_delta and the unchanged original report.')
        dump(job.root/'installer-qa.json',report)
    except Exception as error:
        dump(job.root/'installer-qa.json',{'success':False,'checks':job.checks,'error':str(error)}); raise


if __name__=='__main__': main()
