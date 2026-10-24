#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Real, isolated legacy upgrade and dual-profile switching acceptance."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from accept_windows import Acceptance, Registration, dump, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('qa-root','legacy','offline','online','gpu-runtime'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    job=Acceptance(SimpleNamespace(qa_root=a.qa_root,gpu_runtime=a.gpu_runtime,installer=a.offline))
    protected={}
    def unchanged():
        if any(not path.is_file() or sha(path)!=value for path,value in protected.items()):
            raise RuntimeError('An upgraded or uninstalled user file changed')
    try:
        with Registration():
            job.setup('legacy-2.1-install',job.install,a.legacy)
            records=json.loads((job.install/'WINDOWS-FILES.json').read_text())
            old_sources=[r for r in records if r['path'].startswith('sources/')]
            source=next(r for r in old_sources if (job.install/r['path']).is_file())
            for path in (job.install/source['path'],job.install/'models/yolov5n.onnx',
                         job.install/'private-user-model.pt',job.data/'private-history.json'):
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('ab') as stream: stream.write(b'\nprivate modification preserved\n')
                protected[path]=sha(path)
            job.setup('upgrade-2.1-to-offline',job.install,a.offline)
            unchanged()
            stale=[r['path'] for r in records if r['path'].startswith(('sources/','docs/','assets/','models/'))
                   and job.install/r['path'] not in protected and (job.install/r['path']).is_file()]
            if stale: raise RuntimeError('Unchanged obsolete files survived: '+str(stale[:12]))
            job.check('legacy-upgrade-cleans-unmodified-only',old_sources=len(old_sources),protected_files=len(protected))
            runtime=job.install/'runtime'; private=runtime/'private-user-file.txt'
            private.write_text('preserved runtime addition',encoding='utf-8'); protected[private]=sha(private)
            descriptor=json.loads((job.install/'cpu-runtime.json').read_text())
            inventory=sha(runtime/'CPU-FILES.json')
            job.setup('switch-offline-to-online',job.install,a.online)
            unchanged()
            if not (runtime/'python.exe').is_file() or sha(runtime/'CPU-FILES.json')!=inventory:
                raise RuntimeError('Compatible bundled runtime was not retained')
            result=job.smoke('--smoke-cpu-runtime','retained-runtime-without-download')
            if Path(result['python']).resolve()!=runtime/'python.exe': raise RuntimeError('Retained runtime was not selected')
            job.check('offline-to-online-reuses-compatible-runtime')
            job.uninstall('uninstall-online-retained-runtime')
            unchanged()
            if (runtime/'python.exe').exists() or (runtime/'CPU-FILES.json').exists():
                raise RuntimeError('Uninstall left the retained managed CPU environment')
            job.check('online-uninstall-cleans-retained-managed-runtime')
            job.setup('online-fresh-before-switch',job.install,a.online)
            if (runtime/'python.exe').exists(): raise RuntimeError('Fresh online installer includes Python')
            job.setup('switch-online-to-offline',job.install,a.offline)
            unchanged()
            if not (runtime/'python.exe').is_file(): raise RuntimeError('Offline switch did not restore CPU environment')
            job.smoke('--smoke-cpu-runtime','offline-after-online-switch')
            job.check('online-to-offline-adds-compatible-runtime')
            job.uninstall('uninstall-final-offline'); unchanged()
            job.check('upgrade-uninstall-preserves-modified-legacy-and-user-files')
        report={'success':True,'version':'2.9.1','checks':job.checks,
                'installers':{name:{'file':getattr(a,name).name,'sha256':sha(getattr(a,name))} for name in ('legacy','offline','online')},
                'cpu_sha256':descriptor['sha256'],'protected_files':{str(k):v for k,v in protected.items()}}
    except Exception as error:
        dump(job.root/'upgrade-qa.json',{'success':False,'checks':job.checks,'error':str(error)}); raise
    dump(job.root/'upgrade-qa.json',report)


if __name__=='__main__': main()
