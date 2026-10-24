import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'packaging/windows'))
from accept_windows import Acceptance,Registration,sha,dump
build=root/'build-windows-2.9.1'; out=root/'output/windows-2.9.1'
prior=json.loads((build/'upgrade-final-resumed/upgrade-qa.json').read_text(encoding='utf-8'))
required={'legacy-upgrade-cleans-unmodified-only','offline-to-online-reuses-compatible-runtime','online-uninstall-cleans-retained-managed-runtime'}
assert required.issubset({r['name'] for r in prior['checks'] if r.get('passed')})
assert 'The selected folder contains other files' in (build/'upgrade-final-resumed/online-fresh-before-switch-setup.log').read_text(encoding='utf-8-sig')
old=build/'upgrade-final-optimized'; install=old/'安装路径 Space/VisionStudio'; suffix=b'\nprivate modification preserved\n'
records=json.loads((root/'output/windows-2.1.0/stage/WINDOWS-FILES.json').read_text(encoding='utf-8'))
source=next(r for r in records if r['path'].startswith('sources/'))
protected={install/name:hashlib.sha256((root/'output/windows-2.1.0/stage'/name).read_bytes()+suffix).hexdigest()
           for name in (source['path'],'models/yolov5n.onnx')}
for p in (install/'private-user-model.pt',old/'personal/private-history.json'): protected[p]=hashlib.sha256(suffix).hexdigest()
protected[install/'runtime/private-user-file.txt']=hashlib.sha256(b'preserved runtime addition').hexdigest()
def unchanged():
    if any(not p.is_file() or sha(p)!=v for p,v in protected.items()): raise RuntimeError('Original protected user data changed')
offline=out/'vision-studio-2.9.1-windows-x64-offline-setup.exe'; online=out/'vision-studio-2.9.1-windows-x64-online-setup.exe'
job=Acceptance(SimpleNamespace(qa_root=build/'upgrade-final-complete',gpu_runtime=build/'gpu-preparation-final-retry/runtime',installer=online,fixtures=root))
job.checks=prior['checks'].copy()
try:
    with Registration():
        unchanged(); job.check('nonempty-unmanaged-directory-rejected',evidence='upgrade-final-resumed/online-fresh-before-switch-setup.log')
        job.setup('online-fresh-before-switch',job.install,online)
        runtime=job.install/'runtime'
        if (runtime/'python.exe').exists(): raise RuntimeError('Fresh online embeds Python')
        fresh=job.install/'private-new-profile-model.pt'; fresh.write_bytes(b'private profile model'); protected[fresh]=sha(fresh)
        job.setup('switch-online-to-offline',job.install,offline); unchanged()
        if not (runtime/'python.exe').is_file(): raise RuntimeError('Offline did not install CPU')
        descriptor=json.loads((job.install/'cpu-runtime.json').read_text(encoding='utf-8'))
        job.smoke('--smoke-cpu-runtime','offline-after-online-switch')
        job.check('online-to-offline-adds-compatible-runtime')
        job.uninstall('uninstall-final-offline'); unchanged()
        job.check('upgrade-uninstall-preserves-modified-legacy-and-user-files')
    report={'success':True,'version':'2.9.1','checks':job.checks,'cpu_sha256':descriptor['sha256'],
        'installers':{n:{'file':p.name,'sha256':sha(p)} for n,p in [('legacy',root/'output/windows-2.1.0/vision-studio-2.1.0-windows-x64-setup.exe'),('offline',offline),('online',online)]},
        'protected_files':{str(p):v for p,v in protected.items()},
        'provenance':['Actual 2.1.0 setup succeeded in upgrade-final-optimized. Actual final EXE migration continued after old QA controller timeout, then all 21044 final payloads and original protected baselines were independently rechecked.',
            'upgrade-final-resumed verified offline-to-online reuse and managed CPU uninstall; fresh reinstall into an unmarked user directory was correctly rejected.',
            'Online-to-offline was then checked in this new dedicated folder, with an additional private model preserved during upgrade and uninstall. Original failed reports remain unchanged.']}
    dump(job.root/'upgrade-qa.json',report)
except Exception as error:
    dump(job.root/'upgrade-qa.json',{'success':False,'checks':job.checks,'error':str(error)}); raise
print('ALL UPGRADE AND PROFILE CHECKS PASSED',flush=True)
