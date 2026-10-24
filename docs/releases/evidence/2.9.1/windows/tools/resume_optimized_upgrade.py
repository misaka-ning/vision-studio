import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'packaging/windows'))
from accept_windows import Acceptance,Registration,sha,dump
build=root/'build-windows-2.9.1'; out=root/'output/windows-2.9.1'
ready=root/'build-windows/long-upgrade-inner-finished.json'
started=time.monotonic()
while not ready.is_file():
    if time.monotonic()-started>1800: raise RuntimeError('Inner installer did not complete')
    time.sleep(1)
a=SimpleNamespace(qa_root=build/'upgrade-final-resumed',gpu_runtime=build/'gpu-preparation-final-retry/runtime',
    installer=out/'vision-studio-2.9.1-windows-x64-offline-setup.exe',fixtures=root)
job=Acceptance(a); previous=build/'upgrade-final-optimized'
job.install=previous/'安装路径 Space/VisionStudio'; job.data=previous/'personal'
job.env.update(VISION_STUDIO_DATA_DIR=str(job.data),YOLO_CONFIG_DIR=str(job.data/'yolo-config'),MPLCONFIGDIR=str(job.data/'matplotlib'))
records=json.loads((root/'output/windows-2.1.0/stage/WINDOWS-FILES.json').read_text(encoding='utf-8'))
source=next(r for r in records if r['path'].startswith('sources/'))
suffix=b'\nprivate modification preserved\n'
protected={}
for name in (source['path'],'models/yolov5n.onnx'):
    original=root/'output/windows-2.1.0/stage'/name
    protected[job.install/name]=hashlib.sha256(original.read_bytes()+suffix).hexdigest()
for path in (job.install/'private-user-model.pt',job.data/'private-history.json'):
    protected[path]=hashlib.sha256(suffix).hexdigest()
def unchanged():
    if any(not p.is_file() or sha(p)!=v for p,v in protected.items()): raise RuntimeError('Protected baseline changed')
def payload():
    expected=json.loads((out/'stage-offline/WINDOWS-FILES.json').read_text(encoding='utf-8'))
    if sha(job.install/'WINDOWS-FILES.json')!=sha(out/'stage-offline/WINDOWS-FILES.json'):
        raise RuntimeError('Actual installation manifest differs from final package')
    def check(row):
        target=job.install/row['path']
        if not target.is_file() or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
            raise RuntimeError('Installed final payload mismatch: '+row['path'])
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(check,expected))
    job.check('resumed-final-installed-payload-hashes',files=len(expected))
try:
    with Registration():
        marker=json.loads((job.install/'windows-install.json').read_text(encoding='utf-8'))
        if marker['version']!='2.9.1' or marker['profile']!='offline': raise RuntimeError('Actual inner installation did not finish')
        unchanged(); payload()
        stale=[r['path'] for r in records if r['path'].startswith(('sources/','docs/','assets/','models/'))
               and job.install/r['path'] not in protected and (job.install/r['path']).is_file()]
        if stale: raise RuntimeError('Unchanged obsolete files survived: '+str(stale[:12]))
        job.check('legacy-upgrade-cleans-unmodified-only',old_files=len(records),protected_files=len(protected),
            method='Actual old EXE installation and final EXE migration; waited for inner setup after old controller timed out, then rehashed all final payloads and original protected baselines')
        runtime=job.install/'runtime'; private=runtime/'private-user-file.txt'
        private.write_text('preserved runtime addition',encoding='utf-8'); protected[private]=sha(private)
        descriptor=json.loads((job.install/'cpu-runtime.json').read_text(encoding='utf-8')); inventory=sha(runtime/'CPU-FILES.json')
        online=out/'vision-studio-2.9.1-windows-x64-online-setup.exe'; offline=a.installer
        job.setup('switch-offline-to-online',job.install,online); unchanged()
        if not (runtime/'python.exe').is_file() or sha(runtime/'CPU-FILES.json')!=inventory: raise RuntimeError('Compatible CPU not retained')
        result=job.smoke('--smoke-cpu-runtime','retained-runtime-without-download')
        if Path(result['python']).resolve()!=runtime/'python.exe': raise RuntimeError('Retained CPU not selected')
        job.check('offline-to-online-reuses-compatible-runtime')
        job.uninstall('uninstall-online-retained-runtime'); unchanged()
        if (runtime/'python.exe').exists() or (runtime/'CPU-FILES.json').exists(): raise RuntimeError('Managed CPU survived online uninstall')
        job.check('online-uninstall-cleans-retained-managed-runtime')
        job.setup('online-fresh-before-switch',job.install,online)
        if (runtime/'python.exe').exists(): raise RuntimeError('Fresh online contains Python')
        job.setup('switch-online-to-offline',job.install,offline); unchanged()
        if not (runtime/'python.exe').is_file(): raise RuntimeError('Offline CPU missing')
        job.smoke('--smoke-cpu-runtime','offline-after-online-switch')
        job.check('online-to-offline-adds-compatible-runtime')
        job.uninstall('uninstall-final-offline'); unchanged()
        job.check('upgrade-uninstall-preserves-modified-legacy-and-user-files')
    report={'success':True,'version':'2.9.1','checks':job.checks,'cpu_sha256':descriptor['sha256'],
        'installers':{n:{'file':p.name,'sha256':sha(p)} for n,p in [('legacy',root/'output/windows-2.1.0/vision-studio-2.1.0-windows-x64-setup.exe'),('offline',offline),('online',online)]},
        'protected_files':{str(p):v for p,v in protected.items()},
        'resumed_from':'upgrade-final-optimized: old 600-second QA loader limit expired while Inno continued; inner setup succeeded and every final payload was independently rehashed before continuation'}
    dump(job.root/'upgrade-qa.json',report)
except Exception as error:
    dump(job.root/'upgrade-qa.json',{'success':False,'checks':job.checks,'error':str(error)}); raise
print('ALL UPGRADE AND PROFILE CHECKS PASSED',flush=True)
