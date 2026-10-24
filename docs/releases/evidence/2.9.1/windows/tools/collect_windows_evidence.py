import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]; build=root/'build-windows-2.9.1'
evidence=root/'docs/releases/evidence/2.9.1/windows'; evidence.mkdir(exist_ok=True)
native={}
for name in ('native-tests-final-1','native-tests-final-2','native-tests-final-3'):
    for path in sorted((build/name).glob('*/result.json')):
        row=json.loads(path.read_text()); row['log_directory']=str(path.parent.relative_to(root)).replace('\\','/')
        native[row['name']]=row
native['vision-gpu-runtime-tests']={'name':'vision-gpu-runtime-tests','status':'not-applicable',
    'reason':'POSIX-specific installer suite; Windows uses actual GPU preparation and CUDA acceptance instead.'}
summary={'success':all(r.get('passed',r.get('status')=='not-applicable') for r in native.values()),
    'method':'Latest affected-suite retest replaces the earlier failed suite; original rounds retained in raw logs.',
    'results':[native[k] for k in sorted(native)]}
(evidence/'native-tests-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
selected={}
def add(path):
    if path.is_file(): selected[path.relative_to(root).as_posix()]=path
for folder in build.iterdir():
    if not folder.is_dir() or not folder.name.startswith(('native-tests','installer-online-final','installer-offline-final',
            'component-stage-qa','component-final-qa','upgrade-final','installer-final-delta')): continue
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix not in {'.log','.txt','.json'}: continue
        relative=path.relative_to(folder)
        if any(p in {'personal','data','ultralytics','matplotlib','isolated-user-profile','recording-tool',
                     '安装路径 Space','runtimes'} for p in relative.parts): continue
        add(path)
for path in (root/'build-windows').glob('*.log'):
    if any(key in path.name for key in ('291','final','component-','windows-','source-checks','iscc-','freeze-','upgrade','optimized')): add(path)
for path in (build/'installer-inputs').rglob('*'):
    if path.suffix in {'.json','.yaml'}: add(path)
add(build/'fixtures/coco8-source.json')
temporary=evidence/'raw-logs.tar.gz.partial'
with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
    for name,path in sorted(selected.items()): archive.add(path,arcname=name,recursive=False)
temporary.replace(evidence/'raw-logs.tar.gz')
index=[{'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in sorted(selected.items())]
(evidence/'raw-logs-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
tools=evidence/'tools';tools.mkdir(exist_ok=True)
for name in ('run_native_tests.py','make_acceptance_inputs.py','probe_runtime_final.py','collect_windows_evidence.py',
             'run_final_acceptance.py','finalize_windows_qa.py','write_windows_acceptance_readme.py',
             'freeze_windows_source.py','finalize_stages.py','build-291.cmd','resume_optimized_upgrade.py',
             'protect_long_upgrade.py','finish_resumed_acceptance.py','compile_optimized.py','run_optimized_upgrade.py',
             'finish_profile_switch.py','finish_complete_acceptance.py'):
    shutil.copy2(root/'build-windows'/name,tools/name)
print(json.dumps({'native_success':summary['success'],'suites':len(native),'raw_files':len(index),
    'archive_bytes':(evidence/'raw-logs.tar.gz').stat().st_size}),flush=True)
