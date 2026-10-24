import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; build=root/'build-windows-2.9.1'
out=root/'output/windows-2.9.1'; ev=root/'docs/releases/evidence/2.9.1/windows'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
profiles={}; originals={'offline':'installer-offline-final-1','online':'installer-online-final-2'}
for profile in ('offline','online'):
    report=read(build/f'installer-final-delta2-{profile}/installer-qa.json')
    assert report['success'] is True
    exe=out/f'vision-studio-2.9.1-windows-x64-{profile}-setup.exe'
    assert report['sha256']==sha(exe)
    shutil.copy2(build/f'installer-final-delta2-{profile}/installer-qa.json',ev/f'installer-{profile}-qa.json')
    shutil.copy2(build/originals[profile]/'installer-qa.json',ev/f'installer-{profile}-full-original.json')
    profiles[profile]={'file':exe.name,'sha256':report['sha256'],'bytes':exe.stat().st_size,
                       'installed_bytes':report['final_delta']['installed_bytes']}
upgrade=read(build/'upgrade-final-complete/upgrade-qa.json'); assert upgrade['success'] is True
shutil.copy2(build/'upgrade-final-complete/upgrade-qa.json',ev/'upgrade-qa.json')
component=read(build/'component-final-qa/result.json')
assert component['success'] and component['single_download']
runtime=out/'vision-studio-2.9.1-windows-x64-cpu-runtime.zip'; runtime_sha=sha(runtime)
assert component['report']['runtime']['sha256']==runtime_sha
assert upgrade['cpu_sha256']==runtime_sha
native=read(ev/'native-tests-summary.json'); assert native['success']
assert 'GPU 环境自检通过' in (root/'build-windows/gpu-preparation-final-retry.log').read_text(encoding='utf-8')
assert 'Ran 10 tests' in (root/'build-windows/windows-packaging-optimized.log').read_text(encoding='utf-8')
assert (root/'build-windows/windows-packaging-optimized.log').read_text(encoding='utf-8').rstrip().endswith('OK')
runtime_qa={'success':True,'sha256':runtime_sha,'version':'2.9.1','checks':[
    {'name':'dependency-self-test','passed':True,'evidence':'component-final-qa/result.json; PowerShell extraction imported CPU dependencies before ready marker'},
    {'name':'cancel-and-retry','passed':True,'evidence':'CpuRuntimeTests cancellationReleasesPreparationLock and checksumFailureCanRetryWithoutExecutingArchive; actual successful preparation logged separately'},
    {'name':'checksum-rejection','passed':True,'evidence':'CpuRuntimeTests checksumFailureCanRetryWithoutExecutingArchive'},
    {'name':'concurrent-preparation','passed':True,'evidence':'component-final-qa/result.json: two real processes, exactly one download, then source stopped for offline reuse'},
    {'name':'archive-path-protection','passed':True,'evidence':'windows-packaging-optimized.log: Windows PowerShell traversal, duplicate, ADS and bad inventory rejection'},
    {'name':'upgrade-cleanup-protection','passed':True,'evidence':'windows-packaging-optimized.log and upgrade-qa.json: complete preflight and modified-file retention'},
    {'name':'profile-switching','passed':True,'evidence':'upgrade-qa.json: both final EXEs, compatible runtime retained, clean managed uninstall'},
    {'name':'gpu-preparation-from-trimmed-cpu','passed':True,'evidence':'gpu-preparation-final-retry.log and both final_delta CUDA retests'}],
    'component_preparation':component,'gpu_preparation':'Pinned local wheels; real torch CUDA and ONNX CUDA MatMul passed; driver unchanged'}
write(ev/'cpu-runtime-qa.json',runtime_qa)
source=out/'vision-studio-2.9.1-windows-sources.zip'
qa={'schema_version':2,'success':True,'version':'2.9.1','package_revision':'windows.1','profiles':profiles,
    'cpu_runtime':{'file':runtime.name,'bytes':runtime.stat().st_size,'sha256':runtime_sha,
                   'uncompressed_bytes':component['report']['runtime']['uncompressed_bytes']},
    'source_snapshot':{'file':source.name,'bytes':source.stat().st_size,'sha256':sha(source)},
    'binary_sha256':sha(out/'stage-online/bin/vision-studio.exe'),
    'native_tests':'27 registered suites: 26 applicable suites pass after targeted retests; POSIX-only installer suite not applicable; platform skips recorded in raw logs.',
    'limitations':read(ev/'installer-offline-qa.json')['limitations'],
    'built_application_commit':'a29c7a2','packaged_runtime_script_commit':'54f01d2',
    'source_snapshot_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
write(ev/'windows-package-qa.json',qa); write(out/'windows-package-qa.json',qa)
names=[profiles[p]['file'] for p in ('offline','online')]+[runtime.name,source.name]
(out/'SHA256SUMS-windows-x64').write_text(''.join(sha(out/n)+'  '+n+'\n' for n in names),encoding='ascii')
write(ev/'sizes.json',{'old_installer_bytes':411996451,'profiles':profiles,'cpu_runtime':qa['cpu_runtime'],'source_snapshot':qa['source_snapshot']})
print(json.dumps(qa,ensure_ascii=False,indent=2),flush=True)
