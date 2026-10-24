import json,subprocess,sys,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[1]; build=root/'build-windows-2.9.1'
out=root/'output/windows-2.9.1'; python=root/'build-windows/python/python.exe'
try:
    psutil.Process(28368).wait(timeout=900)
except psutil.NoSuchProcess: pass
original=build/'installer-offline-final-1/installer-qa.json'
assert json.loads(original.read_text(encoding='utf-8'))['success'] is True
gpu=build/'gpu-preparation-final-retry/runtime'
assert (gpu/'ready.json').is_file()
for profile, previous in [('online','installer-online-final-2'),('offline','installer-offline-final-1')]:
    command=[python,root/'packaging/windows/accept_final_delta.py','--profile',profile,
        '--qa-root',build/f'installer-final-delta-{profile}',
        '--installer',out/f'vision-studio-2.9.1-windows-x64-{profile}-setup.exe',
        '--previous-report',build/previous/'installer-qa.json',
        '--previous-manifest',out/f'components/pre-gpu-probe-fix-{profile}-files.json',
        '--final-manifest',out/f'stage-{profile}/WINDOWS-FILES.json',
        '--gpu-runtime',gpu,'--fixtures',root,
        '--runtime-archive',out/'vision-studio-2.9.1-windows-x64-cpu-runtime.zip',
        '--existing-data',build/previous/'personal']
    print('START final-delta',profile,flush=True)
    with (root/f'build-windows/final-delta-{profile}.log').open('wb') as log:
        subprocess.run([str(x) for x in command],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS final-delta',profile,flush=True)
print('START upgrade/profile-switch',flush=True)
command=[python,root/'packaging/windows/accept_upgrade.py','--qa-root',build/'upgrade-final',
    '--legacy',root/'output/windows-2.1.0/vision-studio-2.1.0-windows-x64-setup.exe',
    '--offline',out/'vision-studio-2.9.1-windows-x64-offline-setup.exe',
    '--online',out/'vision-studio-2.9.1-windows-x64-online-setup.exe',
    '--gpu-runtime',gpu]
with (root/'build-windows/upgrade-final.log').open('wb') as log:
    subprocess.run([str(x) for x in command],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PASS upgrade/profile-switch',flush=True)
for script in ('collect_windows_evidence.py','finalize_windows_qa.py'):
    subprocess.run([str(python),str(root/'build-windows'/script)],cwd=root,check=True)
print('ALL FINAL ACCEPTANCE PASSED',flush=True)
