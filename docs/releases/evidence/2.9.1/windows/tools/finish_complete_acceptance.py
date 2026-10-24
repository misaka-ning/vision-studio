import json,subprocess,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[1]; build=root/'build-windows-2.9.1'; out=root/'output/windows-2.9.1'
matches=[]
for p in psutil.process_iter(['cmdline']):
    try:
        if any(str(x).endswith('finish_profile_switch.py') for x in p.info['cmdline'] or []): matches.append(p)
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
assert len(matches)==1, 'Expected one isolated upgrade controller'
matches[0].wait(timeout=2400)
assert json.loads((build/'upgrade-final-complete/upgrade-qa.json').read_text(encoding='utf-8'))['success'] is True
python=root/'build-windows/python/python.exe'
for profile, previous in [('online','installer-online-final-2'),('offline','installer-offline-final-1')]:
    command=[python,root/'packaging/windows/accept_final_delta.py','--profile',profile,
        '--qa-root',build/f'installer-final-delta2-{profile}',
        '--installer',out/f'vision-studio-2.9.1-windows-x64-{profile}-setup.exe',
        '--previous-report',build/previous/'installer-qa.json',
        '--previous-manifest',out/f'components/pre-gpu-probe-fix-{profile}-files.json',
        '--final-manifest',out/f'stage-{profile}/WINDOWS-FILES.json',
        '--gpu-runtime',build/'gpu-preparation-final-retry/runtime','--fixtures',root,
        '--runtime-archive',out/'vision-studio-2.9.1-windows-x64-cpu-runtime.zip',
        '--existing-data',build/previous/'personal']
    print('START optimized final-delta',profile,flush=True)
    with (root/f'build-windows/final-delta2-{profile}.log').open('wb') as log:
        subprocess.run([str(x) for x in command],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS optimized final-delta',profile,flush=True)
for name in ('freeze_windows_source.py','collect_windows_evidence.py','finalize_windows_qa.py','write_windows_acceptance_readme.py'):
    subprocess.run([str(python),str(root/'build-windows'/name)],cwd=root,check=True)
print('ALL OPTIMIZED FINAL ACCEPTANCE PASSED',flush=True)
