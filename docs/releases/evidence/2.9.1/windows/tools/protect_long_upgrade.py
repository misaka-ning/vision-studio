import json,sys,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'packaging/windows'))
from accept_windows import Registration
matches=[]
for p in psutil.process_iter(['cmdline']):
    try:
        if any(str(x).endswith('run_optimized_upgrade.py') for x in p.info['cmdline'] or []): matches.append(p)
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
assert len(matches)==1
control=matches[0]
children=[p for p in control.children(recursive=True) if p.name().lower().endswith('.tmp')]
control.wait(timeout=2400)
qa=root/'build-windows-2.9.1/upgrade-final-optimized/upgrade-qa.json'
report=json.loads(qa.read_text(encoding='utf-8'))
if report.get('success') is not True:
    # The controller's finally block has restored the original registration.
    # Contain a still-running Inno child in another restore scope.
    with Registration():
        for child in children:
            try: child.wait(timeout=1800)
            except psutil.NoSuchProcess: pass
    (root/'build-windows/long-upgrade-inner-finished.json').write_text(json.dumps({'finished':True,'children':[p.pid for p in children]}),encoding='utf-8')
    print('Inner setup finished; original registration and links restored.',flush=True)
else: print('Original controller completed successfully.',flush=True)
