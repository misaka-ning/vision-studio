import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'packaging/windows'))
import accept_upgrade
from accept_windows import Acceptance,sha
class WaitForFinal(Acceptance):
    def setup(self,name,*args,**kwargs):
        if name=='upgrade-2.1-to-offline':
            ready=root/'build-windows/optimized-installers-ready.json'
            print('Legacy installation prepared; waiting for final compilers',flush=True)
            started=time.monotonic()
            while not ready.is_file():
                if time.monotonic()-started>1200: raise RuntimeError('Final installers were not prepared')
                time.sleep(1)
            for row in json.loads(ready.read_text(encoding='utf-8')):
                if sha(Path(row['file']))!=row['sha256']: raise RuntimeError('Installer changed after ready marker')
        return super().setup(name,*args,**kwargs)
accept_upgrade.Acceptance=WaitForFinal
sys.argv=[str(root/'packaging/windows/accept_upgrade.py'),
    '--qa-root',str(root/'build-windows-2.9.1/upgrade-final-optimized'),
    '--legacy',str(root/'output/windows-2.1.0/vision-studio-2.1.0-windows-x64-setup.exe'),
    '--offline',str(root/'output/windows-2.9.1/vision-studio-2.9.1-windows-x64-offline-setup.exe'),
    '--online',str(root/'output/windows-2.9.1/vision-studio-2.9.1-windows-x64-online-setup.exe'),
    '--gpu-runtime',str(root/'build-windows-2.9.1/gpu-preparation-final-retry/runtime')]
accept_upgrade.main()
