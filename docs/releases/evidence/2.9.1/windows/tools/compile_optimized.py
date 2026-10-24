import concurrent.futures,hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'output/windows-2.9.1'
def compile(profile):
    exe=out/f'vision-studio-2.9.1-windows-x64-{profile}-setup.exe'
    destination=out/f'components/pre-cleanup-optimization-{profile}-setup.exe'
    if exe.exists(): exe.rename(destination)
    command=[root/'build-windows/tools/InnoSetup7/ISCC.exe','/Qp','/DAppVersion=2.9.1',
        '/DInstallProfile='+profile,'/DStageDir='+str(out/('stage-'+profile)),
        '/DOutputDir='+str(out),str(root/'packaging/windows/vision-studio.iss')]
    with (root/f'build-windows/iscc-{profile}-optimized.log').open('wb') as log:
        subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,check=True)
    with exe.open('rb') as stream: checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'file':str(exe),'bytes':exe.stat().st_size,'sha256':checksum}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(compile,('offline','online')))
(root/'build-windows/optimized-installers-ready.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2),flush=True)
