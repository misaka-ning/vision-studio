import importlib.util,json,pathlib,shutil,subprocess
root=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',root/'packaging/windows/build_windows.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
out=root/'output/windows-2.9.1'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
for profile in ('offline','online'):
    stage=out/f'stage-{profile}'
    shutil.copy2(root/'build-windows-2.9.1/bin/vision-studio.exe',stage/'bin/vision-studio.exe')
    for name in b.RUNTIME_SCRIPTS:shutil.copy2(root/'scripts'/name,stage/'scripts'/name)
    shutil.copy2(root/'packaging/windows/cleanup_install.ps1',stage/'scripts/cleanup_install.ps1')
    marker=json.loads((stage/'windows-install.json').read_text());marker['build_commit']=commit
    (stage/'windows-install.json').write_text(json.dumps(marker,indent=2)+'\n',encoding='utf-8')
    print('hashing',profile,flush=True)
    records=b.write_manifest(stage)
    print(json.dumps({'profile':profile,'files':len(records),'installed_bytes':sum(f['bytes'] for f in records),
        'binary_sha256':b.digest(stage/'bin/vision-studio.exe')}),flush=True)
