import importlib.util,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',root/'packaging/windows/build_windows.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
path=root/'output/windows-2.9.1/vision-studio-2.9.1-windows-sources.zip'
m.make_source_snapshot(path,root/'build-windows/downloads/ffmpeg-sources')
print(json.dumps({'file':path.name,'bytes':path.stat().st_size,'sha256':m.digest(path)}),flush=True)
