from pathlib import Path
import hashlib, json, tarfile, importlib.util
root=Path.cwd()
release=root/'output/releases/1.6.0'
version='1.6.0'

def digest_file(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def digest_stream(stream):
 d=hashlib.sha256()
 for b in iter(lambda:stream.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def manifest(path):
 return {line.split('  ',1)[1]:line.split('  ',1)[0] for line in path.read_text().splitlines()}
expected=manifest(release/'SOURCE-SHA256SUMS')
public=manifest(release/'SHA256SUMS')
for name,sha in (expected|public).items():
 assert digest_file(release/name)==sha, name
complete=release/f'vision-studio-{version}-complete-source.tar.xz'
verified=[]
with tarfile.open(complete,'r|xz') as tf:
 for info in tf:
  relative=info.name.split('/',1)[1] if '/' in info.name else ''
  if relative in expected:
   assert info.isfile(),relative
   actual=digest_stream(tf.extractfile(info))
   assert actual==expected[relative],relative
   verified.append(relative)
  elif relative=='SOURCE-SHA256SUMS':
   assert tf.extractfile(info).read()==(release/relative).read_bytes()
assert set(verified)==set(expected),(set(expected)-set(verified))
app=release/f'vision-studio-{version}-sources.tar.xz'
rows=[]
with tarfile.open(app,'r:xz') as tf:
 for info in tf:
  rel=info.name.split('/',1)[1]
  local=root/rel
  assert local.exists(),rel
  if info.isfile():
   actual=digest_stream(tf.extractfile(info))
   assert actual==digest_file(local),rel
   rows.append({'file':rel,'bytes':info.size,'sha256':actual})
  elif info.issym():
   assert local.is_symlink() and str(local.readlink())==info.linkname,rel
  else:
   raise AssertionError(f'Unexpected member {rel}')
required={'src/core/gpuruntime.cpp','src/core/onnxcudabackend.cpp','scripts/gpu_setup.py','scripts/gpu_probe.py','scripts/setup_gpu.sh','requirements-gpu.lock.txt','requirements-gpu.txt','vendor/onnxruntime/onnxruntime_c_api.h','vendor/onnxruntime/LICENSE','packaging/verify_gpu_deb.py','tests/gpu_backend_tests.cpp','tests/gpu_runtime_tests.py'}
assert required <= {r['file'] for r in rows},required-{r['file'] for r in rows}
report={'success':True,'version':version,'errors':[],
'embedded_source_files_sha256_verified':len(verified),
'frozen_application_sources_byte_equal':len(rows),
'application_source':{'file':app.name,'bytes':app.stat().st_size,'sha256':digest_file(app)},
'complete_source':{'file':complete.name,'bytes':complete.stat().st_size,'sha256':digest_file(complete)},
'deb_sha256_confirmed':public[f'vision-studio_{version}-1_amd64.deb'],
'required_gpu_source_files_verified':sorted(required),
'note':'All application archive files match frozen source inputs byte for byte. Final package/source verification reports are added to Git after archive creation to avoid recursive self-hashing; product/build sources remain identical.'}
(release/'source-bundle-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(root/'output/v1.6-development/frozen-source-files.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
