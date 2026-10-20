from pathlib import Path
import json,gzip,hashlib,stat,os,shutil,subprocess,time
ROOT=Path.cwd();E=ROOT/'docs/releases/evidence/2.4.0';R=ROOT/'output/releases/2.4.0'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
new=json.loads((R/'release-verify.json').read_text());old=json.loads((R/'previous-2.3.0-verify.json').read_text());assert new['success'] is True and old['success'] is True and new['tag']=='v2.4.0' and old['tag']=='v2.3.0'
assert len(new['verified_assets'])==len(old['verified_assets'])==3
assets=json.loads((E/'final-assets.json').read_text())['assets']
assert all(assets[a['name']]['sha256']==a['sha256'] and assets[a['name']]['bytes']==a['bytes'] for a in new['verified_assets'])
preview=json.loads((E/'cleanup-old-230-preview.json').read_text());assert preview['local_preflight_checks_passed'] and not preview['errors']
for row in preview['preserved_backups']:
 p=ROOT/row['preserved_copy'];assert p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256']
allowed={'source':ROOT/'output/v2.3.0/clean-source','build':ROOT/'output/build/2.3.0-release','stage':ROOT/'output/deb-stage/2.3.0-1'}
verified=[]
for label,path in allowed.items():
 assert path.is_dir() and not path.is_symlink() and path.resolve()==path
 manifest=E/f'cleanup-old-230-{label}-members.jsonl.gz'
 with gzip.open(manifest,'rt') as f:expected={r['path']:r for r in map(json.loads,f)}
 actual={p.relative_to(path).as_posix():p for p in path.rglob('*') if label!='source' or not p.is_dir()}
 if label=='stage':
  # dpkg-deb payload/control member audit enumerates the control files but
  # not their generated DEBIAN parent directory. Permit only that exact parent.
  assert set(actual)-set(expected)=={'DEBIAN'} and not (set(expected)-set(actual))
  parent=actual['DEBIAN'];assert parent.is_dir() and not parent.is_symlink() and stat.S_IMODE(parent.lstat().st_mode)==0o755
  expected['DEBIAN']={'type':'directory','mode':0o755,'origin':'generated_control_parent'}
 assert set(actual)==set(expected),(label,'members differ')
 total=0
 for rel,row in expected.items():
  p=actual[rel];mode=p.lstat().st_mode
  if row['type']=='directory':assert stat.S_ISDIR(mode) and not p.is_symlink()
  elif row['type']=='symlink':assert p.is_symlink() and os.readlink(p)==row['target']
  else:
   assert stat.S_ISREG(mode) and not p.is_symlink() and p.stat().st_size==row['size'] and digest(p)==row['sha256'],(label,rel)
   total+=row['size']
  if 'mode' in row:assert stat.S_IMODE(mode)==row['mode'],(label,rel,'mode')
 assert not list(path.rglob('.git'))
 verified.append({'path':str(path),'members_rechecked':len(expected),'logical_bytes':total,'manifest_sha256':digest(manifest)})
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:exe=os.readlink(process/'exe').removesuffix(' (deleted)')
 except OSError:continue
 assert not any(exe==str(p) or exe.startswith(str(p)+'/') for p in allowed.values()),'An old build still runs'
for path in allowed.values():shutil.rmtree(path)
# The unpublished original 2.4.0 source archives have no unique application inputs;
# their immutable Git revision, checksums and original QA reports are retained.
record=json.loads((E/'diagnostics/pre-fixture-source/archive-record.json').read_text());assert record['published'] is False
removed=[]
for name,row in record['archives'].items():
 path=ROOT/'output/v2.4.0/diagnostics/pre-fixture-source'/name
 assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
 path.unlink();removed.append({'path':str(path),**row})
report={'schema_version':1,'version':'2.4.0','success':True,'errors':[],'remote_new_and_old_verified':True,'preview':'cleanup-old-230-preview.json','exact_deleted_old_paths':verified,'unpublished_redundant_archives_removed':removed,'protected_user_data_and_other_qa':True,'shared_qt_and_cpu_gpu_runtimes_removed':False,'bytes_in_redundant_archives':sum(r['bytes'] for r in removed)}
(R/'old-source-cleanup.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'success':True,'deleted_old_paths':len(verified),'redundant_archives':len(removed)}))
