#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Remove only three verified, unpublished, reconstructible 2.5.0 intermediates."""
from pathlib import Path
import hashlib,json,stat,datetime
ROOT=Path('/home/misaka/.codex/worktrees/ef0b/yolo')
OUT=ROOT/'output/v2.5.0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(v,m):
 if not v:raise RuntimeError(m)
def check(row):
 p=Path(row['path']);s=p.lstat()
 require(p.resolve()==p and stat.S_ISREG(s.st_mode) and not p.is_symlink(),'Not an ordinary exact file')
 require(s.st_size==row['bytes'] and sha(p)==row['sha256'],'Preserved or audited file changed')
 return p
report_path=OUT/'redundant-unpublished-archive-cleanup.json'
require(not report_path.exists(),'Receipt already exists')
audit=json.loads((OUT/'redundant-unpublished-archive-audit.json').read_text())
require(audit['success'] and not audit['errors'] and audit['eligible_after_explicit_cleanup_authorization'] and not audit['deletions_performed'],'Audit not eligible')
paths=[ROOT/'output/v2.5.0/diagnostics/pre-async-source/vision-studio-2.5.0-complete-source.tar.xz',ROOT/'output/v2.5.0/diagnostics/pre-async-source/vision-studio-2.5.0-sources.tar.xz',ROOT/'output/v2.5.0/source-input.tar']
require([str(p) for p in paths]==audit['exact_candidate_paths'],'Unexpected deletion scope')
remote=audit['final_remote_verified_replacement'];check(remote['receipt'])
v=json.loads(Path(remote['receipt']['path']).read_text());require(v['success'] and v['tag']=='v2.5.0' and v['commit']=='f25961a125d23dd2d8568a9b9a3cd144b3a16e86' and v['verified_assets']==remote['verified_assets'],'Final remote proof differs')
for row in v['verified_assets']:
 p=ROOT/'output/releases/2.5.0'/row['name'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Final retained asset changed')
preserved=[]
for row in audit['required_preservation_before_any_future_deletion']['keep_old_checksum_inventory_freeze_qa_and_logs']:check(row);preserved.append(row['path'])
def proofs(value):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=value.keys():
   p=Path(value['path'])
   if p.is_relative_to(OUT/'redundant-unpublished-archive-preservation') or p.is_relative_to(ROOT/'docs/releases/evidence/2.5.0/async-review-repair'):
    check(value);preserved.append(str(p))
  for child in value.values():proofs(child)
 elif isinstance(value,list):
  for child in value:proofs(child)
proofs(audit)
require(audit['application_archive_reconstruction']['recoverable_without_initial_remote_branch'] and audit['complete_archive_reconstruction']['all_18_upstream_materials_byte_identical'] and audit['source_input_tar_reconstruction']['byte_identical_to_regenerated_git_archive'],'Reconstruction proof incomplete')
rows=audit['candidates'];require([Path(r['path']) for r in rows]==paths,'Candidates differ')
for row in rows:
 p=check(row);s=p.stat();require(s.st_dev==row['device'] and s.st_ino==row['inode'] and s.st_mtime_ns==row['mtime_ns'],'Candidate identity changed')
r={'schema_version':1,'version':'2.5.0','success':False,'errors':[],'scope':'Exactly three unpublished duplicate source intermediates; final assets and diagnostics are retained','audit_sha256':sha(OUT/'redundant-unpublished-archive-audit.json'),'script_sha256':sha(Path(__file__)),'remote_tag':v['tag'],'remote_commit':v['commit'],'preserved_small_proof_files':sorted(set(preserved)),'deletions_performed':[],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 for row in rows:
  p=Path(row['path']);s=p.lstat();require(stat.S_ISREG(s.st_mode) and s.st_dev==row['device'] and s.st_ino==row['inode'] and s.st_mtime_ns==row['mtime_ns'] and s.st_size==row['bytes'],'Identity changed immediately before unlink')
  p.unlink();require(not p.exists(),'Unlink failed');r['deletions_performed'].append(row)
 r['success']=True;r['logical_bytes_removed']=sum(x['bytes'] for x in rows)
except Exception as e:r['errors'].append(str(e))
with report_path.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'success':r['success'],'deleted_files':len(r['deletions_performed']),'logical_bytes_removed':r.get('logical_bytes_removed'),'errors':r['errors']}))
require(r['success'],'Cleanup incomplete')
