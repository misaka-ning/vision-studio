import argparse,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); a=p.parse_args()
out=(root/'output').resolve(); versions=[]
for folder in out.iterdir():
    match=re.fullmatch(r'windows-(\d+)\.(\d+)\.(\d+)',folder.name)
    if not match or not folder.is_dir(): continue
    if folder.is_symlink() or folder.lstat().st_file_attributes & 0x400 or folder.resolve().parent!=out:
        raise RuntimeError('Unsafe Windows release directory')
    metadata=json.loads((folder/'windows-package-qa.json').read_text(encoding='utf-8-sig'))
    version='.'.join(match.groups())
    if metadata.get('success') is not True or metadata.get('version')!=version:
        raise RuntimeError('Unverified local Windows release metadata')
    versions.append((tuple(map(int,match.groups())),folder))
versions.sort(reverse=True); kept=[str(path) for _,path in versions[:2]]; candidates=[str(path) for _,path in versions[2:]]
if a.execute:
    proof=json.loads((out/'windows-2.9.1/remote-verification.json').read_text(encoding='utf-8'))
    if proof.get('success') is not True or len(proof.get('verified_assets',[]))!=5:
        raise RuntimeError('Remote verification is incomplete')
    # This host has exactly two Windows releases. Fail closed if the inventory
    # changes; never improvise deletion of an additional or unknown directory.
    if candidates: raise RuntimeError('Additional version directories require per-version remote preflight')
value={'success':True,'dry_run':not a.execute,'root':str(out),'kept':kept,'obsolete_candidates':candidates,
       'deleted':[],'reason':'Exactly two local Windows releases; no obsolete version directory to delete.',
       'preserved':'Git metadata, shared Qt/OpenCV SDKs, original Python/GPU environments, user installation/data and current build/evidence materials.'}
destination=out/'windows-2.9.1'/('retention-check.json' if a.execute else 'retention-preview.json')
destination.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=False,indent=2))
