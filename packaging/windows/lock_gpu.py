#!/usr/bin/env python3
"""Resolve fixed official Windows GPU wheels to a reviewable SHA256 lock."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k,v in attrs if k == 'href')


def main():
    records=[]
    for name,version in [('torch','2.9.1+cu128'),('torchvision','0.24.1+cu128')]:
        index=f'https://download.pytorch.org/whl/cu128/{name}/'
        parser=Links(); parser.feed(urlopen(index,timeout=60).read().decode())
        filename=f'{name}-{version}-cp311-cp311-win_amd64.whl'
        link=next(url for url in parser.links if unquote(urlsplit(url).path.rsplit('/',1)[-1]) == filename)
        sha=urlsplit(link).fragment.removeprefix('sha256=')
        if len(sha) != 64: raise ValueError('Official wheel link has no SHA256')
        records.append(dict(name=name,version=version,filename=filename,sha256=sha,index=index,
                            url='https://download.pytorch.org/whl/cu128/'+filename.replace('+','%2B')))
    for name,version in [('onnxruntime-gpu','1.23.2'),('coloredlogs','15.0.1'),('flatbuffers','25.9.23'),('humanfriendly','10.0')]:
        metadata=json.load(urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60))
        matches=[p for p in metadata['urls'] if p['filename'].endswith(('cp311-cp311-win_amd64.whl','py2.py3-none-any.whl','py3-none-any.whl'))]
        if len(matches) != 1: raise ValueError((name,[p['filename'] for p in matches]))
        wheel=matches[0]
        records.append(dict(name=name,version=version,filename=wheel['filename'],sha256=wheel['digests']['sha256'],url=wheel['url']))
    (ROOT/'requirements-gpu-windows.lock.txt').write_text(
        '# Windows x64 / CPython 3.11. Exact official wheel hashes.\n'
        '# CPU dependencies are cloned from the bundled offline runtime.\n'+
        ''.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in records),encoding='ascii')
    (ROOT/'packaging/windows/gpu-wheel-sources.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Locked',len(records),'Windows GPU wheels')


if __name__ == '__main__': main()
