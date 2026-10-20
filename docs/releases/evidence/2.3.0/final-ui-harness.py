#!/usr/bin/env python3
"""Capture real process return codes, frozen binaries and complete Qt logs."""
import hashlib,json,os,pathlib,shlex,subprocess,time
from datetime import datetime,timezone
ROOT=pathlib.Path('/home/misaka/.codex/worktrees/ef0b/yolo')
OUTPUT=ROOT/'output/qa-2.3.0/final-ui'
BUILD=ROOT/'output/build/2.3.0-release'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(value):
 temporary=OUTPUT/'run-qa.json.tmp'
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 temporary.replace(OUTPUT/'run-qa.json')
OUTPUT.mkdir(parents=True,exist_ok=False)
report={'schema_version':1,'application_binary_sha256':sha(BUILD/'bin/vision-studio'),'created_at_utc':utc(),'runs':[]}
save(report)
environment=os.environ.copy()
environment.update(QT_QPA_PLATFORM='offscreen',QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu')
for name,basename,extra in [
 ('vision-evaluation-integration-tests','integration',{'VISION_EVALUATION_INTEGRATION_EVIDENCE':str(OUTPUT)}),
 ('vision-ui-tests','ui-tests',{'VISION_UI_TEST_SCREENSHOT':str(OUTPUT/'ordinary-minimum-window.png')})]:
 binary=BUILD/name
 command=[str(binary),'-o',str(OUTPUT/(basename+'.log'))+',txt']
 run={'name':name,'command':command,'exit_code':None,'log':basename+'.log','binary':str(binary),'binary_sha256':sha(binary),'started_at_utc':utc()}
 report['runs'].append(run);save(report)
 print('Running '+shlex.join(command),flush=True)
 env=environment.copy();env.update(extra)
 begin=time.monotonic()
 with open(OUTPUT/(basename+'.stdout'),'wb') as stdout,open(OUTPUT/(basename+'.stderr'),'wb') as stderr:
  result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
 run.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-begin,finished_at_utc=utc(),log_sha256=sha(OUTPUT/(basename+'.log')),binary_unchanged=sha(binary)==run['binary_sha256'])
 save(report);print(name+' exit '+str(result.returncode),flush=True)
report['application_binary_unchanged']=sha(BUILD/'bin/vision-studio')==report['application_binary_sha256']
report['success']=report['application_binary_unchanged'] and all(run['exit_code']==0 and run['binary_unchanged'] for run in report['runs'])
report['finished_at_utc']=utc();save(report)
raise SystemExit(0 if report['success'] else 1)
