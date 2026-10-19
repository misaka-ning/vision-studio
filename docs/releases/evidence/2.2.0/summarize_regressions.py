from pathlib import Path
import hashlib,json,re,importlib.util,subprocess,datetime,copy
root=Path.cwd(); evidence=root/'docs/releases/evidence/2.2.0'
spec=importlib.util.spec_from_file_location('qa',root/'scripts/qa_detection_scope_220.py');qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inspected(name,log,xml):
 text=log.read_text(); totals,skipped,n,failed=qa.inspect_output(text,xml.read_text())
 assert n==1 and not skipped and not failed and '100% tests passed' in text,(name,totals)
 return {'name':name,'success':True,'exit_code':0,'qt_totals':totals,'skipped':False,'log':str(log.relative_to(evidence)),'log_sha256':sha(log),'junit':str(xml.relative_to(evidence)),'python_tests':sum(map(int,re.findall(r'Ran (\d+) tests? in ',text)))}
base=json.loads((evidence/'regression-01/detection-regression-qa.json').read_text()); rows=[]
for run in base['runs']:
 name=run['name']
 if name=='vision-ui-tests': folder=evidence/'ui-replay'; log=folder/(name+'.log'); xml=folder/(name+'.xml')
 elif name=='vision-conversion-inference-tests':folder=evidence/'conversion-final';log=folder/'ctest.log';xml=folder/'result.xml'
 else: folder=evidence/'regression-01';log=folder/run['log'];xml=folder/run['junit']
 rows.append(inspected(name,log,xml))
assert len(rows)==14 and len({r['name'] for r in rows})==14
production=[]
for name in ['vision-scope-filter-ui-tests','vision-ui-tests','vision-batch-responsiveness-tests']:
 folder=evidence/('production-ui-replay' if name=='vision-ui-tests' else 'production-ui')
 production.append(inspected(name,folder/(name+'.log'),folder/(name+'.xml')))
production_binary=root/'output/build/2.2.0-release/bin/vision-studio'
assert sha(production_binary)==json.loads((evidence/'production-ui-replay/detection-regression-qa.json').read_text())['binary_sha256']
qt=sum(t['passed'] for row in rows for t in row['qt_totals']); python=sum(row['python_tests'] for row in rows)
assert qt==226 and python==52,(qt,python)
profiles=[]
for p in sorted((evidence/'regression-01/artifacts').glob('onnx-cuda*.json')):
 events=json.loads(p.read_text());cuda=[x for x in events if x.get('cat')=='Node' and x.get('args',{}).get('provider')=='CUDAExecutionProvider'];assert len(cuda)>0
 profiles.append({'file':str(p.relative_to(evidence)),'sha256':sha(p),'cuda_node_events':len(cuda)})
assert len(profiles)==2
codefiles=[p for directory in ['src','tests'] for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ['.cpp','.h','.py']]
codefiles.extend(root/p for p in ['CMakeLists.txt','resources.qrc','scripts/pt_worker.py','scripts/model_convert.py','scripts/qa_detection_scope_220.py'])
report={'schema_version':1,'version':'2.2.0','success':True,'errors':[],'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Actual 14-suite regression with complete affected-suite replays, plus production-resource-path UI acceptance. DEB acceptance is recorded separately.','qt_tests_passed':qt,'python_tests_passed':python,'failed':0,'skipped':0,'blacklisted':0,'suites':rows,'production_ui_suites':production,'production_binary_sha256':sha(production_binary),'development_binary_sha256':base['binary_sha256'],'cuda_profiles':profiles,'hardware':'NVIDIA GeForce RTX 4060 / driver 595.99.02','source_code_input_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(codefiles)},'test_binary_sha256':{r['name']:sha(Path('/tmp/vision-studio-2.2.0-build')/r['name']) for r in rows if (Path('/tmp/vision-studio-2.2.0-build')/r['name']).is_file()},'superseded_runs':[{'path':'regression-01/detection-regression-qa.json','reason':'UI expectations assumed cached results survived changed thresholds; conversion runtime default referenced an unavailable old build directory. Actual failures retained. Full UI replay and full conversion replay replace only these two suites.'},{'path':'production-ui/detection-regression-qa.json','reason':'Production-source test initially found old installed PT helper at /opt and outdated threshold-cache expectations; explicit current helper and full corrected UI replay pass.'},{'path':'conversion-replay/ctest.log','reason':'Old converted-model test assumed COCO80 after preset invalidation. Updated to actual two-class metadata and reran the entire 14-case suite.'}],'limitations':['No physical camera hardware acceptance','No Windows 2.2.0 build or installation','Timing observations are local regression measurements, not performance guarantees']}
(evidence/'ctest-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'success':True,'suites':len(rows),'qt_passed':qt,'python_passed':python,'cuda_nodes':[x['cuda_node_events'] for x in profiles]}))
