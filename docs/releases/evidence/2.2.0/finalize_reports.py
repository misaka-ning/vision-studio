from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
root=Path.cwd();release=root/'output/releases/2.2.0';ev=root/'docs/releases/evidence/2.2.0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(name):return json.loads((release/name).read_text())
now=datetime.datetime.now(datetime.timezone.utc).isoformat();deb=release/'vision-studio_2.2.0-1_amd64.deb';dh=sha(deb)
qa=load('qa-report.json');install=load('install-qa.json');gpu=load('packaged-gpu-qa.json');source=load('source-bundle-qa.json');ct=json.loads((ev/'ctest-qa.json').read_text())
for r in [qa,install,source]:assert r['success'] and not r['errors'] and r['details']['deb_sha256']==dh
assert gpu['success'] and not gpu['errors'] and gpu['deb_sha256']==dh
assert ct['success'] and ct['failed']==ct['skipped']==0
stage=root/'output/deb-stage/2.2.0-1/opt/VisionStudio/build/bin/vision-studio';stagehash=sha(stage)
checks={'onnx_cuda_nodes':all(p['cuda_node_events']>0 and sha(ev/p['file'])==p['sha256'] for p in ct['cuda_profiles']),'pt_cuda':False,'packaged_onnx_cuda':gpu['checks']['packaged_onnx_cuda'],'packaged_pt_cuda':gpu['checks']['packaged_pt_cuda'],'cpu_regression':gpu['checks']['cpu_regression'],'grayscale_c1_c3':False,'stereo_recording':False}
gpulog=(ev/'regression-01/vision-gpu-tests.log').read_text();reclog=(ev/'regression-01/vision-recording-tests.log').read_text()
checks['pt_cuda']='PASS   : GpuBackendTests::realModernAndLegacyPtColorAndGrayscale()' in gpulog
checks['grayscale_c1_c3']=all('PASS   : GpuBackendTests::'+name+'()' in gpulog for name in ['torchScriptC1AndC3GrayscaleDetectionAndClassification','nativeGrayC1C3Classification','onnxC1AndC3GrayPixelsMatchModelInput'])
checks['stereo_recording']='PASS   : RecordingTests::gpuRightEyeGrayscaleRecordingPreservesActualDevice()' in reclog
assert all(checks.values())
report={'schema_version':1,'version':'2.2.0','success':True,'errors':[],'generated_at_utc':now,'scope':'Read-only aggregation of actual CUDA profiles, complete effective regressions, and four final-package CPU/CUDA cases; no extra inference executed by this aggregation.','checks':checks,'details':{'deb_sha256':dh,'deb_bytes':deb.stat().st_size,'packaged_binary_sha256':stagehash,'production_unstripped_binary_sha256':ct['production_binary_sha256'],'hardware':ct['hardware'],'ctest_report':'ctest-qa.json','packaged_report':'packaged-gpu-qa.json','cuda_profiles':ct['cuda_profiles']}}
(release/'gpu-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert all(qa['details']['smoke']['results'][n]['success'] for n in ['conversion_onnx','conversion_torchscript'])
assert next(s for s in ct['suites'] if s['name']=='vision-conversion-inference-tests')['qt_totals'][0]['passed']==14
conversion={'schema_version':1,'version':'2.2.0','success':True,'errors':[],'generated_at_utc':now,'details':{'deb_sha256':dh,'packaged_binary_sha256':stagehash,'full_conversion_regression':'conversion-final/ctest.log','python_actual_conversion_regression':'regression-01/vision-model-conversion-tests.log','readonly_final_package_smokes':{k:qa['details']['smoke']['results'][k] for k in ['conversion_onnx','conversion_torchscript']}},'scope':'Actual full conversion regressions plus final read-only DEB ONNX and TorchScript export/inference; not a new converter implementation.'}
(release/'conversion-qa.json').write_text(json.dumps(conversion,ensure_ascii=False,indent=2)+'\n')
raw=root/'output/qa-2.2.0/final-gui';g=json.loads((raw/'report/installed-gui-qa.json').read_text());assert g['success'] and not g['error'] and len(g['cases'])==10 and all(c['success'] for c in g['cases']) and g['executable_sha256']==stagehash
pkgapp=Path(g['executable']).parents[2]
assert all(not(p.stat().st_mode&0o222) for p in pkgapp.rglob('*') if p.is_file() and not p.is_symlink())
g.update(version='2.2.0',errors=[],details={'deb_sha256':dh,'deb_bytes':deb.stat().st_size,'packaged_binary_sha256':stagehash,'application_tree_read_only':True,'scope':'Actual final DEB executable through Qt public controls, real CPU inference, native target tables and native saved results; separate QA instrumentation only.'})
(release/'packaged-scope-filter-qa.json').write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n')
driver=ev/'gui-driver';driver.mkdir(exist_ok=True)
proto=Path('/tmp/vision-studio-gui-qa-220');p=proto/'README.md';s=p.read_text().replace('Stop an 11-image tail starting at row 51','Stop a 51-image tail starting at row 11');p.write_text(s)
for n in ['gui_qa_preload.cpp','run_gui_qa.sh','README.md']:shutil.copy2(proto/n,driver/n)
shutil.copy2(raw/'raw.log',ev/'packaged-gui.log')
exports=raw/'data/results';actual=sorted(p for p in exports.iterdir() if p.is_file());assert len(list(exports.glob('*.json')))==83
manifest={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in actual};(ev/'packaged-export-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ev/'QA-ARTIFACTS.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in actual:
  if p.suffix in ['.json','.csv']:z.write(p,'gui-exports/'+p.name)
 png=next(exports.glob('*.png'));z.write(png,'gui-example/'+png.name)
 z.write(ev/'packaged-export-manifest.json','packaged-export-manifest.json')
for n in ['qa-report.json','install-qa.json','packaged-gpu-qa.json','source-bundle-qa.json','gpu-qa.json','conversion-qa.json','packaged-scope-filter-qa.json','source-freeze.json','qa-report.log','install-qa.log','packaged-gpu-qa.log','source-bundle-qa.log','build-deb.log','build-source.log','final-build.log','SHA256SUMS','SOURCE-SHA256SUMS']:
 shutil.copy2(release/n,ev/n)
shutil.copy2(ev/'ctest-qa.json',release/'ctest-qa.json')
print(json.dumps({'success':True,'deb_sha256':dh,'gui_cases':len(g['cases']),'real_gui_exports':83,'source_files':source['frozen_application_sources_byte_equal'],'embedded_sources':source['embedded_source_files_sha256_verified'],'gpu_checks':checks,'evidence_zip_bytes':(ev/'QA-ARTIFACTS.zip').stat().st_size}))
