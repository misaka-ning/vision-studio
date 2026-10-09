#!/usr/bin/env python3
"""Bind unchanged baseline suites and the actual affected final replays."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
EVID = ROOT/'docs/releases/evidence/2.1.0'
OLD = EVID/'candidate-before-output-comparison'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p): return {'bytes':p.stat().st_size,'sha256':sha(p)}
def require(ok, message):
    if not ok: raise RuntimeError(message)
def same_file(p, item): return p.is_file() and identity(p) == {'bytes':item['bytes'],'sha256':item['sha256']}
def raw_status(text, number, name):
    return re.search(r'(?m)^\s*\d+/\d+\s+Test\s+#'+str(number)+r':\s+'+re.escape(name)+r'\s+\.+\s+Passed\s+',text)
def qt_counts(text, number):
    match = re.search(r'(?m)^'+str(number)+r': Totals: (\d+) passed, (\d+) failed, (\d+) skipped, (\d+) blacklisted, (\d+)ms',text)
    require(match is not None, f'Missing actual Qt totals for suite {number}')
    return tuple(map(int,match.groups()))

old = read(OLD/'ctest-qa.json')
old_library = read(OLD/'model-library-conversion-2.1-final-report.json')
baseline = read(EVID/old['execution'])
baseline_log = EVID/old['log']
require(sha(baseline_log)==old['log_sha256'] and baseline['exit_code']==8,
        'Original failing full baseline was modified')
original_candidate = read(EVID/old['original_full_run']['summary_report'])
require(original_candidate['registered_suites']==13 and original_candidate['all_suites_passed']==12
        and original_candidate['failed']==1 and not original_candidate['success'],
        'Original twelve-pass/one-failure baseline missing')
require(re.search(r'(?m)^\s*\d+/13\s+Test\s+#6:\s+vision-ui-tests\s+\.+\*\*\*Failed',baseline_log.read_text()),
        'Original UI failure raw record missing')
backend = read(EVID/'backend-qa.json')
require(backend['success'] and backend['tests']['passed']==20 and backend['tests']['failed']==0
        and backend['tests']['skipped']==0 and len(backend['actual_conversions'])>=18,
        'Actual final backend report missing')
for relative, expected in backend['source_files'].items():
    require(sha(ROOT/relative)==expected,'Current backend differs from its actual execution: '+relative)

descriptions = [
    {'log':'ctest-final-replay.log','execution':'ctest-final-replay.log.execution.json',
     'accepted_suite_numbers':[6], 'validated_keys':[
         'build-2.1.0/bin/vision-studio','build-2.1.0/vision-ui-tests',
         'src/ui/mainwindow.cpp','src/ui/modelconversionpage.cpp','assets/fluent-dark.qss','tests/ui_tests.cpp'],
     'explanation':'Only the UI suite remains accepted from this earlier actual three-suite replay. Its application, UI test binary and UI sources are unchanged. The earlier converter-related two suites are superseded by the two newer replays.'},
    {'log':'conversion-portability-proof/ctest-full-model-conversion.log',
     'execution':'conversion-portability-proof/ctest-full-model-conversion.log.execution.json',
     'accepted_suite_numbers':[10], 'validated_keys':None,
     'explanation':'All twelve real conversion-to-MainWindow-to-VisionEngine data rows, including full640 modern/legacy ONNX CPU/CUDA and TorchScript CUDA, run on the final frozen converter and expanded test binary.'},
    {'log':'conversion-portability-proof/ctest-backend-conversion-final.log',
     'execution':'conversion-portability-proof/ctest-backend-conversion-final.log.execution.json',
     'accepted_suite_numbers':[11], 'validated_keys':None,
     'explanation':'All twenty actual backend safety/conversion tests replayed with the final frozen converter; eighteen exports include both full640 generations and real system OpenCV4.5.4 numerical checks.'}
]
replays = []
texts = {old['log']:baseline_log.read_text()}
frozen = {}
for description in descriptions:
    log = EVID/description['log']
    execution = read(EVID/description['execution'])
    require(execution['exit_code']==0 and execution['unchanged'] is True
            and execution['files_before']==execution['files_after'],'Replay mutation/failure: '+description['log'])
    logged = execution.get('log')
    if logged:
        require(sha(log)==logged['sha256'] and log.stat().st_size==logged['bytes'],
                'Replay raw log differs from executed record: '+description['log'])
    else:
        require(description['log']==old['final_replay']['log']
                and sha(log)==old['final_replay']['log_sha256'],
                'Earlier replay raw log differs from preserved candidate: '+description['log'])
    keys = description['validated_keys'] or list(execution['files_after'])
    validated = {key:execution['files_after'][key] for key in keys}
    for relative, item in validated.items():
        require(same_file(ROOT/relative,item),'Accepted replay dependency changed: '+relative)
        if relative in frozen:
            require(frozen[relative]==item,'Conflicting accepted source/binary bindings: '+relative)
        frozen[relative] = item
    text = log.read_text()
    summary = re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)',text)
    require(summary and int(summary[1])==100 and int(summary[2])==0,'Replay completion failure: '+str(log))
    replays.append({k:v for k,v in description.items() if k!='validated_keys'} | {
        'log_sha256':sha(log),'registered_suites':int(summary[3]),'failed':int(summary[2]),
        'elapsed_seconds':execution['elapsed_seconds'],'validated_current_files':validated})
    texts[description['log']] = text

qt = [dict(row) for row in old['qt_suites']]
python = [dict(row) for row in old['python_suites']]
qt10 = next(row for row in qt if row['ctest_number']==10)
full_log = descriptions[1]['log']
passed, failed, skipped, blacklisted, ms = qt_counts(texts[full_log],10)
qt10.update(passed=passed,failed=failed,skipped=skipped,blacklisted=blacklisted,elapsed_ms=ms,evidence_log=full_log)
functional_rows = re.findall(r'^10: PASS\s+: ConversionInferenceTests::convertedModelsUseTheirOwnTaskClassesChannelsAndPixels\(([^)]+)\)',texts[full_log],re.M)
require(len(functional_rows)==12 and passed==len(functional_rows)+2 and failed==skipped==blacklisted==0,
        'Twelve complete actual converter rows did not pass')
python11 = next(row for row in python if row['ctest_number']==11)
backend_log = descriptions[2]['log']
match = re.search(r'^11: Ran (\d+) tests in ([\d.]+)s',texts[backend_log],re.M)
require(match and re.search(r'^11: OK$',texts[backend_log],re.M),'Final backend CTest did not finish successfully')
python11.update(tests=int(match[1]),elapsed_seconds=float(match[2]),skipped=0,success=True,evidence_log=backend_log)
require(python11['tests']==backend['tests']['total']==20,'Final backend counts disagree')
binary_baseline = {row['file']:row for row in baseline['binaries']}
for row in qt+python:
    selected = row['evidence_log']
    text = texts[selected]
    require(raw_status(text,row['ctest_number'],row['name']),'Actual accepted final suite status missing: '+row['name'])
    if 'passed' in row:
        require(qt_counts(text,row['ctest_number'])[:4]==(row['passed'],0,0,0)
                and row['failed']==row['skipped']==row['blacklisted']==0,'Qt effective counts differ: '+row['name'])
        if row['ctest_number'] not in {6,10}:
            key='build-2.1.0/'+row['name']
            require(same_file(ROOT/key,binary_baseline[key]),'Unreplayed Qt binary changed: '+key)
            frozen[key] = {'bytes':binary_baseline[key]['bytes'],'sha256':binary_baseline[key]['sha256']}
    else:
        match = re.search(r'^'+str(row['ctest_number'])+r': Ran (\d+) tests in ',text,re.M)
        require(match and int(match[1])==row['tests'] and row['success'] and row['skipped']==0
                and re.search(r'^'+str(row['ctest_number'])+r': OK$',text,re.M),'Python effective counts differ: '+row['name'])
require(len(qt)+len(python)==13 and {row['ctest_number'] for row in qt+python}==set(range(1,14)),
        'All thirteen effective registered suite identities required')
app = binary_baseline['build-2.1.0/bin/vision-studio']
require(same_file(ROOT/app['file'],app),'Application changed after complete original build')
exports = [json.loads(match[1]) for match in re.finditer(r'CONVERSION_EVIDENCE (\{[^\n]+\})',texts[backend_log])]
require(len(exports)>=18 and all(x.get('source_unchanged') is True and x.get('job_removed') is True for x in exports),
        'Final eighteen actual exports not recovered from final CTest raw log')
qt_total = sum(row['passed'] for row in qt)
python_total = sum(row['tests'] for row in python)
report = dict(old)
for key in ('final_replay','replay_execution'):
    report.pop(key,None)
report.update(schema_version=3,success=True,errors=[],qt_suites=qt,python_suites=python,
    all_suites_passed=13,failed=0,failed_suite_names=[],qt_skipped=0,python_skipped=0,
    qt_passed=qt_total,python_tests=python_total,
    elapsed_seconds=round(baseline['elapsed_seconds']+sum(row['elapsed_seconds'] for row in replays),3),
    accepted_replays=replays,frozen_final_files=frozen,application_unchanged_from_full_build=True,final_application=app,
    actual_conversion_export_count=len(exports),actual_conversion_exports=exports,
    backend_report={'file':'backend-qa.json','sha256':sha(EVID/'backend-qa.json'),'tests':backend['tests'],
                    'source_files':backend['source_files']},
    superseded_candidate_report={'file':'candidate-before-output-comparison/ctest-qa.json',
                                 'sha256':sha(OLD/'ctest-qa.json')},
    failure_resolution='The original complete thirteen-suite execution had twelve successful suites and one UI-test failure caused by stale navigation-count and globally scoped wheel-control lookup assertions. Those test-only assumptions were fixed and the complete UI suite passed; its application and UI dependencies remain unchanged. A later targeted full640 bus numerical check exposed genuine converter defects: OpenCV4.5.4 reversed constant-left Sub, and TorchScript CPU-traced static detection caches/TF32 flags were not portable to CUDA. Those converter defects were fixed. Only the two affected conversion suites were then replayed on the final converter: twelve real C++ UI/Core data rows and twenty backend tests. Earlier converter positive-count passes are preserved as candidates and superseded by these numerical/device checks.',
    note='Effective acceptance is composed from the actual complete baseline plus three explicitly identified accepted replay sources. It is not a claim that one final full thirteen-suite command passed. UI remains 38 passing cases on unchanged UI source/binary; conversion integration now has twelve functional rows plus init/cleanup (14 Qt passes), and backend has 20 tests/18 real exports. Other ten unaffected actual baseline results are retained. All effective suites have zero failures/skips/blacklists. Actual CUDA tests were enabled. No physical camera was exercised.')
(EVID/'ctest-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

# Model-library integration has the unchanged complete UI evidence plus the new
# actual converter rows, rather than a retrospective hash replacement.
onnx = read(EVID/'conversion-portability-proof/portability-onnx-full-model-qa.json')
require(onnx['success'] and onnx['errors']==[],'Actual full-model ONNX numerical/device proof missing')
library = dict(old_library)
library['schema_version']=2
library['stage']='Unchanged complete UI validation plus final twelve-row CPU/CUDA conversion integration and full640 original-model numerical replay'
library['generated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
library['result']='passed'
library['success']=True
library['runtime']={**library['runtime'],'device':'CPU and actual NVIDIA CUDA','qt':'6.8.3','target_opencv':'4.5.4'}
library['raw_logs'] = [
    {'path':row['log'],'bytes':(EVID/row['log']).stat().st_size,'sha256':sha(EVID/row['log']),
     'result':'passed','scope':row['explanation']} for row in replays
]
library['raw_logs'] += [
    {'path':row['execution'],'bytes':(EVID/row['execution']).stat().st_size,'sha256':sha(EVID/row['execution']),
     'result':'passed','scope':'Actual command, isolated environment, frozen before/after files'} for row in replays
]
library['sha256'] = {path:sha(ROOT/path) for path in old_library['sha256']}
library['sha256']['tests/model_conversion_tests.py'] = sha(ROOT/'tests/model_conversion_tests.py')
library['conversion_cases'] = functional_rows
library['checks'] = old_library['checks'] + [
    'full640 real bus YOLOv8n and YOLOv5n ONNX CPU (system OpenCV4.5.4) and CUDA (ONNX Runtime) compare all output boxes, classes, labels and confidence to the selected original PT',
    'full640 real bus YOLOv8n and YOLOv5n TorchScript CUDA compare output boxes/classes/labels/confidence to original PT CPU and assert actual CUDA backend',
    'independent full640 ONNX actual CUDAExecutionProvider node traces and original PT CPU numerical comparisons'
]
library['limitations'] = [
    'Qt UI tests use the offscreen platform; a native context event is supplied after the test right-click because offscreen does not synthesize it.',
    'Opening the file manager is validated through a QDesktopServices URL capture rather than visually inspecting an external manager.',
    'The complete UI suite remains accepted from its earlier execution because all relevant UI/application source and binary hashes remain unchanged. Both converter suites were independently replayed after the late full-model portability fixes.',
    'No physical camera was tested. Earlier positive-count converter/package reports remain preserved in candidate-before-output-comparison and are superseded by the new numerical full-model checks.'
]
library['candidate_report'] = {'file':'candidate-before-output-comparison/model-library-conversion-2.1-final-report.json',
    'sha256':sha(OLD/'model-library-conversion-2.1-final-report.json'),
    'scope':'Earlier six-row conversion evidence on a superseded converter; preserved candidate history'}
library['final_actual_conversion_inference_suite']={**qt10,'functional_rows':functional_rows}
library['final_backend_suite']=python11
library['frozen_source_and_binary_binding']=frozen
library['accepted_replays']=replays
library['full_model_onnx_proof']={'file':'conversion-portability-proof/portability-onnx-full-model-qa.json',
    'sha256':sha(EVID/'conversion-portability-proof/portability-onnx-full-model-qa.json'),
    'comparisons':onnx['comparisons'],'cuda_node_counts':{row['name']:row['cuda_node_count'] for row in onnx['runs'] if 'cuda_node_count' in row}}
library['final_replay_explanation']=report['failure_resolution']+' The accepted raw evidence and current dependency bindings are listed for each replay separately.'
(EVID/'model-library-conversion-2.1-final-report.json').write_text(json.dumps(library,ensure_ascii=False,indent=2)+'\n')
release = ROOT/'output/releases/2.1.0'
require(release.is_dir(),'Release directory missing; do not invent one')
for name in ['ctest-qa.json','model-library-conversion-2.1-final-report.json']:
    shutil.copy2(EVID/name,release/name)
print(json.dumps({'success':True,'registered_suites':13,'effective_passed':13,'qt_passed':qt_total,
    'python_tests':python_total,'failed':0,'skipped':0,'converter_functional_rows':len(functional_rows),
    'converter_qt_passed':passed,'backend_tests':python11['tests'],'actual_exports':len(exports),
    'reports':{name:sha(EVID/name) for name in ['ctest-qa.json','model-library-conversion-2.1-final-report.json']}},indent=2))
