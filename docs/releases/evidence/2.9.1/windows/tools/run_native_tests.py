import json, os, pathlib, subprocess, time, concurrent.futures, sys
root=pathlib.Path(__file__).resolve().parents[1]
build=root/'build-windows-2.9.1'
out=build/'native-tests-final-3'; out.mkdir(exist_ok=True)
spec=json.loads(subprocess.check_output(['D:/Qt/bin/ctest.exe','--test-dir',str(build),'--show-only=json-v1']))
base=os.environ.copy()
base.update(PATH='D:/Qt/6.8.3/msvc2022_64/bin;D:/Opencv-C++/opencv/build/x64/vc16/bin;C:/Windows/System32;C:/Windows;C:/Windows/System32/WindowsPowerShell/v1.0',
    VISION_PUBLIC_COCO8_ROOT=str(build/"fixtures/coco8"),VISION_STUDIO_GPU_TESTS="1",VISION_STUDIO_HOME=str(root),VISION_STUDIO_EVALUATION_PYTHON=str(root/"build-windows/python/python.exe"),VISION_STUDIO_TEST_PYTHON=str(root/"build-windows/python/python.exe"),VISION_STUDIO_CONVERSION_PYTHON=str(root/"build-windows/python/python.exe"),VISION_STUDIO_OPENCV_PROBE=str(build/"vision-opencv-target-probe.exe"),VISION_STUDIO_PYTHON=str(root/'build-windows/python/python.exe'),
    VISION_STUDIO_GPU_RUNTIME_DIR=str(root/'build-windows/qa-gpu-runtime'),
    PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
native_gui={'vision-ui-tests','vision-batch-responsiveness-tests','vision-scope-filter-ui-tests',
 'vision-modelviewer-tests','vision-conversion-inference-tests','vision-input-drop-ui-tests',
 'vision-dataset-session-ui-tests','vision-evaluation-integration-tests'}
def execute(test):
    name=test['name']; env=base.copy(); folder=out/name; folder.mkdir(exist_ok=True)
    for d in ('data','ultralytics','matplotlib'): (folder/d).mkdir(exist_ok=True)
    env.update(VISION_STUDIO_DATA_DIR=str(folder/'data'),YOLO_CONFIG_DIR=str(folder/'ultralytics'),MPLCONFIGDIR=str(folder/'matplotlib'))
    for prop in test.get('properties',[]):
        if prop['name']=='ENVIRONMENT':
            for item in prop['value']:
                k,v=item.split('=',1); env[k]=v
    if name in native_gui: env['QT_QPA_PLATFORM']='windows'
    cmd=test['command'].copy()
    if pathlib.Path(cmd[0]).suffix.lower()=='.exe' and pathlib.Path(cmd[0]).name.startswith('vision-'):
        cmd+=['-o',str(folder/'qtest.txt')+',txt']
    if name=='vision-gpu-runtime-tests':
        return {'name':name,'status':'not-applicable','reason':'POSIX-specific GPU installer tests; Windows GPU has separate actual acceptance'}
    started=time.monotonic(); print('START',name,flush=True)
    try:
        with (folder/'process.log').open('wb') as log:
            r=subprocess.run(cmd,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
        result={'name':name,'exit_code':r.returncode,'passed':r.returncode==0,'seconds':round(time.monotonic()-started,3)}
    except subprocess.TimeoutExpired: result={'name':name,'passed':False,'reason':'900-second timeout'}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print('END',json.dumps(result),flush=True); return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    selected=[t for t in spec['tests'] if not sys.argv[1:] or t['name'] in sys.argv[1:]]
    results=list(pool.map(execute,selected))
(out/'summary.json').write_text(json.dumps({'success':all(r.get('passed',r.get('status')=='not-applicable') for r in results),'results':results},indent=2))
