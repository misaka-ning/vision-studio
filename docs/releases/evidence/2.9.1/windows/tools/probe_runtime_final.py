import functools, http.server, json, os, pathlib, shutil, subprocess, threading, time
root=pathlib.Path(__file__).resolve().parents[1]
out=root/'output/windows-2.9.1'; stage=out/'stage-online'
qa=root/'build-windows-2.9.1/component-final-qa'; qa.mkdir(exist_ok=True)
requests=[]
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        requests.append(self.path); super().do_GET()
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(out)))
threading.Thread(target=server.serve_forever,daemon=True).start()
shutil.copy2(root/'build-windows-2.9.1/bin/vision-studio.exe',stage/'bin/vision-studio.exe')
shutil.copy2(root/'scripts/prepare_cpu_runtime.ps1',stage/'scripts/prepare_cpu_runtime.ps1')
path=stage/'cpu-runtime.json'; original=path.read_bytes(); descriptor=json.loads(original)
descriptor['url']=f'http://127.0.0.1:{server.server_port}/'+descriptor['file']
path.write_text(json.dumps(descriptor),encoding='utf-8')
env=os.environ.copy()
for key in ('VISION_STUDIO_HOME','VISION_STUDIO_PYTHON','PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM',
            'QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH',
            'QML_IMPORT_PATH','QML2_IMPORT_PATH'): env.pop(key,None)
env.update(PATH='C:/Windows/System32;C:/Windows;C:/Windows/System32/WindowsPowerShell/v1.0',
           VISION_STUDIO_DATA_DIR=str(qa/'personal'),PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
try:
    started=time.monotonic()
    with (qa/'process.log').open('wb') as log, (qa/'concurrent-process.log').open('wb') as log2:
        first=subprocess.Popen([str(stage/'bin/vision-studio.exe'),'--smoke-cpu-runtime',str(qa/'prepare')],
            env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        second=subprocess.Popen([str(stage/'bin/vision-studio.exe'),'--smoke-cpu-runtime',str(qa/'concurrent')],
            env=env,stdout=log2,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        code=first.wait(timeout=900); code2=second.wait(timeout=900)
    report=json.loads((qa/'prepare/cpu-runtime-setup-report.json').read_text(encoding='utf-8'))
    concurrent=json.loads((qa/'concurrent/cpu-runtime-setup-report.json').read_text(encoding='utf-8'))
    server.shutdown()
    with (qa/'offline-reuse-process.log').open('wb') as log:
        reused=subprocess.run([str(stage/'bin/vision-studio.exe'),'--smoke-cpu-runtime',str(qa/'offline-reuse')],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    offline=json.loads((qa/'offline-reuse/cpu-runtime-setup-report.json').read_text(encoding='utf-8'))
    result={'success':code==0 and code2==0 and report['success'] and concurrent['success'] and reused.returncode==0,
            'exit_code':code,'concurrent_exit_code':code2,'single_download':len(requests)==1,'offline_reuse':offline,
            'seconds':round(time.monotonic()-started,2),'requests':requests,'report':report}
    (qa/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
finally:
    path.write_bytes(original); server.shutdown()
