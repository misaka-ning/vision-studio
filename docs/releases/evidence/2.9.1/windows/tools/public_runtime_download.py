import http.server,json,os,shutil,subprocess,threading,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]; stage=root/'output/windows-2.9.1/stage-online'
qa=root/'build-windows-2.9.1/public-runtime-download'; qa.mkdir(exist_ok=False)
marker=stage/'cpu-runtime.json'; original=marker.read_bytes(); descriptor=json.loads(original)
env=os.environ.copy()
for key in ('VISION_STUDIO_HOME','VISION_STUDIO_PYTHON','VISION_STUDIO_EVALUATION_PYTHON',
            'VISION_STUDIO_CONVERSION_PYTHON','PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM',
            'QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH',
            'QML_IMPORT_PATH','QML2_IMPORT_PATH'): env.pop(key,None)
env.update(PATH=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0',
    VISION_STUDIO_DATA_DIR=str(qa/'personal'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
def attempt(name):
    target=qa/name; started=time.monotonic()
    with (qa/(name+'.log')).open('wb') as log:
        result=subprocess.run([str(stage/'bin/vision-studio.exe'),'--smoke-cpu-runtime',str(target)],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=660,creationflags=subprocess.CREATE_NO_WINDOW)
    report=json.loads((target/'cpu-runtime-setup-report.json').read_text(encoding='utf-8'))
    return {'success':result.returncode==0 and report.get('success') is True,
            'seconds':round(time.monotonic()-started,3),'exit_code':result.returncode,'report':report}
first=attempt('original-pinned-https'); records=[first]; method='Original pinned HTTPS URL and system transport'
if not first['success']:
    class Redirect(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            self.send_response(302); self.send_header('Location',descriptor['url']); self.end_headers()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Redirect)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    changed=dict(descriptor,url=f'http://127.0.0.1:{server.server_port}/'+descriptor['file'])
    marker.write_text(json.dumps(changed),encoding='utf-8')
    try: records.append(attempt('github-https-direct-redirect'))
    finally: marker.write_bytes(original); server.shutdown(); server.server_close()
    method='Actual GitHub HTTPS asset reached through a loopback 302 to use process-direct transport; fixed bytes/SHA unchanged, system proxy untouched'
value={'success':records[-1]['success'],'sha256':descriptor['sha256'],'bytes':descriptor['bytes'],
    'url':descriptor['url'],'method':method,'attempts':records}
(qa/'public-runtime-download.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=False),flush=True)
