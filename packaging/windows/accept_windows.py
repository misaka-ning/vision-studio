#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Accept final installers in isolated folders, preserving the existing app registration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


class Registration:
    """Only the exact per-user Vision Studio uninstall key is changed/restored."""
    path = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\{02E34A30-7577-45F3-A6CD-420EAA1D7021}_is1'

    def __enter__(self):
        import winreg as r
        start=Path(os.environ['APPDATA'])/'Microsoft/Windows/Start Menu/Programs/Vision Studio'
        self.shortcuts={p:(p.read_bytes() if p.is_file() else None) for p in
            (start/'Vision Studio.lnk',start/'卸载 Vision Studio.lnk',start/'Uninstall Vision Studio.lnk')}
        self.saved = None
        try:
            with r.OpenKey(r.HKEY_CURRENT_USER, self.path, 0, r.KEY_READ | r.KEY_WOW64_64KEY) as key:
                children, values, _ = r.QueryInfoKey(key)
                if children: raise RuntimeError('Existing app registration has unexpected subkeys; preserved without QA mutation')
                self.saved = [r.EnumValue(key, i) for i in range(values)]
            r.DeleteKeyEx(r.HKEY_CURRENT_USER, self.path, r.KEY_WOW64_64KEY)
        except FileNotFoundError: pass
        return self

    def __exit__(self, *args):
        import winreg as r
        try: r.DeleteKeyEx(r.HKEY_CURRENT_USER, self.path, r.KEY_WOW64_64KEY)
        except FileNotFoundError: pass
        if self.saved is not None:
            with r.CreateKeyEx(r.HKEY_CURRENT_USER, self.path, 0, r.KEY_WRITE | r.KEY_WOW64_64KEY) as key:
                for name, value, kind in self.saved: r.SetValueEx(key, name, 0, kind, value)
        for path,value in self.shortcuts.items():
            if value is None:
                if path.is_file(): path.unlink()
            else:
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(value)


class Acceptance:
    def __init__(self, args):
        self.a = args; self.root = args.qa_root.resolve()
        if self.root.exists(): raise RuntimeError('Use a new dedicated QA directory')
        self.root.mkdir(parents=True)
        self.install = self.root/'安装路径 Space/VisionStudio'
        self.data = self.root/'personal'; self.data.mkdir()
        self.checks = []; self.modules = set()
        self.env = os.environ.copy()
        for key in ('VISION_STUDIO_HOME','VISION_STUDIO_PYTHON','VISION_STUDIO_SYSTEM_INSTALL','PYTHONHOME','PYTHONPATH',
                    'QT_PLUGIN_PATH','QT_QPA_PLATFORM','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH',
                    'QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH','QML_IMPORT_PATH','QML2_IMPORT_PATH'):
            self.env.pop(key, None)
        self.env.update(PATH=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0',
            VISION_STUDIO_DATA_DIR=str(self.data), VISION_STUDIO_GPU_RUNTIME_DIR=str(args.gpu_runtime.resolve()),
            PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
            HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9',
            NO_PROXY='127.0.0.1,localhost', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for name in ('yolo-config','matplotlib'):
            (self.data/name).mkdir()
        self.env['YOLO_CONFIG_DIR'] = str(self.data/'yolo-config')
        self.env['MPLCONFIGDIR'] = str(self.data/'matplotlib')

    def check(self, name, **details):
        self.checks.append(dict(name=name, passed=True, **details))
        print('PASS', name, flush=True)

    def run(self, command, name, timeout=600, env=None):
        started=time.monotonic()
        with (self.root/(name+'.log')).open('wb') as log:
            process=subprocess.Popen([str(x) for x in command], env=env or self.env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            # Observe the installed process, never the controller's Python modules.
            try:
                import psutil
                time.sleep(.3)
                for row in psutil.Process(process.pid).memory_maps(grouped=False): self.modules.add(row.path)
            except (ImportError, OSError, psutil.Error if 'psutil' in locals() else OSError): pass
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); raise RuntimeError(name+' timed out')
        if code: raise RuntimeError(f'{name} exited {code}; see {name}.log')
        return round(time.monotonic()-started,3)

    def setup(self, name, target=None, installer=None, env=None):
        command=[installer or self.a.installer,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-',
                 '/NOICONS','/MERGETASKS=!desktopicon', '/GROUP=Vision Studio QA '+self.root.name,
                 '/LANG=chinesesimplified', '/LOG='+str(self.root/(name+'-setup.log'))]
        if target is not None: command.append('/DIR='+str(target))
        self.run(command,name,env=env)

    def uninstall(self, name, target=None):
        target=target or self.install
        files=sorted(target.glob('unins*.exe'))
        if len(files)!=1: raise RuntimeError('Expected exactly one current uninstaller')
        self.run([files[0],'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/LOG='+str(self.root/(name+'-setup.log'))],name)

    def smoke(self, option, name, extra=(), report=None):
        destination=self.root/name
        seconds=self.run([self.install/'bin/vision-studio.exe', option, destination,
            '--smoke-input-root', self.a.fixtures, *extra],name)
        filename=report or {'--smoke-model':'model-display-report.json','--smoke-conversion':'conversion-smoke-report.json',
            '--smoke-cpu-runtime':'cpu-runtime-setup-report.json','--smoke-evaluation':'evaluation-smoke-report.json',
            '--smoke-dataset-session':'dataset-session-smoke-report.json','--smoke-preview-drop':'preview-drop-smoke-report.json'}.get(option,'smoke-report.json')
        value=json.loads((destination/filename).read_text(encoding='utf-8-sig'))
        if value.get('success') is not True: raise RuntimeError(name+' did not report success: '+str(value)[:1200])
        self.check(name, seconds=seconds, report=str((destination/filename).relative_to(self.root)))
        return value

    def payload(self):
        records=json.loads((self.install/'WINDOWS-FILES.json').read_text())
        def verify(record):
            target=self.install/record['path']
            if not target.is_file() or target.stat().st_size!=record['bytes'] or sha(target)!=record['sha256']:
                raise RuntimeError('Wrong installed payload: '+record['path'])
        with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(verify,records))
        self.check('all-payload-paths-and-sizes', files=len(records), sha256_files=len(records))
        for forbidden in ('sources','docs','models','assets'):
            if (self.install/forbidden).exists(): raise RuntimeError('Unrelated installed folder: '+forbidden)
        self.check('no-source-docs-examples')

    def prepare_online(self):
        requests=[]; archive=self.a.runtime_archive.resolve()
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                if self.path!='/'+archive.name: self.send_error(404); return
                requests.append(self.path); self.send_response(200)
                self.send_header('Content-Length',str(archive.stat().st_size)); self.end_headers()
                with archive.open('rb') as stream: shutil.copyfileobj(stream,self.wfile,1024*1024)
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        marker=self.install/'cpu-runtime.json'; original=marker.read_bytes(); descriptor=json.loads(original)
        if descriptor['sha256']!=sha(archive) or descriptor['bytes']!=archive.stat().st_size: raise RuntimeError('Unpinned QA runtime')
        descriptor['url']=f'http://127.0.0.1:{server.server_port}/'+archive.name
        marker.write_text(json.dumps(descriptor),encoding='utf-8')
        try:
            self.smoke('--smoke-cpu-runtime','online-runtime-preparation')
        finally:
            marker.write_bytes(original); server.shutdown(); server.server_close()
        if len(requests)!=1: raise RuntimeError('Expected one pinned CPU download')
        self.smoke('--smoke-cpu-runtime','online-runtime-offline-reuse')
        self.cpu=self.data/'runtimes/cpu'/descriptor['sha256']/'python.exe'

    def acceptance(self):
        # Exercise the real USERPROFILE default without replacing the maintainer's
        # existing installation. Registration() temporarily hides only this AppId.
        isolated_profile=self.root/'isolated-user-profile'; isolated_profile.mkdir()
        environment=dict(self.env,USERPROFILE=str(isolated_profile))
        self.setup('install-default-userprofile',env=environment)
        default=isolated_profile/'VisionStudio'
        if not (default/'windows-install.json').is_file(): raise RuntimeError('USERPROFILE default directory did not expand')
        self.check('default-userprofile-directory',path=str(default),isolated_userprofile=True)
        self.uninstall('uninstall-default-userprofile',default)
        self.setup('install-custom-directory',self.install); self.check('install-custom-directory')
        marker=json.loads((self.install/'windows-install.json').read_text())
        if marker['profile']!=self.a.profile or marker['version']!='2.9.1': raise RuntimeError('Wrong installer profile/version')
        self.payload()
        if self.a.profile=='online':
            if (self.install/'runtime').exists(): raise RuntimeError('Online installer embeds Python')
            self.smoke('--smoke','online-no-runtime-onnx',('--device','cpu'))
            self.prepare_online()
        else: self.cpu=self.install/'runtime/python.exe'
        self.smoke('--smoke','installed-onnx-cpu',('--device','cpu'))
        self.smoke('--smoke-pt','installed-pt-cpu',('--device','cpu'))
        self.smoke('--smoke-model','installed-model-views',('--display-model',self.a.fixtures/'models/yolov5n.onnx'))
        self.smoke('--smoke-model','installed-pt-model-views',('--display-model',self.a.fixtures/'models/yolov8n.pt'))
        self.smoke('--smoke-conversion','installed-conversion-onnx',('--conversion-source',self.a.fixtures/'models/yolov8n.pt','--conversion-format','onnx'))
        self.smoke('--smoke-conversion','installed-conversion-torchscript',('--conversion-source',self.a.fixtures/'models/yolov5n.pt','--conversion-format','torchscript'))
        self.smoke('--smoke-evaluation','installed-evaluation',('--evaluation-request',self.a.evaluation_request))
        self.smoke('--smoke-dataset-session','installed-dataset-session')
        self.smoke('--smoke-preview-drop','installed-preview-drop',('--preview-drop-request',self.a.preview_request))
        self.smoke('--smoke','installed-onnx-cuda',('--device','cuda'))
        self.smoke('--smoke-pt','installed-pt-cuda',('--device','cuda'))
        if self.a.recording_test:
            tool=self.root/'recording-tool'; tool.mkdir()
            target=tool/'vision-recording-tests.exe'; shutil.copy2(self.a.recording_test,target)
            shutil.copy2(self.a.qt_test_dll,tool/'Qt6Test.dll')
            (tool/'qt.conf').write_text('[Paths]\nPrefix='+str((self.install/'bin').as_posix())+'\nPlugins=.\n',encoding='utf-8')
            environment=dict(self.env,VISION_STUDIO_HOME=str(self.install),VISION_STUDIO_PYTHON=str(self.cpu),
                             VISION_STUDIO_SYSTEM_INSTALL='1',VISION_STUDIO_GPU_TESTS='1',QT_QPA_PLATFORM='windows',
                             PATH=str(self.install/'bin')+';'+self.env['PATH'])
            self.run([target,'-o',str(self.root/'installed-recording-qtest.txt')+',txt'],'installed-recording',env=environment)
            self.check('installed-recording', dependency_root=str(self.install/'bin'),
                       extra_test_only_dependency='Qt6Test.dll in the isolated QA tool folder; never packaged')
        if self.a.profile=='offline':
            self.check('offline-cpu-without-network',method='External Python download proxies blocked; Qt model browser limited to loopback; no dependency preparation',physical_network_disconnected=False)
        forbidden=('d:/qt/','d:/opencv-c++/','e:/vision-studio/build-windows/python/')
        bad=[m for m in self.modules if m.replace('\\','/').lower().startswith(forbidden)]
        if bad: raise RuntimeError('Loaded developer dependencies: '+str(bad))
        if not self.modules: raise RuntimeError('No module observations captured for dependency isolation')
        dump(self.root/'loaded-module-paths.json',sorted(self.modules))
        self.check('isolated-developer-paths',observed_modules=len(self.modules))
        sentinel=self.install/'preserve-imported-user-model.pt'; sentinel.write_bytes(b'private user model preserved')
        private=self.data/'preserve-external-user-model.pt'; private.write_bytes(b'private external model preserved')
        protected={p:sha(p) for p in (sentinel,private,self.data/'preferences.ini',self.data/'history.json') if p.exists()}
        self.setup('upgrade-reinstall',self.install); self.check('upgrade-reinstall')
        if any(not p.exists() or sha(p)!=v for p,v in protected.items()): raise RuntimeError('Upgrade changed personal data')
        self.check('upgrade-preserves-user-files')
        self.uninstall('uninstall'); self.check('uninstall')
        if (self.install/'bin/vision-studio.exe').exists(): raise RuntimeError('Uninstall left app executable')
        if any(not p.exists() or sha(p)!=v for p,v in protected.items()): raise RuntimeError('Uninstall changed personal data')
        self.check('uninstall-preserves-user-files')
        return {'schema_version':2,'success':True,'profile':self.a.profile,'version':'2.9.1',
            'sha256':sha(self.a.installer),'checks':self.checks,
            'limitations':['Current Windows 11 host with SDK/Python paths isolated and DLL paths audited; no clean VM or Windows 10 test.',
                'External download proxies were blocked for CPU acceptance; the physical host network remained connected.',
                'Desktop icon task disabled; isolated Start Menu group used; existing app registration and Start Menu links restored.',
                'No physical webcam tested; real synthetic-video and recording tests use installed DLLs.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('installer','qa-root','fixtures','gpu-runtime','runtime-archive','evaluation-request','preview-request'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('offline','online'),required=True)
    p.add_argument('--recording-test',type=Path)
    p.add_argument('--qt-test-dll',type=Path)
    a=p.parse_args()
    job=Acceptance(a)
    try:
        with Registration(): report=job.acceptance()
    except Exception as error:
        dump(job.root/'installer-qa.json',{'success':False,'profile':a.profile,'sha256':sha(a.installer),
            'checks':job.checks,'error':str(error)})
        raise
    dump(job.root/'installer-qa.json',report)


if __name__=='__main__': main()
