# SPDX-License-Identifier: AGPL-3.0-only
"""Payload selection and adversarial installer/runtime inputs, using temporary trees."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('windows_builder', ROOT/'packaging/windows/build_windows.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PayloadTests(unittest.TestCase):
    def test_runtime_modules_and_data_are_retained(self):
        for path in ('python.exe', 'python311.dll', 'Lib/site-packages/torch/_dynamo/__init__.py',
                     'Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml',
                     'Lib/site-packages/pip/__main__.py', 'Lib/site-packages/torch-2.9.1.dist-info/METADATA',
                     'Lib/site-packages/numpy.libs/libopenblas.dll'):
            self.assertFalse(builder.runtime_excluded(path), path)

    def test_only_known_development_material_is_omitted(self):
        for path in ('include/Python.h', 'libs/python311.lib', 'Scripts/pip.exe', 'Lib/test/test_os.py',
                     'Lib/site-packages/torch/include/torch.h', 'Lib/site-packages/pandas/tests/test_frame.py',
                     'Lib/site-packages/numpy/_core/tests/test_numeric.py', 'Lib/__pycache__/os.pyc'):
            self.assertTrue(builder.runtime_excluded(path), path)

    def test_latest_evaluation_helpers_and_native_extractor_are_shipped(self):
        expected = {'evaluation_worker.py', 'evaluation_data.py', 'evaluation_metrics.py',
                    'evaluation_import.py', 'evaluation_snapshot.py', 'evaluation_reporting.py',
                    'prepare_cpu_runtime.ps1', 'pt_worker.py', 'netron_server.py', 'model_convert.py'}
        self.assertTrue(expected.issubset(builder.RUNTIME_SCRIPTS))
        for name in builder.RUNTIME_SCRIPTS: self.assertTrue((ROOT/'scripts'/name).is_file(), name)


@unittest.skipUnless(sys.platform == 'win32', 'Windows PowerShell extraction and upgrade semantics')
class NativeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='vision-windows-safety-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def powershell(self, script, **parameters):
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                   '-File', str(ROOT/script)]
        for key,value in parameters.items():
            command.append('-'+key)
            if value is not True: command.append(str(value))
        return subprocess.run(command, capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)

    def extraction(self, members, expected_error):
        archive = self.root/'cpu.zip'
        with zipfile.ZipFile(archive,'w') as output:
            for name,data in members: output.writestr(name,data)
        stage = self.root/('.staging-'+str(uuid.uuid4()))
        result = self.powershell('scripts/prepare_cpu_runtime.ps1', Archive=archive, Destination=stage,
                                ExpectedSha256=hashlib.sha256(archive.read_bytes()).hexdigest(), ExpectedBytes=archive.stat().st_size)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(expected_error.encode(), result.stderr, result.stderr.decode(errors='replace'))
        return stage

    def test_extraction_rejects_traversal_before_writing(self):
        stage = self.extraction([('../escape.txt', b'never')], 'Unsafe runtime path')
        self.assertFalse(stage.exists()); self.assertFalse((self.root/'escape.txt').exists())

    def test_extraction_rejects_case_collisions(self):
        self.assertFalse(self.extraction([('module.py',b'a'),('MODULE.PY',b'b')], 'Duplicate or linked runtime member').exists())

    def test_extraction_rejects_ntfs_alternate_stream(self):
        self.assertFalse(self.extraction([('file.txt:stream',b'a')], 'Unsafe runtime path').exists())

    def test_extraction_rejects_bad_component_inventory(self):
        inventory = {'schema_version':1,'files':[{'path':'python.exe','bytes':5,'sha256':'a'*64}]}
        stage = self.extraction([('python.exe',b'wrong'),('CPU-FILES.json',json.dumps(inventory).encode())], 'CPU component checksum mismatch')
        self.assertFalse((stage/'ready.json').exists())

    def cleanup_fixture(self, unsafe=False):
        install = self.root/'安装 Space'; install.mkdir()
        (install/'windows-install.json').write_text('{"version":"2.1.0"}')
        records=[]
        for name in ('docs/old.txt','sources/changed.txt','preserve-user-model.pt'):
            path=install/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'original')
            if name != 'preserve-user-model.pt':
                records.append({'path':name,'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()})
        (install/'sources/changed.txt').write_bytes(b'user modification')
        if unsafe: records.append({'path':'../escape.txt','bytes':0,'sha256':'0'*64})
        (install/'WINDOWS-FILES.json').write_text(json.dumps(records))
        incoming=self.root/'incoming.json'; incoming.write_text('[]')
        runtime=self.root/'runtime.json'; runtime.write_text('{}')
        result=self.powershell('packaging/windows/cleanup_install.ps1',InstallRoot=install,
                               IncomingManifest=incoming,RuntimeDescriptor=runtime)
        return install,result

    def test_upgrade_removes_only_unchanged_owned_files(self):
        install,result=self.cleanup_fixture()
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        self.assertFalse((install/'docs/old.txt').exists())
        self.assertEqual((install/'sources/changed.txt').read_bytes(),b'user modification')
        self.assertEqual((install/'preserve-user-model.pt').read_bytes(),b'original')

    def test_upgrade_validates_all_paths_before_deleting(self):
        install,result=self.cleanup_fixture(unsafe=True)
        self.assertNotEqual(result.returncode,0)
        self.assertTrue((install/'docs/old.txt').exists())
        self.assertEqual((install/'sources/changed.txt').read_bytes(),b'user modification')

    def test_online_uninstall_cleans_retained_runtime_but_preserves_user_file(self):
        install=self.root/'installed'; runtime=install/'runtime'; runtime.mkdir(parents=True)
        (install/'windows-install.json').write_text('{"version":"2.9.1"}')
        managed=runtime/'module.py'; managed.write_bytes(b'managed runtime')
        personal=runtime/'user-model.pt'; personal.write_bytes(b'personal model')
        inventory=runtime/'CPU-FILES.json'
        inventory.write_text(json.dumps({'schema_version':1,'files':[
            {'path':'module.py','bytes':managed.stat().st_size,'sha256':hashlib.sha256(managed.read_bytes()).hexdigest()}]}))
        descriptor=self.root/'descriptor.json'
        descriptor.write_text(json.dumps({'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest()}))
        result=self.powershell('packaging/windows/cleanup_install.ps1',InstallRoot=install,
                               RuntimeDescriptor=descriptor,RuntimeOnly=True)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        self.assertFalse(managed.exists()); self.assertFalse(inventory.exists())
        self.assertEqual(personal.read_bytes(),b'personal model')


if __name__ == '__main__': unittest.main()
