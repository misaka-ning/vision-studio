import importlib.util, unittest, tempfile, hashlib, json, types
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('qa', '/home/misaka/.codex/worktrees/ef0b/yolo/scripts/qa_detection_scope_220.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RunnerTests(unittest.TestCase):
 def test_measured_success(self):
  totals,skipped,count,failed=m.inspect_output('Totals: 22 passed, 0 failed, 0 skipped, 0 blacklisted','<testsuite><testcase/></testsuite>')
  self.assertEqual(totals[0]['passed'],22);self.assertFalse(skipped);self.assertEqual(count,1);self.assertFalse(failed)
 def test_internal_skip_is_visible(self):
  self.assertTrue(m.inspect_output('Totals: 4 passed, 0 failed, 1 skipped, 0 blacklisted','<testsuite><testcase/></testsuite>')[1])
 def test_blacklist_is_visible(self):
  self.assertTrue(m.inspect_output('Totals: 4 passed, 0 failed, 0 skipped, 1 blacklisted','<testsuite><testcase/></testsuite>')[1])
 def test_qt_failure_is_visible(self):
  self.assertTrue(m.inspect_output('Totals: 4 passed, 1 failed, 0 skipped, 0 blacklisted','<testsuite><testcase/></testsuite>')[3])
 def test_xml_failure_is_visible(self):
  self.assertTrue(m.inspect_output('','<testsuite><testcase><failure/></testcase></testsuite>')[3])
 def test_xml_skip_is_visible(self):
  self.assertTrue(m.inspect_output('','<testsuite><testcase><skipped/></testcase></testsuite>')[1])
 def test_missing_case_is_visible(self):
  self.assertEqual(m.inspect_output('',None)[2],0)
 def test_real_child_capture_and_hash(self):
  code,text,seconds=m.execute(['/usr/bin/python3','-c','print("tooling selftest child")'],environment={},timeout=5)
  self.assertEqual(code,0);self.assertEqual(text,'tooling selftest child\n');self.assertGreaterEqual(seconds,0)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'artifact';p.write_bytes(b'tooling selftest\n');self.assertEqual(m.sha256(p),hashlib.sha256(b'tooling selftest\n').hexdigest())
 def test_cli_retains_venv_entry_and_feature_records_with_mocked_ctest(self):
  with tempfile.TemporaryDirectory(prefix='qa-runner-tool-unit-') as td:
   root=Path(td);build=root/'build';build.mkdir();(build/'CTestTestfile.cmake').write_text('')
   output=root/'output';captured={}
   def simulated_ctest(command,*,environment,timeout):
    captured.update(environment)
    (output/'vision-tooling-unit.xml').write_text('<testsuite><testcase/></testsuite>')
    return 0,'Totals: 2 passed, 0 failed, 0 skipped, 0 blacklisted\n1: QINFO : test CLASS_FILTER_EVIDENCE {"class_ids":[32]}\n1: QINFO : test ORT_PROFILE_EVIDENCE {"cuda_node_events":3}\n1: QINFO : test detection_scope_evidence: {"indices":[4]}\n',.01
   listing=types.SimpleNamespace(stdout=json.dumps({'tests':[{'name':'vision-tooling-unit'}]}))
   args=['qa','--build-dir',str(build),'--output-dir',str(output),'--test','vision-tooling-unit','--python','/opt/VisionStudio/runtime/bin/python']
   with patch('sys.argv',args),patch.object(m.subprocess,'run',return_value=listing),patch.object(m,'execute',side_effect=simulated_ctest):
    self.assertEqual(m.main(),0)
   self.assertEqual(captured['VISION_STUDIO_PYTHON'],'/opt/VisionStudio/runtime/bin/python')
   self.assertNotEqual(captured['VISION_STUDIO_PYTHON'],str(Path('/opt/VisionStudio/runtime/bin/python').resolve()))
   self.assertEqual(captured['VISION_STUDIO_QA_ARTIFACTS'],str(output/'artifacts'))
   report=json.loads((output/'detection-regression-qa.json').read_text())
   self.assertEqual([x['kind'] for x in report['runs'][0]['feature_evidence']],['CLASS_FILTER_EVIDENCE','ORT_PROFILE_EVIDENCE','detection_scope_evidence'])
 def test_python_unittest_positive_skip_summaries(self):
  for text in ['OK (skipped=1)', '14: OK (skipped=12)', '14: test_original_runtime ... skipped \'Original inference-only runtime is unavailable\'', '1 skipped in 1.2s', '4: summary skipped = 1']:
   with self.subTest(text=text):self.assertTrue(m.inspect_output(text,'<testsuite><testcase/></testsuite>')[1])
 def test_zero_skip_summaries_are_complete(self):
  for text in ['OK', 'OK (skipped=0)', 'Totals: 22 passed, 0 failed, 0 skipped, 0 blacklisted', '0 skipped in 1.2s']:
   with self.subTest(text=text):self.assertFalse(m.inspect_output(text,'<testsuite><testcase/></testsuite>')[1])
 def test_real_unittest_skip_returns_zero_but_is_incomplete(self):
  child='import unittest\nclass TestSkip(unittest.TestCase):\n @unittest.skip("tool selftest")\n def test_skip(self): pass\nunittest.main(verbosity=2)\n'
  code,text,seconds=m.execute(['/usr/bin/python3','-c',child],environment={},timeout=5)
  self.assertEqual(code,0,text);self.assertIn('skipped=1',text)
  self.assertTrue(m.inspect_output(text,'<testsuite><testcase/></testsuite>')[1])
 def test_invalid_xml_rejects_parse(self):
  with self.assertRaises(m.ET.ParseError):m.inspect_output('','not XML')
if __name__=='__main__':
 print('QA TOOLING SELF-TEST ONLY. CTest is mocked in one CLI unit case; no application/model acceptance is claimed.')
 unittest.main(verbosity=2)
