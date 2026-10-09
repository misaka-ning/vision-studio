#!/usr/bin/env python3
"""Safety/protocol tests and actual CPU conversion/numerical regressions.

Set VISION_STUDIO_CONVERSION_PYTHON to the prepared conversion runtime. Real
tests use only local public models and isolated generated C1/C3 fixtures; they
never install dependencies, download assets or change the user's configuration.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parent.parent
HELPER = PROJECT / "scripts/model_convert.py"
spec = importlib.util.spec_from_file_location("conversion_helper", HELPER)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)
PYTHON = os.environ.get("VISION_STUDIO_CONVERSION_PYTHON") or os.environ.get("VISION_STUDIO_PYTHON") or next(
    (str(path) for path in (PROJECT / "build-2.1.0/release-runtime/bin/python3",
                           PROJECT / "build/release-runtime/bin/python3") if path.is_file()), sys.executable)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolated_environment(path):
    env = os.environ.copy()
    for variable, name in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                           ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state"), ("YOLO_CONFIG_DIR", "ultralytics"),
                           ("YOLOV5_CONFIG_DIR", "yolov5"), ("MPLCONFIGDIR", "matplotlib"),
                           ("TORCH_HOME", "torch"), ("TMPDIR", "tmp")):
        destination = path / name
        destination.mkdir(parents=True, exist_ok=True)
        env[variable] = str(destination)
    env.update(PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false", YOLOv5_AUTOINSTALL="false",
               YOLO_OFFLINE="true", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


class ConversionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vision-conversion-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "原 模型.pt"
        self.source.write_bytes(b"source is never overwritten")
        self.output = self.directory / "新 模型.onnx"

    def arguments(self, **changes):
        values = {"source": str(self.source), "output": str(self.output), "format": "onnx",
                  "image_size": 64, "opset": 12, "work_dir": ""}
        values.update(changes)
        return argparse.Namespace(**values)

    def test_import_has_no_framework_or_stdout_side_effect(self):
        probe = subprocess.run([sys.executable, "-c",
            "import importlib.util,sys; before=sys.stdout; "
            "s=importlib.util.spec_from_file_location('subject',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert sys.stdout is before; assert 'torch' not in sys.modules; "
            "assert 'ultralytics' not in sys.modules; print('import-safe')", str(HELPER)],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(probe.stdout, "import-safe\n")

    def test_paths_preserve_unicode_and_accept_only_valid_static_size(self):
        self.assertEqual(convert.validate_paths(self.arguments()), (self.source, self.output))
        for size in (0, 16, 63, 4097):
            with self.subTest(size=size), self.assertRaises(convert.ConversionError):
                convert.validate_paths(self.arguments(image_size=size))
        self.assertFalse(self.output.exists())

    def test_existing_output_and_dangling_symlink_are_never_overwritten(self):
        self.output.write_bytes(b"user data")
        before = self.output.stat()
        with self.assertRaises(convert.ConversionError):
            convert.validate_paths(self.arguments())
        self.assertEqual(self.output.read_bytes(), b"user data")
        self.assertEqual(self.output.stat().st_mtime_ns, before.st_mtime_ns)
        self.output.unlink()
        self.output.symlink_to(self.directory / "missing-user-target")
        with self.assertRaises(convert.ConversionError):
            convert.validate_paths(self.arguments())
        self.assertTrue(self.output.is_symlink())

    def test_symlink_output_parent_and_mismatched_format_fail(self):
        linked = self.directory / "linked"
        linked.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(convert.ConversionError):
            convert.validate_paths(self.arguments(output=str(linked / "converted.onnx")))
        with self.assertRaises(convert.ConversionError):
            convert.validate_paths(self.arguments(output=str(self.directory / "converted.pt")))

    def test_missing_or_empty_source_has_no_output(self):
        for source in (self.directory / "missing.pt", self.directory / "empty.pt"):
            if source.name == "empty.pt":
                source.touch()
            with self.subTest(source=source.name), self.assertRaises(convert.ConversionError):
                convert.validate_paths(self.arguments(source=str(source)))
        self.assertFalse(self.output.exists())

    def test_atomic_publish_rejects_creation_race_preserving_both_files(self):
        staged = self.directory / "staged"
        staged.write_bytes(b"validated new artifact")
        self.output.write_bytes(b"concurrent user output")
        with self.assertRaises(convert.ConversionError):
            convert.publish_no_replace(staged, self.output)
        self.assertEqual(self.output.read_bytes(), b"concurrent user output")
        self.assertEqual(staged.read_bytes(), b"validated new artifact")
        self.assertEqual(self.source.read_bytes(), b"source is never overwritten")

    def test_precreated_exact_work_directory_is_consumed(self):
        job = self.directory / (convert.JOB_PREFIX + "controller-owned")
        job.mkdir()
        initial = job.stat()
        with convert.JobDirectory(self.output, str(job)) as actual:
            self.assertEqual(actual, job)
            self.assertEqual(actual.stat().st_ino, initial.st_ino)
            (actual / "only-job-data").write_text("temporary")
        self.assertFalse(job.exists())

    def test_nonempty_foreign_and_symlink_work_directories_are_preserved(self):
        foreign = self.directory / (convert.JOB_PREFIX + "foreign")
        foreign.mkdir()
        (foreign / "user-note").write_text("preserve")
        linked = self.directory / (convert.JOB_PREFIX + "linked")
        linked.symlink_to(foreign, target_is_directory=True)
        for job in (foreign, linked, self.directory):
            with self.subTest(path=job.name), self.assertRaises(convert.ConversionError):
                convert.JobDirectory(self.output, str(job))
        self.assertEqual((foreign / "user-note").read_text(), "preserve")
        self.assertTrue(linked.is_symlink())

    def test_replaced_work_directory_is_not_removed(self):
        path = self.directory / (convert.JOB_PREFIX + "replace")
        path.mkdir()
        job = convert.JobDirectory(self.output, str(path))
        preserved = self.directory / "preserved-original-job"
        path.rename(preserved)
        path.mkdir()
        (path / "user-file").write_text("must remain")
        with self.assertRaises(convert.ConversionError):
            job.__exit__(None, None, None)
        self.assertEqual((path / "user-file").read_text(), "must remain")
        self.assertTrue(preserved.exists())

    def test_protocol_contains_message_and_bounds_large_metadata(self):
        for kind in ("progress", "status", "result", "error"):
            row = json.loads(convert.event_bytes(kind, "中文 信息", percent=40))
            self.assertEqual(row["protocol"], 1)
            self.assertEqual(row["message"], "中文 信息")
        with self.assertRaises(convert.ConversionError):
            convert.event_bytes("result", "x", names="x" * convert.MAX_EVENT_BYTES)
        with self.assertRaises(convert.ConversionError):
            convert.labels_list({"1": "non-contiguous"})


FIXTURE_SCRIPT = r'''
import copy,json,os,sys,torch
from pathlib import Path
torch.set_num_threads(1); torch.manual_seed(7321)
root,out=map(Path,sys.argv[1:3]); sys.path.insert(0,str(root/'vendor/yolov5'))
from ultralytics.nn.tasks import DetectionModel,ClassificationModel
from models.yolo import DetectionModel as Legacy
for channels in (1,3):
    detect={'nc':2,'depth_multiple':1.,'width_multiple':1.,'backbone':[[-1,1,'Conv',[8,3,2]],[-1,1,'Conv',[16,3,2]]],'head':[[[1],1,'Detect',['nc']]]}
    classification={'nc':3,'depth_multiple':1.,'width_multiple':1.,'backbone':[[-1,1,'Conv',[8,3,2]]],'head':[[-1,1,'Classify',[3]]]}
    focus={'nc':2,'depth_multiple':1.,'width_multiple':1.,'anchors':[[10,13,16,30,33,23]],'backbone':[[-1,1,'Focus',[8,3]],[-1,1,'Conv',[16,3,2]]],'head':[[[1],1,'Detect',['nc','anchors']]]}
    for title,kind,model in [('modern','detect',DetectionModel(copy.deepcopy(detect),ch=channels,verbose=False)),('classify','classify',ClassificationModel(copy.deepcopy(classification),ch=channels,verbose=False)),('focus','detect',Legacy(copy.deepcopy(focus),ch=channels))]:
        model.names={i:('类别 '+str(i)) for i in range(3 if kind=='classify' else 2)}
        model.task=kind;model.args={'task':kind};model.yaml['ch']=3 # deliberately stale for C1
        torch.save({'model':model.eval(),'train_args':{'task':kind}},out/(title+'-c'+str(channels)+'.pt'))
    class Probe(torch.nn.Module):
        def __init__(self,detect):
            super().__init__(); self.detect=detect; self.conv=torch.nn.Conv2d(channels,3,1,bias=False)
            self.register_buffer('coordinates',torch.ones(1,4,8))
            with torch.no_grad():
                self.conv.weight.fill_(.25)
        def forward(self,image):
            means=self.conv(image).mean((2,3))
            if self.detect:return torch.cat((self.coordinates,means[:,:2].unsqueeze(-1).expand(1,2,8)),1)
            return means*6
    for task in ('detect','classify'):
        model=torch.jit.trace(Probe(task=='detect').eval(),torch.zeros(1,channels,64,64))
        meta={'task':task,'names':['probe','other'] if task=='detect' else ['red','green','blue'],'input_channels':channels,'shape':[1,channels,64,64]}
        torch.jit.save(model,str(out/('script-'+task+'-c'+str(channels)+'.pt')),_extra_files={'config.txt':json.dumps(meta)})
bad=torch.jit.trace(Probe(False).eval(),torch.zeros(1,3,64,64))
torch.jit.save(bad,str(out/'bad-metadata.pt'),_extra_files={'config.txt':json.dumps({'task':'classify','names':['a','b','c'],'input_channels':1})})
torch.save(torch.nn.Conv2d(3,3,1).state_dict(),out/'bare-weights.pt')
print('fixtures-ready')
'''

TARGET_OPENCV_PROBE = r'''
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::runtime_error("model input.raw output.raw channels size required");
        cv::setNumThreads(1);
        const int channels = std::stoi(argv[4]), size = std::stoi(argv[5]);
        const int dimensions[] = {1, channels, size, size};
        cv::Mat input(4, dimensions, CV_32F);
        std::ifstream source(argv[2], std::ios::binary);
        source.read(reinterpret_cast<char *>(input.data), input.total() * input.elemSize());
        if (!source || source.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid input bytes");
        auto network = cv::dnn::readNetFromONNX(argv[1]);
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        network.setInput(input);
        const cv::Mat output = network.forward();
        if (!output.isContinuous() || output.depth() != CV_32F) throw std::runtime_error("unexpected output storage");
        std::ofstream destination(argv[3], std::ios::binary);
        destination.write(reinterpret_cast<const char *>(output.data), output.total() * output.elemSize());
        destination.close();
        if (!destination) throw std::runtime_error("cannot write output");
        std::cout << "{\"opencv_version\":\"" << CV_VERSION << "\",\"shape\":[";
        for (int index = 0; index < output.dims; ++index) {
            if (index) std::cout << ',';
            std::cout << output.size[index];
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
}
'''

VERIFY_SCRIPT = r'''
import importlib.util,json,sys,tempfile,subprocess,torch,numpy as np
from pathlib import Path
model_path=Path(sys.argv[1]); channels=int(sys.argv[2]);size=int(sys.argv[3]);fmt=sys.argv[4]
source=Path(sys.argv[5]);root=Path(sys.argv[6]);torch.set_num_threads(1)
image=torch.linspace(0,1,channels*size*size).reshape(1,channels,size,size)
if fmt=='torchscript':
    extra={'config.txt':''}; model=torch.jit.load(str(model_path),map_location='cpu',_extra_files=extra).eval()
    with torch.inference_mode():output=model(image).numpy()
    metadata=json.loads(extra['config.txt'])
else:
    import onnx,cv2
    model=onnx.load(str(model_path));onnx.checker.check_model(model,full_check=True)
    meta={item.key:item.value for item in model.metadata_props}
    metadata={key:json.loads(value) if key not in ('modelType','task','layout','precision') else value for key,value in meta.items()}
    network=cv2.dnn.readNetFromONNX(str(model_path));network.setInput(image.numpy());output=network.forward()
spec=importlib.util.spec_from_file_location('conversion_subject',root/'scripts/model_convert.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
original_stdout=sys.stdout;sys.stdout=sys.stderr
try:
    with tempfile.TemporaryDirectory(prefix='vision-conversion-verify-') as area:
        with helper.isolate_environment(Path(area)):
            network,task,names,ch,layout,source_format=helper.load_model(source,torch)
            assert ch==channels
            with torch.inference_mode():expected=helper.primary_output(network(image),torch).numpy()
finally:sys.stdout=original_stdout
assert output.shape==expected.shape,(output.shape,expected.shape)
max_abs=float(np.max(np.abs(output-expected)))
assert np.allclose(output,expected,rtol=2e-3,atol=2e-3),max_abs
target=None
if fmt=='onnx' and len(sys.argv)>7 and sys.argv[7]:
    with tempfile.TemporaryDirectory(prefix='vision-conversion-target-') as temporary:
        input_path=Path(temporary)/'input.raw';output_path=Path(temporary)/'output.raw'
        image.numpy().astype('<f4').tofile(input_path)
        probe=subprocess.run([sys.argv[7],str(model_path),str(input_path),str(output_path),str(channels),str(size)],capture_output=True,text=True,timeout=40)
        assert probe.returncode==0,probe.stderr
        details=json.loads(probe.stdout)
        actual=np.fromfile(output_path,dtype='<f4').reshape(details['shape'])
        assert actual.shape==expected.shape,(actual.shape,expected.shape)
        error=float(np.max(np.abs(actual-expected)))
        assert np.isfinite(actual).all() and np.allclose(actual,expected,rtol=.002,atol=.002),('System OpenCV differs from selected PT',details['opencv_version'],error)
        target={'opencv_version':details['opencv_version'],'max_abs_error':error,'finite':True,'shape':details['shape']}
buffers=[name for name,_ in model.named_buffers() if '_vision_static_' in name] if fmt=='torchscript' else []
trace_tf32=[]
if fmt=='torchscript':
    for node in model.inlined_graph.nodes():
        if node.kind()=='aten::_convolution':
            flags=list(node.inputs())
            if len(flags)==13:trace_tf32.append(flags[-1].toIValue())
    assert all(value is False for value in trace_tf32),('Trace captured reduced-precision TF32',trace_tf32)
print(json.dumps({'metadata':metadata,'shape':list(output.shape),'finite':bool(np.isfinite(output).all()),'max_abs_error':max_abs,'reference':'selected-original-model','rtol':.002,'atol':.002,'system_opencv':target,'portable_detection_buffers':buffers,'traced_convolution_allow_tf32':trace_tf32}))
'''


class ActualConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vision-conversion-actual-")
        cls.directory = Path(cls.temporary.name)
        cls.environment = isolated_environment(cls.directory / "environment")
        probe = subprocess.run([PYTHON, "-c", "import torch,onnx,cv2; print(torch.__version__)"],
                               env=cls.environment, capture_output=True, text=True, timeout=30)
        if probe.returncode:
            cls.temporary.cleanup()
            raise unittest.SkipTest("Prepared conversion runtime with torch/onnx/cv2 is required: " + probe.stderr[-400:])
        compiler, pkg_config = shutil.which("c++"), shutil.which("pkg-config")
        if not compiler or not pkg_config:
            cls.temporary.cleanup()
            raise unittest.SkipTest("System OpenCV numerical regressions require the application's C++ build tools")
        flags = subprocess.run([pkg_config, "--cflags", "--libs", "opencv4"], capture_output=True, text=True, timeout=10)
        if flags.returncode:
            cls.temporary.cleanup()
            raise unittest.SkipTest("System OpenCV build dependency unavailable: " + flags.stderr[-300:])
        target_source = cls.directory / "opencv_target.cpp"
        target_source.write_text(TARGET_OPENCV_PROBE)
        cls.target_probe = cls.directory / "opencv_target"
        compiled = subprocess.run([compiler, "-std=c++17", "-O2", str(target_source), "-o", str(cls.target_probe)]
                                  + shlex.split(flags.stdout), capture_output=True, text=True, timeout=40)
        if compiled.returncode:
            cls.temporary.cleanup()
            raise AssertionError("Cannot build actual system OpenCV probe: " + compiled.stderr[-2500:])
        cls.fixtures = cls.directory / "fixtures"
        cls.fixtures.mkdir()
        created = subprocess.run([PYTHON, "-c", FIXTURE_SCRIPT, str(PROJECT), str(cls.fixtures)],
                                 env=cls.environment, capture_output=True, text=True, timeout=90)
        if created.returncode:
            cls.temporary.cleanup()
            raise RuntimeError(created.stderr[-4000:])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.area = self.directory / self._testMethodName
        self.area.mkdir()
        self.environment = isolated_environment(self.area / "persistent-user")
        self.home_sentinel = Path(self.environment["HOME"]) / "user-note"
        self.home_sentinel.write_text("original user configuration")
        self.user_snapshot = {str(path.relative_to(self.area / "persistent-user")): digest(path)
                              for path in (self.area / "persistent-user").rglob("*") if path.is_file()}

    def run_conversion(self, source, *, format="onnx", channels=3, size=64, opset=12, runtime=None):
        source = Path(source)
        source_sha = digest(source)
        output = self.area / (source.stem + "-" + str(opset) + (".onnx" if format == "onnx" else ".torchscript"))
        work = Path(tempfile.mkdtemp(prefix=convert.JOB_PREFIX, dir=self.area))
        identity = work.stat()
        completed = subprocess.run([runtime or PYTHON, "-u", str(HELPER), "--source", str(source), "--format", format,
                                    "--output", str(output), "--image-size", str(size), "--opset", str(opset),
                                    "--work-dir", str(work)], env=self.environment,
                                   capture_output=True, text=True, timeout=90)
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertTrue(rows, completed.stderr[-2500:])
        for row in rows:
            self.assertIn("message", row)
            self.assertEqual(row["protocol"], 1)
        self.assertEqual(rows[0]["job_temp"], str(work))
        self.assertEqual(rows[0]["job_identity"]["inode"], identity.st_ino)
        self.assertEqual(digest(source), source_sha)
        self.assertFalse(work.exists(), completed.stderr[-2500:])
        self.assertEqual(self.home_sentinel.read_text(), "original user configuration")
        self.assertEqual({str(path.relative_to(self.area / "persistent-user")): digest(path)
                          for path in (self.area / "persistent-user").rglob("*") if path.is_file()}, self.user_snapshot)
        self.assertEqual(completed.returncode, 0, repr(rows[-1]) + "\n" + completed.stderr[-3500:])
        result = rows[-1]
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["output"], str(output))
        self.assertEqual(result["shape"], [1, channels, size, size])
        self.assertEqual(result["sha256"], digest(output))
        self.assertEqual(sum(row["type"] == "result" for row in rows), 1)
        verified = subprocess.run([PYTHON, "-c", VERIFY_SCRIPT, str(output), str(channels), str(size), format, str(source), str(PROJECT), str(self.target_probe)],
                                  env=isolated_environment(self.area / ("verify-" + output.stem)), capture_output=True, text=True, timeout=40)
        self.assertEqual(verified.returncode, 0, verified.stderr[-2500:])
        evidence = json.loads(verified.stdout)
        self.assertTrue(evidence["finite"])
        self.assertEqual(evidence["metadata"]["names"], result["names"])
        self.assertEqual(evidence["metadata"]["shape"], result["shape"])
        print("CONVERSION_EVIDENCE " + json.dumps({"source": str(source), "format": format,
              "opset": result["opset"], "input_shape": result["shape"], "output_shape": evidence["shape"],
              "max_abs_error": evidence["max_abs_error"], "reference": evidence["reference"],
              "system_opencv": evidence["system_opencv"], "portable_detection_buffers": evidence["portable_detection_buffers"],
              "traced_convolution_allow_tf32": evidence["traced_convolution_allow_tf32"],
              "sha256": result["sha256"], "source_sha256": source_sha, "source_unchanged": True,
              "original_user_state_unchanged": True, "job_removed": True}, ensure_ascii=False), flush=True)
        return output, result, evidence

    def test_actual_modern_selected_asset_name_not_substituted(self):
        for format in ("onnx", "torchscript"):
            with self.subTest(format=format):
                _, result, evidence = self.run_conversion(PROJECT / "models/yolov8n.pt", format=format, size=640)
                self.assertEqual(result["layout"], "v8")
                self.assertEqual(len(result["names"]), 80)
                if format == "torchscript":
                    self.assertTrue(evidence["portable_detection_buffers"])

    def test_actual_legacy_selected_yolov5_not_replaced_by_ultralytics_asset(self):
        for format in ("onnx", "torchscript"):
            with self.subTest(format=format):
                _, result, evidence = self.run_conversion(PROJECT / "models/yolov5n.pt", format=format, size=640)
                self.assertEqual(result["layout"], "v5")
                self.assertEqual(result["modelType"], "yolov5")
                if format == "torchscript":
                    self.assertTrue(evidence["portable_detection_buffers"])

    def test_actual_c1_modern_detection_ignores_stale_yaml_channel(self):
        _, result, _ = self.run_conversion(self.fixtures / "modern-c1.pt", channels=1)
        self.assertEqual(result["input_channels"], 1)
        self.assertEqual(result["task"], "detect")

    def test_actual_focus_c1_c3_channels_and_both_formats(self):
        for channels in (1, 3):
            for format in ("onnx", "torchscript"):
                with self.subTest(channels=channels, format=format):
                    _, result, _ = self.run_conversion(self.fixtures / ("focus-c" + str(channels) + ".pt"),
                                                       channels=channels, format=format)
                    self.assertEqual(result["layout"], "v5")
                    self.assertEqual(result["input_channels"], channels)

    def test_actual_native_classification_c1_c3(self):
        for channels, format in ((1, "onnx"), (3, "torchscript")):
            with self.subTest(channels=channels, format=format):
                _, result, evidence = self.run_conversion(self.fixtures / ("classify-c" + str(channels) + ".pt"),
                                                          channels=channels, format=format)
                self.assertEqual(result["task"], "classify")
                self.assertEqual(evidence["shape"], [1, 3])

    def test_actual_torchscript_sources_export_onnx_c1_c3_detect_classify(self):
        for channels in (1, 3):
            for task in ("detect", "classify"):
                with self.subTest(channels=channels, task=task):
                    _, result, _ = self.run_conversion(self.fixtures / ("script-" + task + "-c" + str(channels) + ".pt"),
                                                       channels=channels)
                    self.assertEqual(result["source_format"], "torchscript")
                    self.assertEqual(result["task"], task)

    def test_torchscript_export_works_in_original_runtime_without_onnx(self):
        runtime = PROJECT / "build/release-runtime/bin/python3"
        if not runtime.is_file():
            self.skipTest("Original inference-only runtime is unavailable")
        probe = subprocess.run([str(runtime), "-c", "import importlib.util; print(importlib.util.find_spec('onnx') is None)"],
                               env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        if probe.stdout.strip() != "True":
            self.skipTest("Original runtime already has ONNX; cannot prove dependency-free TorchScript here")
        _, result, _ = self.run_conversion(self.fixtures / "classify-c1.pt", format="torchscript", channels=1,
                                           runtime=str(runtime))
        self.assertEqual(result["task"], "classify")

    def test_actual_opset17_static_export(self):
        for filename, channels in (("modern-c3.pt", 3), ("script-detect-c1.pt", 1)):
            with self.subTest(filename=filename, channels=channels):
                _, result, _ = self.run_conversion(self.fixtures / filename, channels=channels, opset=17)
                self.assertEqual(result["opset"], 17)

    def test_bad_metadata_and_bare_weights_leave_no_artifact_or_job(self):
        for filename in ("bad-metadata.pt", "bare-weights.pt"):
            with self.subTest(filename=filename):
                source = self.fixtures / filename
                before = digest(source)
                output = self.area / (filename + ".onnx")
                completed = subprocess.run([PYTHON, str(HELPER), "--source", str(source), "--format", "onnx",
                                            "--output", str(output), "--image-size", "64"], env=self.environment,
                                           capture_output=True, text=True, timeout=40)
                rows = [json.loads(line) for line in completed.stdout.splitlines()]
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(rows[-1]["type"], "error")
                self.assertFalse(output.exists())
                self.assertFalse(any(path.name.startswith(convert.JOB_PREFIX) for path in self.area.iterdir()))
                self.assertEqual(digest(source), before)

    def test_cancel_status_precedes_torch_import_and_cleans_exact_job(self):
        source = self.fixtures / "modern-c3.pt"
        output = self.area / "cancelled.onnx"
        work = Path(tempfile.mkdtemp(prefix=convert.JOB_PREFIX, dir=self.area))
        process = subprocess.Popen([PYTHON, "-u", str(HELPER), "--source", str(source), "--format", "onnx",
                                    "--output", str(output), "--work-dir", str(work), "--image-size", "64"],
                                   env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(8), "Missing early job event")
            first = json.loads(process.stdout.readline())
            self.assertEqual(first["job_temp"], str(work))
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr.decode(errors="replace")[-2000:])
            rows = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(rows[-1]["code"], "cancelled")
            self.assertFalse(work.exists())
            self.assertFalse(output.exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
