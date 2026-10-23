#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""True batch shape/geometry tests plus real resident-worker CPU inference.

No production helper import side effects: function/class AST definitions are
loaded with the same installed torch/OpenCV implementation for focused tests.
"""
import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest

import cv2
import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parents[1]
body = ast.parse((ROOT / "scripts/pt_worker.py").read_text(encoding="utf-8"))
namespace = dict(torch=torch, cv2=cv2, np=np, nms=nms, time=time, Path=Path, json=json,
                 math=__import__("math"), zipfile=__import__("zipfile"))
exec(compile(ast.Module(body=[n for n in body.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                        type_ignores=[]), "pt_worker.py", "exec"), namespace)


class BatchTensorTests(unittest.TestCase):
    def test_native_gray_batch_is_one_predict_call_and_restores_each_geometry(self):
        for channels in (1, 3):
            with self.subTest(channels=channels):
                calls = []
                class Predictor:
                    def predict(self, **options):
                        calls.append(options)
                        source = options["source"]
                        self_shape = tuple(source.shape)
                        assert self_shape == (2, channels, 64, 64), self_shape
                        if channels == 3:
                            torch.testing.assert_close(source[:, 0], source[:, 1])
                            torch.testing.assert_close(source[:, 1], source[:, 2])
                        boxes = types.SimpleNamespace(xyxy=torch.tensor([[16., 24., 48., 40.]]),
                                                      conf=torch.tensor([.8]), cls=torch.tensor([32.]))
                        return [types.SimpleNamespace(boxes=boxes, speed={"inference": 2.}) for _ in range(2)]
                backend = namespace["NativeYolo"].__new__(namespace["NativeYolo"])
                backend.model, backend.device = Predictor(), torch.device("cpu")
                backend.task, backend.labels = "detect", [str(i) for i in range(80)]
                backend.color_mode, backend.input_channels = "grayscale", channels
                backend.class_filter = [32]
                images = [np.full((64, 128, 3), (20, 80, 240), dtype=np.uint8),
                          np.full((128, 64, 3), 64, dtype=np.uint8)]
                results = backend.infer_batch(images, 64, .1, .45, 9)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["batch"], 2)
                self.assertEqual(calls[0]["max_det"], 9)
                self.assertEqual(calls[0]["classes"], [32])
                self.assertAlmostEqual(float(calls[0]["source"][0, 0, 32, 32]), 121 / 255, places=6)
                self.assertEqual(results[0][0][0]["box"], [32., 16., 64., 32.])
                self.assertEqual(results[1][0][0]["box"], [0., 48., 64., 32.])
                self.assertEqual(results[0][0][0]["class_id"], 32)

    def test_native_color_batch_preserves_image_list_order_and_zero_result(self):
        calls = []
        images = [np.zeros((32, 48, 3), np.uint8), np.full((48, 32, 3), 255, np.uint8)]
        class Predictor:
            def predict(self, **options):
                calls.append(options)
                return [types.SimpleNamespace(boxes=types.SimpleNamespace(
                    xyxy=torch.empty((0, 4)), conf=torch.empty(0), cls=torch.empty(0)), speed={}) for _ in images]
        backend = namespace["NativeYolo"].__new__(namespace["NativeYolo"])
        backend.model, backend.device = Predictor(), torch.device("cpu")
        backend.task, backend.labels, backend.class_filter = "detect", ["object"], []
        backend.color_mode, backend.input_channels = "color", 3
        result = backend.infer_batch(images, 64, .25, .45)
        self.assertIs(calls[0]["source"], images)
        self.assertEqual(len(calls), 1)
        self.assertEqual([r[0] for r in result], [[], []])

    def test_legacy_batch_uses_one_forward_and_preserves_original_class_id(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, tensor):
                self.calls.append(tuple(tensor.shape))
                rows = torch.zeros((len(tensor), 1, 85))
                rows[:, 0, :4] = torch.tensor([32., 32., 16., 16.])
                rows[:, 0, 4] = 1.
                rows[:, 0, 5 + 32] = tensor.mean(dim=(1, 2, 3))
                return rows
        model = Detector()
        backend = namespace["LegacyYolo"].__new__(namespace["LegacyYolo"])
        backend.model, backend.device = model, torch.device("cpu")
        backend.labels, backend.class_filter = [str(i) for i in range(80)], [32]
        backend.color_mode, backend.input_channels = "grayscale", 1
        results = backend.infer_batch([np.full((64, 64, 3), 200, np.uint8), np.zeros((64, 64, 3), np.uint8)],
                                      64, .25, .45, 1)
        self.assertEqual(model.calls, [(2, 1, 64, 64)])
        self.assertEqual(results[0][0][0]["class_id"], 32)
        self.assertEqual(results[1][0], [])
        self.assertEqual(results[0][0][0]["box"], [24., 24., 16., 16.])

    def test_raw_decoder_honors_max_detections(self):
        rows = torch.tensor([[[12., 12., 8., 8., 1., .9], [32., 32., 8., 8., 1., .8]]])
        result = namespace["raw_predictions"](rows, np.zeros((64, 64, 3), np.uint8), (0, 0, 1., 1.),
                                               ["object"], "v5", .25, .45, (), 1)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["confidence"], .9, places=6)


class ResidentProtocolTests(unittest.TestCase):
    def start_worker(self, model, mode="color", threads=1):
        temp = tempfile.TemporaryDirectory(prefix="vision-pt-batch-")
        self.addCleanup(temp.cleanup)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", YOLO_CONFIG_DIR=temp.name)
        stderr = open(Path(temp.name) / "stderr.log", "w+b")
        self.addCleanup(stderr.close)
        process = subprocess.Popen([sys.executable, "-B", str(ROOT / "scripts/pt_worker.py"),
                                    "--model", str(model), "--size", "320", "--device",
                                    os.environ.get("VISION_STUDIO_PT_BATCH_DEVICE", "cpu"),
                                    "--color-mode", mode, "--cpu-threads", str(threads),
                                    "--class-filter", "[0,5]"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr, env=environment)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
        self.addCleanup(cleanup)
        ready = json.loads(process.stdout.readline())
        self.assertTrue(ready.get("ok"), ready)
        self.assertEqual(ready["maximum_batch_size"], 32)
        self.assertEqual(ready["cpu_threads"], threads)
        return process

    def request(self, process, request):
        process.stdin.write((json.dumps(request) + "\n").encode())
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    def test_real_modern_and_legacy_batch_color_gray_and_single_parity(self):
        original = cv2.imread(str(ROOT / "assets/bus.jpg"))
        images = [original, np.zeros((192, 320, 3), np.uint8), cv2.resize(original, (240, 320))]
        encoded = [base64.b64encode(cv2.imencode(".png", image)[1]).decode() for image in images]
        for name in ("yolov8n.pt", "yolov5n.pt"):
            for mode in ("color", "grayscale"):
                with self.subTest(model=name, mode=mode):
                    process = self.start_worker(ROOT / "models" / name, mode)
                    options = dict(input_size=320, confidence=.25, iou=.45, max_detections=2)
                    batch = self.request(process, dict(command="infer_batch", id=1, images=encoded, **options))
                    self.assertTrue(batch.get("ok"), batch)
                    self.assertEqual(batch["batch_size"], 3)
                    self.assertEqual([r["index"] for r in batch["results"]], [0, 1, 2])
                    self.assertEqual([r["cpu_threads"] for r in batch["results"]], [1, 1, 1])
                    self.assertEqual(batch["results"][1]["predictions"], [])
                    self.assertTrue(batch["results"][0]["predictions"])
                    for index, image in enumerate(encoded):
                        single = self.request(process, dict(command="infer", id=index + 2, image=image, **options))
                        self.assertTrue(single.get("ok"), single)
                        self.assertEqual(single["cpu_threads"], 1)
                        predictions = batch["results"][index]["predictions"]
                        self.assertLessEqual(len(predictions), 2)
                        self.assertEqual(len(predictions), len(single["predictions"]))
                        for a, b in zip(predictions, single["predictions"]):
                            self.assertIn(a["class_id"], (0, 5))
                            self.assertEqual(a["class_id"], b["class_id"])
                            self.assertAlmostEqual(a["confidence"], b["confidence"], delta=.002)
                            np.testing.assert_allclose(a["box"], b["box"], atol=.5)
                    self.request_quit(process)

    def request_quit(self, process):
        process.stdin.write(b'{"command":"quit"}\n')
        process.stdin.flush()
        self.assertEqual(process.wait(timeout=10), 0)

    def test_invalid_batch_and_maxdet_return_bound_error_then_allow_next_request(self):
        process = self.start_worker(ROOT / "models/yolov8n.pt")
        options = dict(command="infer_batch", input_size=320, confidence=.25, iou=.45)
        for index, value in enumerate(([], ["not png"], [True], ["x"] * 33)):
            result = self.request(process, dict(options, id=index, images=value))
            self.assertFalse(result["ok"])
            self.assertEqual(result["id"], index)
        image = base64.b64encode(cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1]).decode()
        for index, maxdet in enumerate((0, -1, True, 1.5, 10001), 10):
            result = self.request(process, dict(options, id=index, images=[image], max_detections=maxdet))
            self.assertFalse(result["ok"])
        good = self.request(process, dict(options, id=20, images=[image], max_detections=1))
        self.assertTrue(good["ok"], good)
        self.assertEqual(good["results"][0]["predictions"], [])
        self.request_quit(process)

    def test_real_one_channel_native_model_accepts_gray_batch(self):
        # Random weights verify actual C1 model execution, not model accuracy.
        # The exact input pixels/geometry are asserted by the tensor tests above.
        from ultralytics.nn.tasks import DetectionModel
        with tempfile.TemporaryDirectory(prefix="vision-pt-c1-batch-") as directory:
            config = {"nc": 80, "depth_multiple": 1., "width_multiple": 1.,
                      "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 2]],
                                   [-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 2]],
                                   [-1, 1, "Conv", [8, 3, 2]]],
                      "head": [[[-1], 1, "Detect", [80]]]}
            model = DetectionModel(config, ch=1, nc=80, verbose=False).float().eval()
            path = Path(directory) / "synthetic-c1.pt"
            torch.save({"model": model, "train_args": {"task": "detect"}}, path)
            process = self.start_worker(path, "grayscale")
            encoded = [base64.b64encode(cv2.imencode(".png", np.full(shape, value, np.uint8))[1]).decode()
                       for shape, value in (((64, 128, 3), 76), ((128, 64, 3), 150))]
            batch = self.request(process, dict(command="infer_batch", id=1, images=encoded,
                                               input_size=320, confidence=1., iou=.45, max_detections=1))
            self.assertTrue(batch.get("ok"), batch)
            self.assertEqual(batch["batch_size"], 2)
            self.assertEqual([row["predictions"] for row in batch["results"]], [[], []])
            self.request_quit(process)


if __name__ == "__main__":
    unittest.main(verbosity=2)
