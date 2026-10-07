#!/usr/bin/env python3
"""Persistent CPU PyTorch inference worker for the Qt application.

stdin/stdout are a JSON-lines protocol. Imported framework logs, including
native stdout writes, are redirected to stderr before importing PyTorch.
Models are loaded once; requests carry lossless PNG images and inference options.
"""

import os
import sys

_protocol_fd = os.dup(sys.stdout.fileno())
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
_protocol = os.fdopen(_protocol_fd, "wb", buffering=0)
sys.stdout = sys.stderr
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLOv5_AUTOINSTALL"] = "false"

import argparse
import base64
import json
import math
from pathlib import Path
import time
import traceback
import zipfile

MAX_REQUEST = 100 * 1024 * 1024


def reply(message):
    _protocol.write((json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode())


def labels_list(names):
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not isinstance(names, (list, tuple)) or not names or len(names) > 100000:
        raise ValueError("模型没有有效的类别名称；请使用完整的 YOLO 检查点或带类别元数据的 TorchScript。")
    labels = [str(name) if name is not None else "" for name in names]
    if any(not name or len(name) > 1000 for name in labels):
        raise ValueError("模型类别名称无效。")
    return labels


def is_torchscript(path):
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any("/code/" in name for name in archive.namelist())


def install_legacy_path():
    vendor = Path(__file__).resolve().parent.parent / "vendor" / "yolov5"
    if (vendor / "models" / "yolo.py").is_file():
        sys.path.insert(0, str(vendor))
        return True
    return False


def local_checkpoint_loader(path):
    """Use the selected file unchanged; framework asset naming cannot substitute it.

    Ultralytics' normal asset resolver rewrites yolov5n.pt to yolov5nu.pt before
    checking file existence. The helper process replaces that resolver with an
    exact local-path check, and blocks downloads entirely during inference.
    """
    from ultralytics.nn import tasks
    from ultralytics.utils import downloads

    def selected_local_file(file, *args, **kwargs):
        candidate = Path(str(file)).expanduser().resolve()
        if candidate != path:
            raise ValueError("模型框架请求了另一份权重，已拒绝替换所选的本地模型。")
        if not candidate.is_file():
            raise FileNotFoundError("所选的本地模型文件已不存在。")
        return str(candidate)

    def no_download(*args, **kwargs):
        raise RuntimeError("本地推理禁止自动下载模型或资源。请使用完整的本地权重。")

    downloads.attempt_download_asset = selected_local_file
    downloads.safe_download = no_download
    downloads.download = no_download
    loader = getattr(tasks, "load_checkpoint", None) or getattr(tasks, "attempt_load_one_weight", None)
    if loader is None:
        raise RuntimeError("当前 ultralytics 版本没有兼容的本地检查点加载接口。")
    return loader


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - new_height - top,
                                left, size - new_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    return tensor, (left, top, new_width / width, new_height / height)


def primary_tensor(output):
    if isinstance(output, torch.Tensor):
        return output.detach().float().cpu()
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0].detach().float().cpu()
    raise ValueError("模型输出不是支持的 YOLO 原始张量或分类张量。")


def raw_predictions(output, image, geometry, labels, layout, confidence, iou):
    output = primary_tensor(output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("检测输出需要 [1,N,5+C] 或 [1,4+C,N]，分割/姿态/端到端输出尚不受支持。")
    classes = len(labels)
    if layout == "v5":
        if output.shape[2] != classes + 5:
            raise ValueError("YOLOv5 输出与类别数不匹配。")
        rows = output[0]
        probabilities = rows[:, 5:]
        objectness = rows[:, 4]
    else:
        if output.shape[1] != classes + 4:
            raise ValueError("YOLOv8/11 输出与类别数不匹配。")
        rows = output[0].transpose(0, 1)
        probabilities = rows[:, 4:]
        objectness = torch.ones(rows.shape[0])
    if not torch.isfinite(rows).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("模型原始输出含无效数值或未解码的类别概率。")
    if (objectness < 0).any() or (objectness > 1).any():
        raise ValueError("YOLOv5 目标概率无效。")
    scores, class_ids = probabilities.max(1)
    scores = scores * objectness
    selected = (scores >= confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows, scores, class_ids = rows[selected], scores[selected], class_ids[selected]
    if len(rows) > 30000:
        strongest = scores.argsort(descending=True)[:30000]
        rows, scores, class_ids = rows[strongest], scores[strongest], class_ids[strongest]
    if not len(rows):
        return []
    boxes = torch.cat((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), dim=1)
    left, top, scale_x, scale_y = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image.shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image.shape[0])
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    kept = []
    for class_id in class_ids.unique():
        indices = torch.where(class_ids == class_id)[0]
        kept.extend(indices[nms(boxes[indices], scores[indices], iou)].tolist())
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    predictions = []
    for index in kept[:300]:
        x1, y1, x2, y2 = boxes[index].tolist()
        predictions.append({"class_id": int(class_ids[index]), "confidence": float(scores[index]),
                            "box": [x1, y1, x2 - x1, y2 - y1]})
    return predictions


def classification_predictions(output, labels):
    output = primary_tensor(output)
    if output.numel() != len(labels) or output.shape[0] != 1:
        raise ValueError("分类输出与类别数不匹配。")
    values = output.reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("分类输出包含无效数值。")
    probabilities = values if (values >= 0).all() and (values <= 1).all() and abs(float(values.sum()) - 1) <= .001 \
        else values.softmax(0)
    scores, indices = probabilities.topk(min(5, len(labels)))
    return [{"class_id": int(index), "confidence": float(score), "box": []}
            for index, score in zip(indices, scores)]


class NativeYolo:
    def __init__(self, path, model, checkpoint):
        from ultralytics import YOLO
        from ultralytics.engine.model import Model as UltralyticsModel

        # Initialize the standard predictor wrapper around the model already read
        # from the selected checkpoint. No second weight load or filename-based
        # YOLO/World/YOLOE factory selection is involved.
        class LoadedCheckpointYolo(YOLO):
            def _load(wrapper, weights, task=None):
                if Path(str(weights)).resolve() != path:
                    raise ValueError("模型路径与所选文件不匹配。")
                wrapper.model, wrapper.ckpt = model, checkpoint
                wrapper.task = model.task
                wrapper.overrides = wrapper.model.args = wrapper._reset_ckpt_args(model.args)
                wrapper.ckpt_path = str(path)
                wrapper.overrides.update({"model": str(path), "task": wrapper.task})
                wrapper.model_name = str(path)

        self.model = LoadedCheckpointYolo.__new__(LoadedCheckpointYolo)
        UltralyticsModel.__init__(self.model, model=str(path), verbose=False)
        self.task = self.model.task
        if self.task not in ("detect", "classify"):
            raise ValueError(f"此 .pt 模型是 {self.task} 任务；当前支持目标检测与分类，尚不支持分割/姿态/旋转框。")
        self.labels = labels_list(self.model.names)
        self.backend = "PyTorch / Ultralytics / CPU"
        self.layout = "v8"
        self.source_path = path

    def infer(self, image, size, confidence, iou):
        start = time.perf_counter()
        results = self.model.predict(source=image, imgsz=size, conf=confidence, iou=iou,
                                     device="cpu", max_det=300, rect=False, agnostic_nms=False,
                                     verbose=False, save=False, show=False, stream=False)
        if len(results) != 1:
            raise ValueError("模型没有返回单张图像的预测结果。")
        result = results[0]
        predictions = []
        if self.task == "classify":
            if result.probs is None:
                raise ValueError("分类模型没有返回类别概率。")
            predictions = classification_predictions(result.probs.data.unsqueeze(0), self.labels)
        else:
            if result.boxes is None:
                raise ValueError("目标检测模型没有返回检测框。")
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                predictions.append({"class_id": int(class_id), "confidence": float(score),
                                    "box": [x1, y1, x2 - x1, y2 - y1]})
        total_ms = (time.perf_counter() - start) * 1000
        inference_ms = float(result.speed.get("inference", total_ms))
        return predictions, inference_ms


class LegacyYolo:
    def __init__(self, path, existing_model=None):
        if existing_model is None:
            if not install_legacy_path():
                raise ValueError("旧版 YOLOv5 .pt 需要随应用提供的 vendor/yolov5 框架。请保留完整项目目录。")
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("此 .pt 不包含可直接运行的 YOLO 网络；state_dict 权重还需要网络架构。")
            existing_model = checkpoint.get("ema")
            if existing_model is None:
                existing_model = checkpoint.get("model")
        if not isinstance(existing_model, torch.nn.Module) or not existing_model.__class__.__module__.startswith("models."):
            raise ValueError("此 .pt 不是完整的 YOLOv5 检查点；仅有 state_dict 的权重还需要对应的网络架构。")
        if getattr(existing_model, "task", "detect") != "detect":
            raise ValueError("旧版 YOLOv5 当前支持目标检测；此检查点的分类或分割任务尚不受支持。")
        self.model = existing_model.to("cpu").float().eval()
        if hasattr(self.model, "fuse"):
            self.model = self.model.fuse().eval()
        from models.yolo import Detect
        for module in self.model.modules():
            if isinstance(module, Detect):
                module.inplace = True
                if hasattr(module, "anchor_grid") and not isinstance(module.anchor_grid, list):
                    delattr(module, "anchor_grid")
                    module.anchor_grid = [torch.zeros(1)] * module.nl
                if not hasattr(module, "grid") or not isinstance(module.grid, list):
                    module.grid = [torch.zeros(1)] * module.nl
            if isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
                module.recompute_scale_factor = None
        self.labels = labels_list(self.model.names)
        self.task = "detect"
        self.backend = "PyTorch / YOLOv5 / CPU"
        self.layout = "v5"
        self.source_path = path

    def infer(self, image, size, confidence, iou):
        tensor, geometry = letterbox(image, size)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        inference_ms = (time.perf_counter() - start) * 1000
        return raw_predictions(output, image, geometry, self.labels, "v5", confidence, iou), inference_ms


class TorchScriptYolo:
    def __init__(self, path, task, size, supplied_labels):
        extra = {"config.txt": ""}
        self.model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra).float().eval()
        metadata = {}
        if extra["config.txt"]:
            metadata = json.loads(extra["config.txt"])
        self.task = metadata.get("task", "classify" if task == "classify" else "detect")
        if self.task not in ("detect", "classify"):
            raise ValueError("此 TorchScript 模型任务尚不受支持；请选择检测或分类模型。")
        self.labels = labels_list(metadata.get("names", supplied_labels))
        self.layout = "v5" if task == "v5" else "v8"
        with torch.inference_mode():
            output = primary_tensor(self.model(torch.zeros(1, 3, size, size)))
        if self.task == "detect":
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError("TorchScript 检测模型没有标准原始 YOLO 输出。")
            if output.shape[1] == len(self.labels) + 4:
                self.layout = "v8"
            elif output.shape[2] == len(self.labels) + 5:
                self.layout = "v5"
            else:
                raise ValueError("TorchScript 检测输出与类别元数据不匹配。")
        elif output.numel() != len(self.labels):
            raise ValueError("TorchScript 分类输出与类别元数据不匹配。")
        self.backend = "PyTorch / TorchScript / CPU"
        self.source_path = path

    def infer(self, image, size, confidence, iou):
        if self.task == "classify":
            from PIL import Image
            from ultralytics.data.augment import classify_transforms
            transform = classify_transforms(size=size)
            tensor = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            geometry = None
        else:
            tensor, geometry = letterbox(image, size)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        inference_ms = (time.perf_counter() - start) * 1000
        predictions = classification_predictions(output, self.labels) if self.task == "classify" \
            else raw_predictions(output, image, geometry, self.labels, self.layout, confidence, iou)
        return predictions, inference_ms


def load_model(arguments):
    path = Path(arguments.model).resolve()
    if not path.is_file():
        raise ValueError("找不到 .pt 模型文件。")
    if path.stat().st_size == 0:
        raise ValueError(".pt 模型文件为空。")
    labels = json.loads(arguments.labels)
    if is_torchscript(path):
        return TorchScriptYolo(path, arguments.task, arguments.size, labels)
    install_legacy_path()
    try:
        model, checkpoint = local_checkpoint_loader(path)(str(path), device=torch.device("cpu"))
        if Path(str(getattr(model, "pt_path", ""))).resolve() != path:
            raise ValueError("模型框架没有加载所选的本地文件，已拒绝继续推理。")
        module_namespace = model.__class__.__module__
        if module_namespace.startswith("models."):
            return LegacyYolo(path, model)
        if module_namespace.startswith("ultralytics."):
            return NativeYolo(path, model, checkpoint)
        raise ValueError("此 .pt 的网络架构不属于支持的 YOLO 框架。")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("无法直接运行此 .pt。请使用完整的 Ultralytics YOLO 检查点；"
                         "仅有 state_dict 的权重需要网络架构，旧版自定义模型还需要其对应代码。\n"
                         f"检查点错误：{str(error)[:700]}") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("detect", "v5", "classify"), default="detect")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--labels", default="[]")
    arguments = parser.parse_args()
    config_directory = os.environ.get("YOLO_CONFIG_DIR")
    if config_directory:
        Path(config_directory).mkdir(parents=True, exist_ok=True)
    try:
        global cv2, np, torch, nms
        import cv2
        import numpy as np
        import torch
        from torchvision.ops import nms
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    except Exception as error:
        reply({"ok": False, "error": "PyTorch 运行环境不完整；需要 torch、torchvision、ultralytics、numpy 和 OpenCV。",
               "detail": str(error)[:1500]})
        return 1
    try:
        backend = load_model(arguments)
        reply({"ok": True, "event": "ready", "protocol": 1, "task": backend.task,
               "layout": backend.layout,
               "model_path": str(backend.source_path),
               "labels": backend.labels, "backend": backend.backend,
               "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        reply({"ok": False, "error": str(error), "detail": ""})
        return 1
    while True:
        raw = sys.stdin.buffer.readline(MAX_REQUEST + 1)
        if not raw:
            return 0
        if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
            reply({"ok": False, "error": "推理请求过大或格式不完整。"})
            return 1
        request_id = None
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("推理请求应为 JSON 对象。")
            if request.get("command") == "quit":
                return 0
            request_id = request.get("id")
            if isinstance(request_id, bool) or not isinstance(request_id, (int, float)) \
                    or not math.isfinite(request_id) or int(request_id) != request_id:
                raise ValueError("推理请求序号无效。")
            if request.get("command") != "infer":
                raise ValueError("未知推理命令。")
            size = int(request["input_size"])
            confidence, iou = float(request["confidence"]), float(request["iou"])
            if not 16 <= size <= 4096 or (backend.task == "detect" and size % 32):
                raise ValueError("检测输入尺寸必须是 32 的倍数，且不超过 4096。")
            if not math.isfinite(confidence) or not math.isfinite(iou) or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
                raise ValueError("置信度与 NMS 阈值应在 0 到 1 之间。")
            start = time.perf_counter()
            encoded = base64.b64decode(request["image"], validate=True)
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError("无法解码推理图像。")
            if image.shape[0] * image.shape[1] > 100000000:
                raise ValueError("原图超过 1 亿像素，请缩小后重新运行。")
            predictions, inference_ms = backend.infer(image, size, confidence, iou)
            reply({"ok": True, "id": request_id, "task": backend.task,
                   "predictions": predictions, "inference_ms": inference_ms,
                   "total_ms": (time.perf_counter() - start) * 1000})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "id": request_id, "error": "PyTorch 推理失败：" + str(error), "detail": ""})


if __name__ == "__main__":
    raise SystemExit(main())
