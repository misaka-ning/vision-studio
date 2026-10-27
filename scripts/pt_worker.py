#!/usr/bin/env python3
"""Persistent FP32 PyTorch inference worker for the Qt application.

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


def checked_channels(value, color_mode):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 3):
        raise ValueError("当前支持输入通道数为 1 或 3 的模型。请检查模型输入层和 input_channels 元数据。")
    if color_mode == "color" and value != 3:
        raise ValueError("此模型需要单通道输入；请选择灰度模式后重新运行。")
    return value


def module_input_channels(model, color_mode):
    """Read the actual YOLO input stem; YAML may be stale after customization."""
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    actual = convolutions[0].in_channels if convolutions else None
    stem = getattr(model, "model", None)
    if isinstance(stem, (torch.nn.Sequential, torch.nn.ModuleList)) and len(stem) \
            and stem[0].__class__.__name__ == "Focus" and actual is not None:
        # Older YOLOv5 Focus concatenates four spatial slices before its Conv.
        if actual % 4:
            raise ValueError("YOLOv5 Focus 输入层的通道数无效。")
        actual //= 4
    yaml = getattr(model, "yaml", {})
    declared = yaml.get("channels", yaml.get("ch")) if isinstance(yaml, dict) else None
    if actual is None and declared is None:
        raise ValueError("无法确定模型的输入通道数；请使用含输入层或通道元数据的完整模型。")
    return checked_channels(actual if actual is not None else declared, color_mode)


def script_input_channels(model, metadata, color_mode):
    """Infer only from convolutions that consume the image input unchanged.

    Saved TorchScript often erases TensorType input sizes. Reading an arbitrary
    first parameter could mistake a later layer's channels for the input, so we
    trace the graph back to the image argument and resolve its Conv weight.
    """
    graph = model.inlined_graph
    inputs = list(graph.inputs())
    image_inputs = [value for value in inputs if isinstance(value.type(), torch.TensorType)]
    inferred = set()
    if len(image_inputs) == 1:
        image_input = image_inputs[0]
        sizes = image_input.type().sizes()
        if sizes is not None and len(sizes) == 4 and sizes[1] is not None:
            inferred.add(sizes[1])
        preserving = {"aten::to", "aten::contiguous", "aten::detach", "aten::clone", "aten::cpu",
                      "aten::type_as", "aten::div", "aten::mul", "aten::sub", "aten::add"}

        def unchanged_input(value):
            if value == image_input:
                return True
            node = value.node()
            if node.kind() not in preserving:
                return False
            arguments = list(node.inputs())
            # Arithmetic with another image tensor may broadcast/change channels.
            if node.kind() in {"aten::div", "aten::mul", "aten::sub", "aten::add"} \
                    and len(arguments) > 1 and isinstance(arguments[1].type(), torch.TensorType):
                return False
            return bool(arguments) and unchanged_input(arguments[0])

        def attribute_value(value):
            if value == inputs[0]:
                return model
            node = value.node()
            if node.kind() == "prim::GetAttr":
                owner = attribute_value(list(node.inputs())[0])
                return getattr(owner, node.s("name"))
            return value.toIValue()

        for node in graph.nodes():
            if node.kind() not in ("aten::_convolution", "aten::conv2d", "aten::convolution"):
                continue
            arguments = list(node.inputs())
            if not arguments or not unchanged_input(arguments[0]):
                continue
            try:
                weight = attribute_value(arguments[1])
                groups = arguments[6 if node.kind() == "aten::conv2d" else 8].toIValue()
                if node.kind() != "aten::conv2d" and arguments[6].toIValue() is not False:
                    continue  # transposed or dynamic convolution needs explicit metadata
                if isinstance(weight, torch.Tensor) and weight.ndim == 4 \
                        and isinstance(groups, int) and not isinstance(groups, bool) and groups > 0:
                    inferred.add(int(weight.shape[1]) * groups)
            except (AttributeError, RuntimeError, TypeError, IndexError):
                continue
    if len(inferred) > 1:
        raise ValueError("TorchScript 输入层的通道约束不一致。")
    declared = metadata.get("input_channels")
    if declared is None:
        shape = metadata.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            declared = shape[1]
    actual = next(iter(inferred), None)
    if declared is not None and actual is not None and declared != actual:
        raise ValueError("TorchScript 的 input_channels 元数据与实际输入层不一致。")
    if declared is None and actual is None:
        raise ValueError("无法确定 TorchScript 的输入通道数；请在 config.txt 中添加 input_channels: 1 或 3。")
    return checked_channels(declared if declared is not None else actual, color_mode)


def prepared_image(image, color_mode, channels):
    if color_mode == "color":
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray[:, :, None] if channels == 1 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_timed(model, tensor, device):
    # CUDA enqueues work asynchronously; measure completed model work, rather
    # than only Python dispatch. Tensor transfer remains in total request time.
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    synchronize(device)
    return output, (time.perf_counter() - start) * 1000


def checked_device(requested, index):
    if requested == "cpu":
        return torch.device("cpu")
    if index < 0 or index > 255:
        raise ValueError("CUDA 设备编号应在 0 到 255 之间。")
    if torch.version.cuda is None:
        raise ValueError("所选 Python 环境是 CPU 版 PyTorch，不能进行 CUDA 推理。")
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA 设备 {index} 不可用；请检查 NVIDIA 驱动和 GPU 运行环境。")
    device = torch.device("cuda", index)
    # Availability alone is not proof that kernels can execute on the driver.
    torch.zeros(1, device=device).sum().item()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def classification_tensor(image, size, color_mode, channels):
    from PIL import Image
    from ultralytics.data.augment import classify_transforms
    prepared = prepared_image(image, color_mode, 3)
    transform = classify_transforms(size=size, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)) \
        if color_mode == "grayscale" else classify_transforms(size=size)
    tensor = transform(Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)))
    return tensor[:channels].unsqueeze(0)


def letterbox(image, size, color_mode="color", channels=3):
    image = prepared_image(image, color_mode, channels)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - new_height - top,
                                left, size - new_width - left, cv2.BORDER_CONSTANT, value=(114,) * channels)
    if channels == 1:
        padded = padded[:, :, None] if padded.ndim == 2 else padded
    tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    return tensor, (left, top, new_width / width, new_height / height)


def primary_tensor(output):
    if isinstance(output, torch.Tensor):
        return output.detach().float().cpu()
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0].detach().float().cpu()
    raise ValueError("模型输出不是支持的 YOLO 原始张量或分类张量。")


def raw_predictions(output, image, geometry, labels, layout, confidence, iou, class_filter=(), max_detections=300):
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
    if class_filter:
        allowed = torch.tensor(class_filter, dtype=class_ids.dtype)
        selected &= (class_ids[:, None] == allowed[None, :]).any(1)
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
    for index in kept[:max_detections]:
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


def fp32_predict_options():
    """Select the supported FP32 setting without deprecated aliases.

    Current Ultralytics uses quantize=None for FP32; older supported runtimes
    require half=False. Inspect configuration capabilities rather than a version
    string so both paths explicitly override inherited checkpoint precision.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return {"quantize": None} if "quantize" in DEFAULT_CFG_DICT else {"half": False}


class NativeYolo:
    def __init__(self, path, model, checkpoint, color_mode, device):
        self.class_filter = []
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
        head = self.model.model.model[-1]
        declared_classes = getattr(head, "nc", len(self.labels))
        if not isinstance(declared_classes, int) or declared_classes != len(self.labels):
            raise ValueError("模型检测头类别数与内置类别元数据不一致。")
        self.device = device
        self.model.to(device)
        self.model.model.float().eval()
        self.backend = "PyTorch / Ultralytics / " + ("CUDA" if device.type == "cuda" else "CPU")
        self.layout = "v8"
        self.source_path = path
        self.color_mode = color_mode
        self.input_channels = module_input_channels(model, color_mode)

    def infer(self, image, size, confidence, iou, max_detections=300):
        return self.infer_batch([image], size, confidence, iou, max_detections)[0]

    def infer_batch(self, images, size, confidence, iou, max_detections=300):
        geometries = [None] * len(images)
        # Color sources follow the framework's existing BGR loader. A list is
        # one real batch in LoadPilAndNumpy, not repeated predict() invocations.
        source = images
        if self.color_mode == "grayscale":
            if self.task == "classify":
                if len(images) != 1:
                    raise ValueError("分类后端仅支持 batch=1。")
                source = classification_tensor(images[0], size, self.color_mode, self.input_channels).to(self.device)
                output, inference_ms = forward_timed(self.model.model, source, self.device)
                return [(classification_predictions(output, self.labels), inference_ms)]
            prepared = [letterbox(image, size, self.color_mode, self.input_channels) for image in images]
            source = torch.cat([item[0] for item in prepared], dim=0).to(self.device)
            geometries = [item[1] for item in prepared]
        synchronize(self.device)
        start = time.perf_counter()
        results = self.model.predict(source=source, imgsz=size, conf=confidence, iou=iou,
                                     classes=self.class_filter or None, batch=len(images),
                                     device=str(self.device), **fp32_predict_options(), max_det=max_detections, rect=False,
                                     agnostic_nms=False, verbose=False, save=False, show=False, stream=False)
        synchronize(self.device)
        if len(results) != len(images):
            raise ValueError("模型返回的批量图像数量与输入不一致。")
        total_ms = (time.perf_counter() - start) * 1000
        rows = []
        for image, geometry, result in zip(images, geometries, results):
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
                    if not all(math.isfinite(value) for value in box):
                        raise ValueError("模型检测框包含非有限坐标。")
                    if geometry is not None:
                        left, top, scale_x, scale_y = geometry
                        x1, x2 = (x1 - left) / scale_x, (x2 - left) / scale_x
                        y1, y2 = (y1 - top) / scale_y, (y2 - top) / scale_y
                    # Native color predictions are already scaled by the
                    # framework, but clipping to the source bounds can collapse
                    # low-confidence boxes onto an image edge. Apply the same
                    # positive-area rule as gray/legacy/TorchScript inference.
                    x1, x2 = max(0, min(image.shape[1], x1)), max(0, min(image.shape[1], x2))
                    y1, y2 = max(0, min(image.shape[0], y1)), max(0, min(image.shape[0], y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append({"class_id": int(class_id), "confidence": float(score),
                                        "box": [x1, y1, x2 - x1, y2 - y1]})
            rows.append((predictions, float(result.speed.get("inference", total_ms / len(images)))))
        return rows


class LegacyYolo:
    def __init__(self, path, existing_model=None, color_mode="color", device=None):
        self.class_filter = []
        self.device = device if device is not None else torch.device("cpu")
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
        self.model = existing_model.to(self.device).float().eval()
        if hasattr(self.model, "fuse"):
            self.model = self.model.fuse().eval()
        from models.yolo import Detect
        for module in self.model.modules():
            if isinstance(module, Detect):
                module.inplace = True
                if hasattr(module, "anchor_grid") and not isinstance(module.anchor_grid, list):
                    delattr(module, "anchor_grid")
                    module.anchor_grid = [torch.zeros(1, device=self.device)] * module.nl
                if not hasattr(module, "grid") or not isinstance(module.grid, list):
                    module.grid = [torch.zeros(1, device=self.device)] * module.nl
            if isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
                module.recompute_scale_factor = None
        self.labels = labels_list(self.model.names)
        head_classes = getattr(self.model.model[-1], "nc", None)
        if head_classes != len(self.labels):
            raise ValueError("YOLOv5 检测头类别数与内置类别元数据不一致。")
        self.task = "detect"
        self.backend = "PyTorch / YOLOv5 / " + ("CUDA" if self.device.type == "cuda" else "CPU")
        self.layout = "v5"
        self.source_path = path
        self.color_mode = color_mode
        self.input_channels = module_input_channels(self.model, color_mode)

    def infer(self, image, size, confidence, iou, max_detections=300):
        return self.infer_batch([image], size, confidence, iou, max_detections)[0]

    def infer_batch(self, images, size, confidence, iou, max_detections=300):
        prepared = [letterbox(image, size, self.color_mode, self.input_channels) for image in images]
        tensor = torch.cat([item[0] for item in prepared], dim=0).to(self.device)
        output, inference_ms = forward_timed(self.model, tensor, self.device)
        output = primary_tensor(output)
        if output.ndim != 3 or output.shape[0] != len(images):
            raise ValueError("YOLOv5 模型返回的批量维度与输入不一致。")
        return [(raw_predictions(output[index:index + 1], image, prepared[index][1], self.labels,
                                 "v5", confidence, iou, self.class_filter, max_detections),
                 inference_ms / len(images)) for index, image in enumerate(images)]


class TorchScriptYolo:
    def __init__(self, path, task, size, supplied_labels, color_mode, device):
        self.class_filter = []
        self.device = device
        extra = {"config.txt": ""}
        with path.open("rb") as stream:
            self.model = torch.jit.load(stream, map_location="cpu", _extra_files=extra).to(device).float().eval()
        metadata = {}
        if extra["config.txt"]:
            metadata = json.loads(extra["config.txt"])
        if not isinstance(metadata, dict):
            raise ValueError("TorchScript 的 config.txt 必须是 JSON 对象。")
        self.task = metadata.get("task", "classify" if task == "classify" else "detect")
        if self.task not in ("detect", "classify"):
            raise ValueError("此 TorchScript 模型任务尚不受支持；请选择检测或分类模型。")
        # User label files are a display mapping, never evidence of model class count.
        self.labels = labels_list(metadata.get("names", metadata.get("classes")))
        self.layout = "v5" if task == "v5" else "v8"
        self.color_mode = color_mode
        self.input_channels = script_input_channels(self.model, metadata, color_mode)
        with torch.inference_mode():
            output = primary_tensor(self.model(torch.zeros(1, self.input_channels, size, size, device=device)))
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
        self.backend = "PyTorch / TorchScript / " + ("CUDA" if device.type == "cuda" else "CPU")
        self.source_path = path

    def infer(self, image, size, confidence, iou, max_detections=300):
        if self.task == "classify":
            tensor = classification_tensor(image, size, self.color_mode, self.input_channels)
            geometry = None
        else:
            tensor, geometry = letterbox(image, size, self.color_mode, self.input_channels)
        tensor = tensor.to(self.device)
        output, inference_ms = forward_timed(self.model, tensor, self.device)
        predictions = classification_predictions(output, self.labels) if self.task == "classify" \
            else raw_predictions(output, image, geometry, self.labels, self.layout, confidence, iou,
                                 self.class_filter, max_detections)
        return predictions, inference_ms


def load_model(arguments, device):
    path = Path(arguments.model).resolve()
    if not path.is_file():
        raise ValueError("找不到 .pt 模型文件。")
    if path.stat().st_size == 0:
        raise ValueError(".pt 模型文件为空。")
    labels = json.loads(arguments.labels)
    if is_torchscript(path):
        return TorchScriptYolo(path, arguments.task, arguments.size, labels, arguments.color_mode, device)
    install_legacy_path()
    try:
        model, checkpoint = local_checkpoint_loader(path)(str(path), device=torch.device("cpu"))
        if Path(str(getattr(model, "pt_path", ""))).resolve() != path:
            raise ValueError("模型框架没有加载所选的本地文件，已拒绝继续推理。")
        module_namespace = model.__class__.__module__
        if module_namespace.startswith("models."):
            return LegacyYolo(path, model, arguments.color_mode, device)
        if module_namespace.startswith("ultralytics."):
            return NativeYolo(path, model, checkpoint, arguments.color_mode, device)
        raise ValueError("此 .pt 的网络架构不属于支持的 YOLO 框架。")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("无法直接运行此 .pt。请使用完整的 Ultralytics YOLO 检查点；"
                         "仅有 state_dict 的权重需要网络架构，旧版自定义模型还需要其对应代码。\n"
                         f"检查点错误：{str(error)[:700]}") from error


def decode_image(encoded):
    if not isinstance(encoded, str):
        raise ValueError("推理图像必须是 PNG 的 Base64 字符串。")
    raw = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("无法解码推理图像。")
    if image.shape[0] * image.shape[1] > 100000000:
        raise ValueError("原图超过 1 亿像素，请缩小后重新运行。")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("detect", "v5", "classify"), default="detect")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--labels", default="[]")
    parser.add_argument("--class-filter", default="[]")
    parser.add_argument("--color-mode", choices=("color", "grayscale"), default="color")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
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
        if not 0 <= arguments.cpu_threads <= 256:
            raise ValueError("CPU 线程数应在 0 到 256 之间，0 表示自动。")
        torch.set_num_threads(arguments.cpu_threads or min(4, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
        if arguments.cpu_threads:
            # select_device() resets CPU threads when the native predictor is
            # first constructed. Keep that process-local reset consistent with
            # the explicit evaluation setting, without changing automatic mode.
            import ultralytics.utils.torch_utils as torch_utils
            torch_utils.NUM_THREADS = arguments.cpu_threads
    except Exception as error:
        reply({"ok": False, "error": "PyTorch 运行环境不完整；需要 torch、torchvision、ultralytics、numpy 和 OpenCV。",
               "detail": str(error)[:1500]})
        return 1
    try:
        if not 16 <= arguments.size <= 4096:
            raise ValueError("模型输入尺寸必须在 16 至 4096 之间。")
        device = checked_device(arguments.device, arguments.device_index)
        backend = load_model(arguments, device)
        class_filter = json.loads(arguments.class_filter)
        if not isinstance(class_filter, list) or any(isinstance(value, bool) or not isinstance(value, int)
                                                    or value < 0 or value >= len(backend.labels)
                                                    for value in class_filter):
            raise ValueError("类别筛选必须是模型有效范围内的非负整数 ID 列表。")
        if backend.task != "detect" and class_filter:
            raise ValueError("类别筛选仅用于目标检测；分类任务不支持此筛选。")
        backend.class_filter = list(dict.fromkeys(class_filter))
        # Predictors may choose their own automatic value during first setup.
        # Only explicit settings are reported as fixed; 0 honestly means auto.
        if backend.task == "detect" and arguments.size % 32:
            raise ValueError(".pt 检测模型输入尺寸必须是 32 的倍数；请调整输入尺寸后重新加载。")
        if device.type == "cuda":
            # Verify this model's CUDA/cuDNN kernels before announcing ready.
            # Auto mode can then fall back during initialization, rather than
            # silently switching devices halfway through processing a source.
            backend.infer(np.zeros((arguments.size, arguments.size, 3), dtype=np.uint8),
                          arguments.size, 1.0, .45)
            synchronize(device)
        reply({"ok": True, "event": "ready", "protocol": 1, "task": backend.task,
               "layout": backend.layout,
               "maximum_batch_size": 32 if backend.task == "detect" and isinstance(backend, (NativeYolo, LegacyYolo)) else 1,
               "cpu_threads": arguments.cpu_threads,
               "input_channels": backend.input_channels,
               "class_count": len(backend.labels),
               "model_path": str(backend.source_path),
               "labels": backend.labels, "backend": backend.backend,
               "device": device.type,
               "device_index": device.index if device.type == "cuda" else -1,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "versions": {"torch": torch.__version__, "python": sys.version.split()[0],
                            "cuda": torch.version.cuda}})
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
            command = request.get("command")
            if command not in ("infer", "infer_batch"):
                raise ValueError("未知推理命令。")
            size = int(request["input_size"])
            confidence, iou = float(request["confidence"]), float(request["iou"])
            if not 16 <= size <= 4096 or (backend.task == "detect" and size % 32):
                raise ValueError("检测输入尺寸必须是 32 的倍数，且不超过 4096。")
            if not math.isfinite(confidence) or not math.isfinite(iou) or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
                raise ValueError("置信度与 NMS 阈值应在 0 到 1 之间。")
            max_detections = request.get("max_detections", 300)
            if isinstance(max_detections, bool) or not isinstance(max_detections, int) or not 1 <= max_detections <= 10000:
                raise ValueError("最大检测数量应在 1 到 10000 之间。")
            start = time.perf_counter()
            if command == "infer_batch":
                encoded_images = request.get("images")
                if not isinstance(encoded_images, list) or not 1 <= len(encoded_images) <= 32:
                    raise ValueError("批量图片数量应在 1 到 32 之间。")
                if backend.task != "detect" or not isinstance(backend, (NativeYolo, LegacyYolo)):
                    raise ValueError("当前 TorchScript 或分类后端只支持 batch=1。")
                images = [decode_image(value) for value in encoded_images]
                rows = backend.infer_batch(images, size, confidence, iou, max_detections)
                if len(rows) != len(images):
                    raise ValueError("批量结果数量与输入不一致。")
                reply({"ok": True, "id": request_id, "batch_size": len(images),
                       "results": [{"index": index, "task": backend.task, "predictions": predictions,
                                    "inference_ms": inference_ms, "cpu_threads": torch.get_num_threads()}
                                   for index, (predictions, inference_ms) in enumerate(rows)],
                       "total_ms": (time.perf_counter() - start) * 1000})
                continue
            image = decode_image(request["image"])
            predictions, inference_ms = backend.infer(image, size, confidence, iou, max_detections)
            reply({"ok": True, "id": request_id, "task": backend.task,
                   "predictions": predictions, "inference_ms": inference_ms,
                   "cpu_threads": torch.get_num_threads(),
                   "total_ms": (time.perf_counter() - start) * 1000})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "id": request_id, "error": "PyTorch 推理失败：" + str(error), "detail": ""})


if __name__ == "__main__":
    raise SystemExit(main())
