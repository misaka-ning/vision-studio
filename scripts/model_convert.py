#!/usr/bin/env python3
"""Offline, transactional PT/TorchScript -> ONNX/TorchScript conversion.

CLI stdout is protocol-1 JSONL; framework/native stdout goes to stderr. The
job_temp status is emitted before importing any third-party module. A GUI
controller first creates and pins its exact --work-dir under the canonical
output parent; it never chooses a cleanup target from a worker event. Normal failures and TERM/INT clean
up the job themselves. No existing output, selected source, shared runtime or
user configuration is modified. ONNX uses the explicit PyTorch 2.9 legacy
exporter (dynamo=False), fixed FP32 batch=1 and opset 12/17 without NMS.

API references: https://docs.pytorch.org/docs/2.9/onnx.html,
https://docs.pytorch.org/docs/2.9/generated/torch.jit.trace.html,
https://onnx.ai/onnx/api/checker.html. This helper does not import pt_worker.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import sys
import tempfile
import traceback
import zipfile

PROTOCOL = 1
JOB_PREFIX = ".vision-studio-convert-"
MAX_EVENT_BYTES = 1024 * 1024
MAX_NAMES_BYTES = 128 * 1024
SOURCE_SUFFIXES = {".pt", ".torchscript"}
OUTPUT_SUFFIXES = {"onnx": {".onnx"}, "torchscript": {".torchscript", ".pt"}}


class ConversionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ConversionCancelled(Exception):
    pass


def require(condition, code, message):
    if not condition:
        raise ConversionError(code, message)


def event_bytes(kind, message, **fields):
    require(kind in ("progress", "status", "result", "error"), "protocol_error", "事件类型无效。")
    value = {"protocol": PROTOCOL, "type": kind, "message": str(message), **fields}
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    require(len(raw) <= MAX_EVENT_BYTES, "metadata_too_large", "模型元数据过大，无法通过转换协议交付。")
    return raw


class Protocol:
    def __init__(self):
        self.descriptor = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr

    def emit(self, kind, message, **fields):
        remaining = memoryview(event_bytes(kind, message, **fields))
        while remaining:
            written = os.write(self.descriptor, remaining)
            remaining = remaining[written:]

    def close(self):
        os.close(self.descriptor)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ConversionError("invalid_arguments", "转换参数无效：" + message)


def parser():
    value = JsonArgumentParser(description=__doc__)
    value.add_argument("--source", required=True)
    value.add_argument("--format", choices=("onnx", "torchscript"), required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--image-size", type=int, default=640)
    value.add_argument("--opset", type=int, choices=(12, 17), default=12)
    value.add_argument("--work-dir", default="", help="Optional precreated empty owned job directory under output parent")
    return value


def validate_paths(arguments):
    require(32 <= arguments.image_size <= 4096 and arguments.image_size % 32 == 0,
            "invalid_image_size", "输入尺寸应为 32 到 4096 之间的 32 倍数。")
    source = Path(arguments.source).expanduser().resolve()
    require(source.is_file() and stat.S_ISREG(source.stat().st_mode),
            "source_missing", "找不到可读取的本地 PT 或 TorchScript 模型。")
    require(source.suffix.lower() in SOURCE_SUFFIXES and source.stat().st_size > 0,
            "invalid_source", "请选择非空的 .pt 或 .torchscript 模型。")
    # Preserve a dangling output symlink as well as a regular existing file.
    original_output = Path(arguments.output).expanduser().absolute()
    require(not original_output.exists() and not original_output.is_symlink(),
            "output_exists", "输出文件已存在，请选择新文件名；转换不会覆盖现有文件。")
    require(original_output.parent.is_dir() and original_output.parent.resolve() == original_output.parent,
            "invalid_output_directory", "输出父目录必须是已存在的真实目录，不能经过符号链接。")
    output = original_output.parent / original_output.name
    require(output.suffix.lower() in OUTPUT_SUFFIXES[arguments.format], "invalid_output_suffix",
            "输出扩展名与转换格式不匹配。ONNX 使用 .onnx，TorchScript 使用 .torchscript 或 .pt。")
    require(output != source, "source_is_output", "输出文件必须与原模型不同。")
    require(os.access(source, os.R_OK), "source_unreadable", "没有读取所选模型的权限。")
    require(os.access(output.parent, os.W_OK | os.X_OK), "output_unwritable", "输出目录不可写。")
    return source, output


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, file_hash(path))


class JobDirectory:
    def __init__(self, output, provided=""):
        self.parent = output.parent
        self.path = Path(provided).expanduser().absolute() if provided else Path(
            tempfile.mkdtemp(prefix=JOB_PREFIX, dir=self.parent))
        require(self.path.parent == self.parent and self.path.resolve() == self.path
                and self.path.name.startswith(JOB_PREFIX) and self.path.is_dir() and not self.path.is_symlink(),
                "invalid_work_directory", "转换临时目录必须是输出父目录内的本轮真实目录。")
        value = self.path.lstat()
        require(value.st_uid == os.geteuid() and not any(self.path.iterdir()),
                "invalid_work_directory", "转换临时目录必须属于当前用户且为空；不会清理已有资料。")
        self.identity = (value.st_dev, value.st_ino, value.st_uid)

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_value, tb):
        require(self.path.parent == self.parent and self.parent.resolve() == self.parent
                and not self.path.is_symlink(), "work_directory_changed", "转换临时目录身份改变，已拒绝清理。")
        value = self.path.lstat()
        require((value.st_dev, value.st_ino, value.st_uid) == self.identity and stat.S_ISDIR(value.st_mode),
                "work_directory_changed", "转换临时目录已被替换，已拒绝清理。")
        require(shutil.rmtree.avoids_symlink_attacks, "unsupported_filesystem", "当前平台缺少安全的临时目录清理接口。")
        shutil.rmtree(self.path)


@contextmanager
def isolate_environment(job):
    original_environment = os.environ.copy()
    original_temp = tempfile.tempdir
    original_bytecode = sys.dont_write_bytecode
    # Every writable framework/config/cache/temp location belongs to this job.
    for variable, name in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"),
                           ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache"),
                           ("XDG_STATE_HOME", "state"), ("YOLO_CONFIG_DIR", "ultralytics"),
                           ("YOLOV5_CONFIG_DIR", "yolov5"), ("MPLCONFIGDIR", "matplotlib"),
                           ("TORCH_HOME", "torch"), ("TMPDIR", "tmp")):
        destination = job / name
        destination.mkdir()
        os.environ[variable] = str(destination)
    tempfile.tempdir = str(job / "tmp")
    os.environ.update(YOLO_AUTOINSTALL="false", YOLOv5_AUTOINSTALL="false", YOLO_OFFLINE="true",
                      YOLOv5_VERBOSE="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
        tempfile.tempdir = original_temp
        sys.dont_write_bytecode = original_bytecode


def dependency(name):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ConversionError("missing_dependency", "转换运行环境缺少 " + name
                              + "，请使用提供模型转换支持的应用环境。不会自动安装依赖。") from error


def labels_list(names):
    if isinstance(names, dict):
        require(set(str(key) for key in names) == {str(index) for index in range(len(names))},
                "invalid_metadata", "模型类别索引必须从 0 连续排列。")
        names = [names.get(index, names.get(str(index))) for index in range(len(names))]
    require(isinstance(names, (list, tuple)) and 0 < len(names) <= 100000,
            "invalid_metadata", "模型缺少有效类别名称，裸权重无法直接转换。")
    result = [str(name) if name is not None else "" for name in names]
    require(all(name and len(name) <= 1000 for name in result), "invalid_metadata", "模型类别名称无效。")
    require(len(json.dumps(result, ensure_ascii=False).encode()) <= MAX_NAMES_BYTES,
            "metadata_too_large", "模型类别信息过大，无法安全交付转换结果。")
    return result


def checked_channels(value):
    require(isinstance(value, int) and not isinstance(value, bool) and value in (1, 3),
            "unsupported_channels", "仅支持实际输入为 1 或 3 通道的模型。")
    return value


def module_channels(model, torch):
    convolution = next((module for module in model.modules() if isinstance(module, torch.nn.Conv2d)), None)
    actual = convolution.in_channels if convolution is not None else None
    stem = getattr(model, "model", None)
    if isinstance(stem, (torch.nn.Sequential, torch.nn.ModuleList)) and len(stem) \
            and stem[0].__class__.__name__ == "Focus" and actual is not None:
        require(actual % 4 == 0, "unsupported_channels", "YOLOv5 Focus 输入层的通道约束无效。")
        actual //= 4
    configuration = getattr(model, "yaml", {})
    declared = configuration.get("channels", configuration.get("ch")) if isinstance(configuration, dict) else None
    return checked_channels(actual if actual is not None else declared)


def script_channels(model, metadata, torch):
    # A convolution is evidence only if its input traces unchanged to the image.
    graph = model.inlined_graph
    inputs = list(graph.inputs())
    images = [value for value in inputs if isinstance(value.type(), torch.TensorType)]
    inferred = set()
    if len(images) == 1:
        image = images[0]
        sizes = image.type().sizes()
        if sizes is not None and len(sizes) == 4 and sizes[1] is not None:
            inferred.add(sizes[1])
        preserving = {"aten::to", "aten::contiguous", "aten::detach", "aten::clone", "aten::cpu",
                      "aten::type_as", "aten::div", "aten::mul", "aten::sub", "aten::add"}

        def unchanged(value):
            if value == image:
                return True
            node = value.node()
            if node.kind() not in preserving:
                return False
            arguments = list(node.inputs())
            if node.kind() in {"aten::div", "aten::mul", "aten::sub", "aten::add"} and len(arguments) > 1 \
                    and isinstance(arguments[1].type(), torch.TensorType):
                return False
            return bool(arguments) and unchanged(arguments[0])

        def attribute(value):
            if value == inputs[0]:
                return model
            node = value.node()
            if node.kind() == "prim::GetAttr":
                return getattr(attribute(list(node.inputs())[0]), node.s("name"))
            return value.toIValue()

        for node in graph.nodes():
            if node.kind() not in ("aten::_convolution", "aten::conv2d", "aten::convolution"):
                continue
            values = list(node.inputs())
            if not values or not unchanged(values[0]):
                continue
            try:
                weight = attribute(values[1])
                groups = values[6 if node.kind() == "aten::conv2d" else 8].toIValue()
                if node.kind() != "aten::conv2d" and values[6].toIValue() is not False:
                    continue
                if isinstance(weight, torch.Tensor) and weight.ndim == 4 and isinstance(groups, int) \
                        and not isinstance(groups, bool) and groups > 0:
                    inferred.add(int(weight.shape[1]) * groups)
            except (AttributeError, RuntimeError, TypeError, IndexError):
                continue
    require(len(inferred) <= 1, "invalid_metadata", "TorchScript 输入层的通道约束不一致。")
    declared = metadata.get("input_channels", metadata.get("ch"))
    if declared is None and isinstance(metadata.get("shape"), (list, tuple)) and len(metadata["shape"]) == 4:
        declared = metadata["shape"][1]
    actual = next(iter(inferred), None)
    require(actual is None or declared is None or actual == declared,
            "invalid_metadata", "TorchScript 通道元数据与实际输入层不一致。")
    return checked_channels(actual if actual is not None else declared)


def is_torchscript(path):
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any("/code/" in name for name in archive.namelist())


def exact_checkpoint_loader(source):
    vendor = Path(__file__).resolve().parent.parent / "vendor/yolov5"
    if (vendor / "models/yolo.py").is_file():
        sys.path.insert(0, str(vendor))
    from ultralytics.nn import tasks
    from ultralytics.utils import downloads

    def selected(file, *args, **kwargs):
        candidate = Path(str(file)).expanduser().resolve()
        require(candidate == source and candidate.is_file(), "source_substitution",
                "模型框架请求替换所选权重，已拒绝；仅转换指定的本地文件。")
        return str(candidate)

    def disabled(*args, **kwargs):
        raise ConversionError("network_disabled", "本地模型转换禁止自动下载模型或资源。")

    downloads.attempt_download_asset = selected
    downloads.safe_download = downloads.download = disabled
    import torch
    torch.hub.download_url_to_file = torch.hub.load_state_dict_from_url = disabled
    loader = getattr(tasks, "load_checkpoint", None) or getattr(tasks, "attempt_load_one_weight", None)
    require(loader is not None, "unsupported_runtime", "当前 Ultralytics 没有兼容的本地检查点加载接口。")
    return loader


def load_model(source, torch):
    if is_torchscript(source):
        extra = {"config.txt": ""}
        model = torch.jit.load(str(source), map_location="cpu", _extra_files=extra).float().eval()
        try:
            metadata = json.loads(extra["config.txt"]) if extra["config.txt"] else {}
        except (ValueError, UnicodeDecodeError) as error:
            raise ConversionError("invalid_metadata", "TorchScript config.txt 不是有效 JSON。") from error
        require(isinstance(metadata, dict) and metadata.get("task") in ("detect", "classify"),
                "invalid_metadata", "TorchScript 需要 config.txt 中的 detect/classify 任务与类别元数据。")
        return model, metadata["task"], labels_list(metadata.get("names", metadata.get("classes"))), \
            script_channels(model, metadata, torch), metadata.get("layout"), "torchscript"
    dependency("ultralytics")
    try:
        model, checkpoint = exact_checkpoint_loader(source)(str(source), device=torch.device("cpu"), fuse=False)
    except (ConversionError, ConversionCancelled):
        raise
    except Exception as error:
        raise ConversionError("model_load_failed", "无法加载此本地检查点。需要完整的支持架构；裸 state_dict "
                              "还需要网络定义。详细原因：" + str(error)[:1000]) from error
    require(Path(str(getattr(model, "pt_path", ""))).resolve() == source,
            "source_substitution", "框架没有加载所选文件，已拒绝继续转换。")
    namespace = model.__class__.__module__
    require(isinstance(model, torch.nn.Module) and namespace.startswith(("models.", "ultralytics.")),
            "unsupported_model", "仅支持完整的 Ultralytics YOLO、旧 YOLOv5 或带元数据的 TorchScript。")
    task = getattr(model, "task", "detect")
    require(task in ("detect", "classify") and (not namespace.startswith("models.") or task == "detect"),
            "unsupported_task", "当前转换支持标准 YOLO 检测与现代分类；不支持分割、姿态或旋转框。")
    model = model.cpu().float().eval()
    for module in model.modules():
        name = module.__class__.__name__
        require(name not in ("Segment", "Pose", "OBB", "WorldDetect", "YOLOEDetect"),
                "unsupported_task", "该模型含非标准任务或额外输入，不能生成当前支持的单输入模型。")
        if name == "Detect":
            require(not getattr(module, "end2end", False), "unsupported_output",
                    "当前不支持带端到端后处理的检测头，请使用原始 YOLO 检测输出模型。")
            module.export = True
            module.dynamic = False
            module.format = "onnx"
            module.inplace = False
            if hasattr(module, "xyxy"):
                module.xyxy = False
            if namespace.startswith("models."):
                if not isinstance(getattr(module, "anchor_grid", None), list):
                    # Older checkpoints registered this as a Tensor buffer;
                    # remove that registration before adapting the v7 loader.
                    if hasattr(module, "anchor_grid"):
                        delattr(module, "anchor_grid")
                    module.anchor_grid = [torch.zeros(1)] * module.nl
                if not isinstance(getattr(module, "grid", None), list):
                    module.grid = [torch.zeros(1)] * module.nl
        if name == "Classify":
            module.export = True
        if isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model, task, labels_list(getattr(model, "names", None)), module_channels(model, torch), \
        ("v5" if namespace.startswith("models.") else "v8"), "checkpoint"


def primary_output(output, torch):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ConversionError("unsupported_output", "模型没有返回标准原始检测或分类张量。")


def check_output(output, task, names, torch):
    require(output.dtype == torch.float32 and torch.isfinite(output).all().item(),
            "unsupported_output", "模型输出必须是有限数值的 FP32 张量。")
    if task == "classify":
        require(tuple(output.shape) in ((1, len(names)), (1, len(names), 1, 1)),
                "unsupported_output", "分类输出应为 [1, 类别数] 或 [1, 类别数, 1, 1]。")
        return "classify"
    require(output.ndim == 3 and output.shape[0] == 1, "unsupported_output",
            "检测输出应为单张原始 YOLO 张量，不支持内置 NMS 或多输出任务。")
    if output.shape[1] == len(names) + 4 and 0 < output.shape[2] <= 2000000:
        return "v8"
    if output.shape[2] == len(names) + 5 and 0 < output.shape[1] <= 2000000:
        return "v5"
    raise ConversionError("unsupported_output", "检测输出形状与类别元数据不匹配，不能用于本工具。")


def register_detection_caches(model, torch):
    """Keep warmed fixed-shape caches as movable buffers in the traced file.

    Plain tensors become CPU CONSTANTS in torch.jit.trace. Modern Detect uses
    anchors/strides attributes; legacy Detect uses tensor lists for grid and
    anchor_grid. Register the exact list tensor objects too so tracing resolves
    their existing identities through GetAttr rather than freezing CPU data.
    This only changes the freshly loaded conversion model, never its source.
    """
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "Detect" or not module.__class__.__module__.startswith(
                ("models.", "ultralytics.")):
            continue
        for name in ("anchors", "strides", "stride"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and name not in module._buffers:
                # Detect also declares anchors/strides as class attributes;
                # retain its public attribute and register an identity alias.
                # The tracer maps that tensor to the buffer's GetAttr path.
                buffer_name = "_vision_static_" + name
                require(not hasattr(module, buffer_name), "unsupported_model",
                        "检测头包含与转换缓存重名的属性，已取消转换。")
                module.register_buffer(buffer_name, value)
                count += 1
        for name in ("grid", "anchor_grid"):
            values = getattr(module, name, None)
            if not isinstance(values, (list, tuple)):
                continue
            for index, value in enumerate(values):
                if not isinstance(value, torch.Tensor):
                    continue
                buffer_name = "_vision_static_" + name + "_" + str(index)
                require(not hasattr(module, buffer_name), "unsupported_model",
                        "检测头包含与转换缓存重名的属性，已取消转换。")
                module.register_buffer(buffer_name, value)
                count += 1
    print("TorchScript portability: registered " + str(count) + " detection cache buffers.", file=sys.stderr)


def trace_model(model, example, task, names, torch):
    class Primary(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return primary_output(self.network(image), torch)

    wrapper = Primary(model).eval()
    with torch.inference_mode():
        expected = wrapper(example)
        layout = check_output(expected, task, names, torch)
        register_detection_caches(model, torch)
        # Integer nearest-neighbor scales are exact. During tracing only, use
        # them directly instead of exporting scalar Shape/Floor arithmetic
        # from old recompute_scale_factor=True checkpoints (OpenCV cannot
        # import that scalar Floor path). Restore the original modules before
        # comparing the graph with the selected model's actual eager output.
        adapted = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Upsample) or module.mode != "nearest" \
                    or module.size is not None or not getattr(module, "recompute_scale_factor", False):
                continue
            factors = module.scale_factor if isinstance(module.scale_factor, tuple) else (module.scale_factor,)
            if factors and all(isinstance(value, (int, float)) and value > 0 and float(value).is_integer()
                               for value in factors):
                adapted.append((module, module.recompute_scale_factor))
                module.recompute_scale_factor = False
        # aten::_convolution stores allow_tf32 as a trace-time constant.
        # A CPU trace made under the default True setting would silently use
        # reduced-precision TF32 after the saved model moves to CUDA, even if
        # its caller disables TF32. Export a portable, strict FP32 graph.
        allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        finally:
            torch.backends.cudnn.allow_tf32 = allow_tf32
            for module, recompute in adapted:
                module.recompute_scale_factor = recompute
        probe = torch.linspace(0, 1, example.numel(), dtype=torch.float32).reshape_as(example)
        for image in (example, probe):
            eager, actual = wrapper(image), traced(image)
            require(check_output(actual, task, names, torch) == layout
                    and torch.allclose(eager, actual, rtol=2e-4, atol=1e-4),
                    "verification_failed", "静态导出与原始模型的测试输出不一致，已取消发布文件。")
    return traced, layout, tuple(expected.shape)


def metadata_for(task, labels, channels, size, layout):
    names = {str(index): name for index, name in enumerate(labels)}
    return {"version": 1, "modelType": "classification" if task == "classify" else "yolov5" if layout == "v5" else "yolov8",
            "task": task, "names": names, "classes": names, "input_channels": channels, "ch": channels,
            "shape": [1, channels, size, size], "layout": layout, "stride": 32,
            "nms": False, "precision": "fp32", "batch": 1, "dynamic": False}


def fold_static_shape_constants(model, onnx):
    """Evaluate only shape/constant subgraphs with ONNX's own evaluator."""
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {value.name: tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)
              for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
              if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")
              and all(dimension.HasField("dim_value") and dimension.dim_value > 0
                      for dimension in value.type.tensor_type.shape.dim)}
    constants = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
    opsets = {value.domain: value.version for value in model.opset_import}
    operations = {"Constant", "Identity", "Gather", "Add", "Sub", "Mul", "Div", "Cast", "Floor", "Ceil",
                  "Unsqueeze", "Squeeze", "Concat", "Slice", "Reshape", "ConstantOfShape", "Equal", "Where",
                  "Expand", "Transpose", "Size"}
    retained = []
    folded = 0
    for node in model.graph.node:
        values = None
        if node.op_type == "Shape" and node.domain in ("", "ai.onnx") and node.input[0] in shapes:
            attributes = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
            shape = shapes[node.input[0]]
            values = [np.asarray(shape[attributes.get("start", 0):attributes.get("end", len(shape))], dtype=np.int64)]
        elif node.op_type in operations and node.domain in ("", "ai.onnx") \
                and all(name in constants for name in node.input if name):
            # Avoid materializing unbounded constants from an untrusted graph.
            bounded = all(value not in shapes or np.prod(shapes[value], dtype=object) <= 2000000
                          for value in node.output)
            if node.op_type in ("ConstantOfShape", "Expand"):
                shape = constants[node.input[-1]]
                bounded = bounded and shape.size <= 16 and all(int(size) >= 0 for size in shape.reshape(-1)) \
                    and np.prod(shape, dtype=object) <= 2000000
            if bounded:
                values = ReferenceEvaluator(node, opsets=opsets).run(None, {name: constants[name] for name in node.input if name})
        if values is None or len(values) != len(node.output) \
                or any(not isinstance(value, np.ndarray) or value.size > 2000000 for value in values):
            retained.append(node)
            continue
        for name, value in zip(node.output, values):
            constants[name] = value
            shapes[name] = tuple(value.shape)
            model.graph.initializer.append(onnx.numpy_helper.from_array(value, name=name))
        folded += 1
    del model.graph.node[:]
    model.graph.node.extend(retained)
    print("OpenCV compatibility: folded " + str(folded) + " exact static/constant nodes.", file=sys.stderr)
    return model


def expand_static_broadcasts(model, onnx):
    """Express fixed broadcast copies as Concat for the OpenCV 4.5 importer."""
    import numpy as np
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {value.name: tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)
              for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
              if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")
              and all(dimension.HasField("dim_value") and dimension.dim_value > 0
                      for dimension in value.type.tensor_type.shape.dim)}
    constants = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
    retained = []
    for node in model.graph.node:
        if node.op_type != "Expand" or node.input[0] not in shapes or node.input[1] not in constants:
            retained.append(node)
            continue
        shape = shapes[node.input[0]]
        target = tuple(int(value) for value in constants[node.input[1]].reshape(-1))
        require(0 < len(target) <= 8 and all(value > 0 for value in target),
                "unsupported_output", "ONNX 广播形状无效。")
        broadcast = np.broadcast_shapes(shape, target)
        require(len(shape) == len(broadcast), "unsupported_output", "当前不支持增加维数的 Expand 广播。")
        current = node.input[0]
        copies = [(axis, size) for axis, (initial, size) in enumerate(zip(shape, broadcast)) if initial != size]
        require(all(shape[axis] == 1 and size <= 4096 for axis, size in copies),
                "unsupported_output", "ONNX 广播超出当前固定输入支持范围。")
        for index, (axis, size) in enumerate(copies):
            name = node.output[0] if index == len(copies) - 1 else node.output[0] + "__broadcast_" + str(axis)
            retained.append(onnx.helper.make_node("Concat", [current] * size, [name], axis=axis,
                                                 name=node.name + "__concat_" + str(axis)))
            current = name
        if not copies:
            retained.append(onnx.helper.make_node("Identity", [current], list(node.output), name=node.name + "__identity"))
    del model.graph.node[:]
    model.graph.node.extend(retained)
    return model


def canonicalize_broadcast_multipliers(model, onnx):
    """Keep constant multiplication on OpenCV's Scale channel axis.

    OpenCV 4.5's parseMul flattens all constant operands and defaults to axis=1.
    Uniform constants can be scalars (Power). For a proven final-axis vector,
    temporarily move that axis to channel=1, multiply, and invert the same
    transpose. ONNX broadcasting and every float multiplier remain identical.
    """
    import numpy as np
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {value.name: tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)
              for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
              if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")
              and all(dimension.HasField("dim_value") and dimension.dim_value > 0
                      for dimension in value.type.tensor_type.shape.dim)}
    initializers = {value.name: value for value in model.graph.initializer}
    retained = []
    for node in model.graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            retained.append(node)
            continue
        indexes = [index for index, name in enumerate(node.input) if name in initializers]
        if len(indexes) != 1:
            retained.append(node)
            continue
        index = indexes[0]
        array = onnx.numpy_helper.to_array(initializers[node.input[index]])
        data = node.input[1 - index]
        name = node.output[0] + "__multiplier"
        while name in initializers:
            name += "_"
        shape = shapes.get(data)
        # Scalarizing a uniform tensor is safe only when that tensor does not
        # itself expand any dimension (or introduce a new leading dimension).
        same_shape = shape is not None and array.ndim <= len(shape) \
            and np.broadcast_shapes(shape, array.shape) == shape
        if same_shape and array.size > 1 and np.all(array == array.reshape(-1)[0]):
            copied = onnx.numpy_helper.from_array(np.asarray(array.reshape(-1)[0]), name=name)
            model.graph.initializer.append(copied)
            initializers[name] = copied
            node.input[index] = name
            retained.append(node)
            continue
        if same_shape and len(shape) >= 3 and array.ndim > 0 and array.shape[-1] == shape[-1] \
                and all(dimension == 1 for dimension in array.shape[:-1]):
            rank = len(shape)
            permutation = [0, rank - 1] + list(range(1, rank - 1))
            inverse = [permutation.index(axis) for axis in range(rank)]
            transposed = node.output[0] + "__channel_axis"
            multiplied = node.output[0] + "__scaled"
            copied = onnx.numpy_helper.from_array(array.reshape((shape[-1],) + (1,) * (rank - 2)), name=name)
            model.graph.initializer.append(copied)
            initializers[name] = copied
            retained.extend((onnx.helper.make_node("Transpose", [data], [transposed], perm=permutation,
                                                  name=node.name + "__to_channel"),
                             onnx.helper.make_node("Mul", [transposed, name], [multiplied], name=node.name + "__scale"),
                             onnx.helper.make_node("Transpose", [multiplied], list(node.output), perm=inverse,
                                                  name=node.name + "__from_channel")))
        else:
            retained.append(node)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    return model


def canonicalize_constant_left_subtractions(model, onnx):
    """Avoid OpenCV 4.5 reversing constant-left Sub in its Scale importer.

    For floating tensors c - x is exactly c + (-1 * x). Multiplying by -1
    uses a scalar Power layer, while addition is commutative. Keep all runtime
    shapes and broadcast dimensions unchanged; do not rewrite integer types.
    """
    import numpy as np
    initializers = {value.name: value for value in model.graph.initializer}
    reserved = set(initializers) | {name for node in model.graph.node for name in node.output}
    retained = []
    changed = 0
    for node in model.graph.node:
        if node.domain or node.op_type != "Sub" or len(node.input) != 2 or len(node.output) != 1 \
                or node.input[0] not in initializers or node.input[1] in initializers \
                or initializers[node.input[0]].data_type not in (
                    onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE):
            retained.append(node)
            continue
        scalar_name = node.output[0] + "__negative_one"
        while scalar_name in reserved:
            scalar_name += "_"
        reserved.add(scalar_name)
        negative_name = node.output[0] + "__negative_rhs"
        while negative_name in reserved:
            negative_name += "_"
        reserved.add(negative_name)
        dtype = onnx.numpy_helper.to_array(initializers[node.input[0]]).dtype
        model.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(-1, dtype=dtype), scalar_name))
        retained.extend((onnx.helper.make_node("Mul", [node.input[1], scalar_name], [negative_name],
                                               name=node.name + "__negate_rhs"),
                         onnx.helper.make_node("Add", [node.input[0], negative_name], list(node.output),
                                               name=node.name + "__add_lhs")))
        changed += 1
    del model.graph.node[:]
    model.graph.node.extend(retained)
    print("OpenCV compatibility: rewrote " + str(changed) + " constant-left Sub nodes.", file=sys.stderr)
    return model


def prune_unused_constants(model):
    used = {name for node in model.graph.node for name in node.input}
    used.update(value.name for value in model.graph.output)
    values = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(values)
    return model


def make_opencv_compatible(model, onnx):
    """Exact graph rewrites for the application's OpenCV 4.5 importer.

    Torch exports shared initializer tensors through Identity aliases; old
    OpenCV treats those constant aliases as layers with no runtime input.
    Negative Unsqueeze axes also need their equivalent positive form. Neither
    transformation changes weights or math; no inferred values are invented.
    """
    constants = {value.name for value in model.graph.initializer}
    aliases = {}
    outputs = {value.name for value in model.graph.output}

    def original(name):
        while name in aliases:
            name = aliases[name]
        return name

    retained = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            constants.update(node.output)
        if node.op_type == "Identity" and len(node.input) == len(node.output) == 1 \
                and original(node.input[0]) in constants and node.output[0] not in outputs:
            aliases[node.output[0]] = original(node.input[0])
        else:
            retained.append(node)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = original(name)
    information = [value for value in model.graph.value_info if value.name not in aliases]
    del model.graph.value_info[:]
    model.graph.value_info.extend(information)
    # Folding a shape expression makes downstream Slice/Reshape dimensions
    # inferable. Repeat inference to propagate only proven dimensions through
    # successive C2f/DFL blocks; stop as soon as the graph stops changing.
    for _ in range(32):
        previous = len(model.graph.node)
        model = fold_static_shape_constants(model, onnx)
        if len(model.graph.node) == previous:
            break
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks = {value.name: len(value.type.tensor_type.shape.dim)
             for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
             if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")}
    ranks.update({value.name: len(value.dims) for value in model.graph.initializer})
    initializers = {value.name: value for value in model.graph.initializer}
    normalized = 0
    for node in model.graph.node:
        if node.op_type != "Unsqueeze" or not node.input or node.input[0] not in ranks:
            continue
        attribute = next((value for value in node.attribute if value.name == "axes"), None)
        if attribute is not None:
            axes = list(attribute.ints)
        elif len(node.input) == 2 and node.input[1] in initializers:
            axes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()
        else:
            continue
        rank = ranks[node.input[0]] + len(axes)
        if not any(axis < 0 for axis in axes):
            continue
        positive = [axis + rank if axis < 0 else axis for axis in axes]
        require(all(0 <= axis < rank for axis in positive) and len(set(positive)) == len(positive),
                "verification_failed", "ONNX Unsqueeze 维度核验失败。")
        if attribute is not None:
            del attribute.ints[:]
            attribute.ints.extend(positive)
        else:
            name = node.output[0] + "__positive_axes"
            while name in initializers:
                name += "_"
            value = onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [len(positive)], positive)
            model.graph.initializer.append(value)
            initializers[name] = value
            node.input[1] = name
        normalized += 1
    print("OpenCV compatibility: removed " + str(len(aliases)) + " constant Identity aliases, normalized "
          + str(normalized) + " Unsqueeze axes.", file=sys.stderr)
    model = canonicalize_constant_left_subtractions(model, onnx)
    return prune_unused_constants(canonicalize_broadcast_multipliers(expand_static_broadcasts(model, onnx), onnx))


def export_onnx(traced, example, path, metadata, output_shape, opset, torch, onnx):
    with torch.inference_mode():
        torch.onnx.export(traced, (example,), str(path), export_params=True,
                          input_names=["images"], output_names=["output0"], opset_version=opset,
                          dynamo=False, external_data=False, dynamic_axes=None,
                          keep_initializers_as_inputs=False)
    model = make_opencv_compatible(onnx.load(str(path), load_external_data=False), onnx)
    require(not any(tensor.external_data for tensor in model.graph.initializer),
            "unsupported_external_data", "模型需要外部权重文件；当前转换只交付单个模型文件。")
    onnx.helper.set_model_props(model, {key: value if isinstance(value, str) else
                                      json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                      for key, value in metadata.items()})
    onnx.checker.check_model(model, full_check=True)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    require(len(inputs) == 1 and len(model.graph.output) == 1,
            "verification_failed", "ONNX 必须保留一个图像输入和一个原始输出。")
    for value, dimensions in ((inputs[0], metadata["shape"]), (model.graph.output[0], output_shape)):
        tensor = value.type.tensor_type
        require(tensor.elem_type == onnx.TensorProto.FLOAT
                and [dimension.dim_value for dimension in tensor.shape.dim] == list(dimensions)
                and not any(dimension.dim_param for dimension in tensor.shape.dim),
                "verification_failed", "ONNX 的 FP32 或静态输入输出形状核验失败。")
    require(not any(node.op_type == "NonMaxSuppression" for node in model.graph.node),
            "unsupported_output", "输出包含 NMS，无法用于当前原始 YOLO 输出约定。")
    onnx.save_model(model, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path), full_check=True)
    cv2 = dependency("cv2")
    import numpy as np
    try:
        network = cv2.dnn.readNetFromONNX(str(path))
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        probe = torch.linspace(0, 1, example.numel(), dtype=torch.float32).reshape_as(example)
        with torch.inference_mode():
            for image in (example, probe):
                network.setInput(image.numpy())
                actual = network.forward()
                expected = traced(image).numpy()
                require(actual.shape == expected.shape and np.isfinite(actual).all()
                        and np.allclose(actual, expected, rtol=2e-3, atol=2e-3),
                        "verification_failed", "ONNX 实际 OpenCV 推理与原模型不一致，已取消输出。")
    except cv2.error as error:
        raise ConversionError("opencv_incompatible", "导出图无法在当前 OpenCV CPU 后端运行，已取消输出："
                              + str(error)[:1200]) from error


def export_torchscript(traced, example, path, metadata, output_shape, torch):
    torch.jit.save(traced, str(path), _extra_files={"config.txt": json.dumps(metadata, ensure_ascii=False)})
    extra = {"config.txt": ""}
    verified = torch.jit.load(str(path), map_location="cpu", _extra_files=extra).eval()
    require(json.loads(extra["config.txt"]) == metadata, "verification_failed", "TorchScript 元数据保存核验失败。")
    with torch.inference_mode():
        actual = verified(example)
        require(tuple(actual.shape) == output_shape and actual.dtype == torch.float32
                and torch.allclose(traced(example), actual, rtol=2e-4, atol=1e-4),
                "verification_failed", "TorchScript 文件重载后的输出核验失败。")


def publish_no_replace(staged, output):
    require(staged.is_file() and not staged.is_symlink(), "verification_failed", "缺少已验证的临时模型。")
    with staged.open("rb") as stream:
        os.fsync(stream.fileno())
    # Anchor the destination directory before linking, so a rename/symlink
    # swap after this check cannot redirect the committed file elsewhere.
    require(output.parent.resolve() == output.parent, "invalid_output_directory", "输出父目录身份发生变化。")
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.link(staged, output.name, dst_dir_fd=descriptor, follow_symlinks=False)
        os.fsync(descriptor)
    except FileExistsError as error:
        raise ConversionError("output_exists", "输出文件已被创建，请选择新文件名；已有内容没有覆盖。") from error
    except OSError as error:
        raise ConversionError("publication_failed", "无法无覆盖地发布转换文件，请选择可写的本地目录。") from error
    finally:
        os.close(descriptor)


def convert(arguments, emit):
    source, output = validate_paths(arguments)
    original = source_identity(source)
    published_identity = None
    try:
        with JobDirectory(output, getattr(arguments, "work_dir", "")) as job:
            value = job.lstat()
            emit("status", "已创建独立转换任务。", job_temp=str(job), source=str(source), output=str(output),
                 job_identity={"device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid})
            with isolate_environment(job):
                result, published_identity = run_conversion(arguments, source, output, job, original, emit)
        emit("result", "模型转换完成。", **result)
        published_identity = None
        return result
    except BaseException:
        # Remove only our exact committed inode, never an existing/replaced file.
        if published_identity is not None and output.exists() and not output.is_symlink():
            value = output.stat()
            if (value.st_dev, value.st_ino) == published_identity:
                output.unlink()
        raise


def run_conversion(arguments, source, output, job, original, emit):
    published_identity = None
    try:
        emit("progress", "正在加载本地转换运行环境。", percent=5)
        torch = dependency("torch")
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        onnx = dependency("onnx") if arguments.format == "onnx" else None
        emit("progress", "正在读取所选模型。", percent=15)
        model, task, labels, channels, _, source_format = load_model(source, torch)
        example = torch.zeros(1, channels, arguments.image_size, arguments.image_size, dtype=torch.float32)
        emit("progress", "正在验证输入通道并生成静态计算图。", percent=35)
        traced, layout, output_shape = trace_model(model, example, task, labels, torch)
        metadata = metadata_for(task, labels, channels, arguments.image_size, layout)
        staged = job / ("model.onnx" if arguments.format == "onnx" else "model.torchscript")
        emit("progress", "正在导出并核验模型文件。", percent=60)
        if arguments.format == "onnx":
            export_onnx(traced, example, staged, metadata, output_shape, arguments.opset, torch, onnx)
        else:
            export_torchscript(traced, example, staged, metadata, output_shape, torch)
        require(source_identity(source) == original, "source_changed", "转换期间原模型发生变化，已取消输出。")
        result = {"source": str(source), "source_format": source_format, "output": str(output),
                  "format": arguments.format, "sha256": file_hash(staged), "bytes": staged.stat().st_size,
                  "opset": arguments.opset if arguments.format == "onnx" else None, **metadata}
        event_bytes("result", "模型转换完成。", **result)  # size/JSON check before committing anything
        emit("progress", "核验通过，正在保存转换结果。", percent=95)
        # Save only one validated artifact. Atomic link has no overwrite race.
        value = staged.stat()
        published_identity = (value.st_dev, value.st_ino)
        publish_no_replace(staged, output)
        return result, published_identity
    except BaseException:
        # A failure between commit and the success event can roll back only
        # our exact inode, never a preexisting/replaced user file.
        if published_identity is not None and output.exists() and not output.is_symlink():
            value = output.stat()
            if (value.st_dev, value.st_ino) == published_identity:
                output.unlink()
        raise

def main(argv=None):
    protocol = Protocol()
    old_handlers = {}

    def cancelled(signum, frame):
        raise ConversionCancelled("模型转换已取消。")

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, cancelled)
        arguments = parser().parse_args(argv)
        convert(arguments, protocol.emit)
        return 0
    except ConversionCancelled:
        protocol.emit("error", "模型转换已取消，原模型和已有输出保持不变。", code="cancelled")
        return 130
    except ConversionError as error:
        protocol.emit("error", str(error)[:2000], code=error.code)
        return 1
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        protocol.emit("error", "转换失败：" + str(error)[:2000], code="conversion_failed")
        return 1
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        protocol.close()


if __name__ == "__main__":
    sys.exit(main())
