# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only YOLO evaluation inputs for Vision Studio.

Adapted from the maintainer's model-evaluation/dataset_io.py: keep the reviewed
football metadata rules, but accept normal multi-class YOLO datasets without
private project roots or mandatory metadata. Missing annotations are never
silently interpreted as negative samples. No model or dataset is downloaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCENES = ("Small", "Occluded", "Blurred", "No Ball")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value):
    # QJson serializes an exactly integral double such as 22.0 as JSON 22.
    # Canonicalize that semantic equivalence without rounding fractional values;
    # both writers round-trip their IEEE-754 doubles with shortest precision.
    def normalize(item):
        if isinstance(item, float) and math.isfinite(item) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {key: normalize(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(normalize(value), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def absolute_dataset_path(value):
    value = str(value)
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def resolve_path(value, base):
    path = Path(str(value)).expanduser()
    return (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_dataset_yaml(value):
    """Resolve a dataset root without guessing classes, splits or nested roots.

    Training recipes often sit beside a dataset configuration. Prefer one
    conventional dataset filename, but never choose between multiple such
    files. Explicit YAML inputs retain their existing behavior.
    """
    path = resolve_path(value, Path.cwd())
    if not path.is_dir():
        return path
    candidates = sorted((child for child in path.iterdir()
                         if child.is_file() and child.suffix.lower() in (".yaml", ".yml")),
                        key=lambda child: (child.name.casefold(), child.name))
    conventional = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}
    preferred = [child for child in candidates if child.name.casefold() in conventional]
    if len(preferred) == 1:
        return preferred[0].resolve()
    if len(preferred) > 1:
        raise ValueError("文件夹中存在多个标准数据集配置（" + "、".join(child.name for child in preferred)
                         + "），请使用“选择 YAML”入口明确选择。")
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ValueError("所选文件夹本层没有 YAML 数据集配置；请选择含 data.yaml／dataset.yaml 的数据集根目录，"
                         "或使用“选择 YAML”入口。配置须包含 names 和 train／val／test 集合路径。")
    raise ValueError("文件夹中存在多个 YAML 配置（" + "、".join(child.name for child in candidates)
                     + "），请使用“选择 YAML”入口明确选择数据集配置。")


def read_json(path):
    def no_constant(value):
        raise ValueError(f"JSON 不允许非有限数：{value}")
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=no_constant)


def class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            if isinstance(key, bool) or not re.fullmatch(r"0|[1-9][0-9]*", str(key)):
                raise ValueError("数据集 names 的 ID 必须是从 0 开始的连续整数。")
            index = int(key)
            if index in normalized:
                raise ValueError("数据集 names 含重复 ID。")
            normalized[index] = value
        if set(normalized) != set(range(len(normalized))):
            raise ValueError("数据集 names 的 ID 必须是从 0 开始的连续整数。")
        names = [normalized[i] for i in range(len(normalized))]
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("数据集 YAML 必须提供非空的 names 类别名称列表或连续 ID 映射。")
    if "nc" in config and (isinstance(config["nc"], bool) or config["nc"] != len(names)):
        raise ValueError("数据集 nc 与 names 类别数量不一致。")
    return names


def dataset_entry_path(value, base, relocation=None):
    path = resolve_path(value, base)
    if relocation is not None and absolute_dataset_path(value) and not path.exists():
        previous_root, selected_root = relocation
        try:
            relative = path.relative_to(previous_root)
        except ValueError:
            pass
        else:
            path = (selected_root / relative).resolve()
    return path


def list_images(entry, root, source_hashes, relocation=None):
    images = []
    for value in entry if isinstance(entry, list) else [entry]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("集合路径必须是图片目录、图片名单或它们的列表。")
        path = dataset_entry_path(value, root, relocation)
        if path.is_dir():
            images.extend(sorted((p.resolve() for p in path.rglob("*")
                                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=str))
        elif path.is_file() and path.suffix.lower() == ".txt":
            source_hashes[path.as_posix()] = digest(path)
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    images.append(dataset_entry_path(line.strip(), path.parent, relocation))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"集合路径不存在或不是图片目录／名单：{path}")
    if len(set(images)) != len(images):
        raise ValueError("集合中存在重复图片路径，请修正目录／名单后重试。")
    for path in images:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"图片名单含缺失文件或非图片：{path}")
        if any(c in path.as_posix() for c in ("\n", "\r", "*")):
            raise ValueError("图片路径不能包含换行或通配符 *。")
    if not images:
        raise ValueError("选择的数据集集合没有图片。")
    return images


def label_path(image):
    parts = list(Path(image).parts)
    positions = [i for i, value in enumerate(parts) if value == "images"]
    if not positions:
        raise ValueError(f"图片须遵循 images/… → labels/… 的 YOLO 目录约定：{image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(path, width, height, count):
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or not re.fullmatch(r"0|[1-9][0-9]*", fields[0]):
            raise ValueError(f"仅支持 class_id x_center y_center width height 检测标注：{path}:{number}")
        class_id = int(fields[0])
        if class_id >= count:
            raise ValueError(f"标注类别 ID {class_id} 超出数据集 names 范围：{path}:{number}")
        try:
            x, y, w, h = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"标注坐标不是数字：{path}:{number}") from exc
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"标注含非有限数或无效宽高：{path}:{number}")
        normalized = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        if min(normalized) < -1e-6 or max(normalized) > 1 + 1e-6:
            raise ValueError(f"标注框超出图片范围：{path}:{number}")
        clipped = [min(1., max(0., v)) for v in normalized]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"标注框裁剪到图片后面积为零：{path}:{number}")
        boxes.append({"class_id": class_id, "box": [clipped[0] * width, clipped[1] * height,
                      clipped[2] * width, clipped[3] * height]})
    return boxes


def read_metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"图片", "split", "annotation_status", "label_review_required", "scene_review_required", *SCENES}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"审核 metadata 缺少字段：{sorted(required - set(reader.fieldnames or []))}")
        rows = {}
        for row in reader:
            if not row["图片"] or row["图片"] in rows:
                raise ValueError("审核 metadata 含空图片名或重复图片名。")
            rows[row["图片"]] = row
    return rows


def prepare_dataset(request):
    from PIL import Image
    import yaml

    if not isinstance(request, dict) or not request.get("dataset_yaml"):
        raise ValueError("请选择本地 YOLO 数据集 YAML。")
    dataset_source = resolve_path(request["dataset_yaml"], Path.cwd())
    data = resolve_dataset_yaml(dataset_source)
    source_hashes = {data.as_posix(): digest(data)}
    with data.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("数据集配置必须是 YAML 映射。")
    names = class_names(config)
    root_value = config.get("path", ".")
    root = resolve_path(root_value, data.parent)
    relocation = None
    if (dataset_source.is_dir() and dataset_source == data.parent
            and absolute_dataset_path(root_value) and not root.exists()):
        # Selecting a directory explicitly identifies the new dataset root.
        # Only a missing absolute configured root is eligible; direct YAML
        # inputs and missing paths outside that root remain strict failures.
        relocation = (root, dataset_source)
        root = dataset_source
    split = request.get("split", "val")
    if split not in ("train", "val", "test") or not config.get(split):
        raise ValueError(f"数据集未提供有效的 {split} 集合。")
    images = list_images(config[split], root, source_hashes, relocation)
    metadata_value = request.get("metadata_path") or request.get("metadata")
    metadata_path = resolve_path(metadata_value, data.parent) if metadata_value else None
    if metadata_path:
        source_hashes[metadata_path.as_posix()] = digest(metadata_path)
    metadata = read_metadata(metadata_path) if metadata_path else None
    if metadata is not None and len(names) != 1:
        raise ValueError("足球审核 metadata 适配只支持单类数据集；普通多类 YOLO 数据集无需提供 metadata。")
    if metadata_path:
        if len({p.name for p in images}) != len(images):
            raise ValueError("提供审核 metadata 时，不同目录不能存在同名图片。")

    samples, skipped = [], []
    missing = 0
    for image in images:
        flags = {key: 0 for key in SCENES}
        row = metadata.get(image.name) if metadata is not None else None
        if metadata is not None:
            if row is None:
                raise ValueError(f"图片未登记在审核 metadata：{image.name}")
            reason = None
            if row["annotation_status"] == "reviewed_excluded":
                reason = "reviewed_excluded"
            elif row["label_review_required"] != "0" or row["scene_review_required"] != "0":
                reason = "review_pending"
            elif row["annotation_status"] not in ("reviewed_annotation", "negative_confirmed"):
                reason = "not_confirmed_usable"
            if reason:
                skipped.append({"image": image.as_posix(), "reason": reason})
                continue
            if row["split"] != split:
                raise ValueError(f"metadata.split 与选择集合不一致：{image.name}")
            if any(row[key] not in ("0", "1") for key in SCENES):
                raise ValueError(f"审核场景值必须是 0 或 1：{image.name}")
            flags = {key: int(row[key]) for key in SCENES}
        label = label_path(image)
        if not label.is_file():
            if relocation is not None:
                raise ValueError("数据集只读重定位后仍存在缺失标签，请修复所选文件夹中的 images／labels 配对后重新导入。")
            skipped.append({"image": image.as_posix(), "label": label.as_posix(), "reason": "missing_label"})
            missing += 1
            continue
        image_sha256, label_sha256 = digest(image), digest(label)
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        # PNG getexif() may load pixels and close its file handle; verification
        # must use the freshly opened image, before reading EXIF metadata.
        with Image.open(image) as im:
            orientation = im.getexif().get(274, 1)
            if orientation != 1:
                raise ValueError(f"图片含非标准 EXIF 方向，请先统一图片与标注方向再导入：{image}")
        if width <= 0 or height <= 0:
            raise ValueError(f"图片尺寸无效：{image}")
        boxes = read_boxes(label, width, height, len(names))
        if row is not None:
            if flags["No Ball"] != int(not boxes):
                raise ValueError(f"metadata 的 No Ball 与最终标签不一致：{image.name}")
            expected = "reviewed_annotation" if boxes else "negative_confirmed"
            if row["annotation_status"] != expected:
                raise ValueError(f"审核状态与标注框数量不一致：{image.name}")
        else:
            flags["No Ball"] = int(not boxes)
        samples.append({"image": image.as_posix(), "label": label.as_posix(), "width": width, "height": height,
                        "gt": boxes, "flags": flags, "image_sha256": image_sha256,
                        "label_sha256": label_sha256})
    manifest = {"schema_version": 1, "data_yaml": data.as_posix(), "dataset_yaml": data.as_posix(),
                "dataset_source": dataset_source.as_posix(),
                "root": root.as_posix(), "split": split, "names": names,
                "metadata_path": metadata_path.as_posix() if metadata_path else None,
                "samples": samples, "skipped": skipped, "source_hashes": source_hashes,
                "available_images": len(images), "selected_images": len(samples),
                "requires_attention": bool(missing), "validation": {
                    "positive_images": sum(bool(s["gt"]) for s in samples),
                    "negative_images": sum(not s["gt"] for s in samples),
                    "missing_labels": missing, "excluded": len(skipped) - missing}}
    if relocation is not None:
        manifest.update({"root_relocated": True, "configured_root": relocation[0].as_posix(),
                         "relocation_notice": "原配置中的绝对根目录已失效，按所选文件夹只读重定位；"
                                              "配置、图片名单与标注均未修改。"})
    manifest["fingerprint"] = canonical_digest(manifest)
    assert_unchanged(manifest)
    return manifest


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("数据集清单格式不支持。")
    expected = manifest.get("fingerprint")
    if not expected or canonical_digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != expected:
        raise ValueError("数据集清单内容或校验指纹已变化。")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len({s.get("image") for s in samples}) != len(samples):
        raise ValueError("数据集清单包含重复图片或格式错误。")


def assert_unchanged(manifest, model_path=None, model_sha256=None):
    for path, expected in manifest.get("source_hashes", {}).items():
        if not Path(path).is_file() or digest(path) != expected:
            raise RuntimeError(f"评估期间数据集配置／名单／metadata 发生变化：{path}")
    for sample in manifest.get("samples", []):
        for key in ("image", "label"):
            path = sample[key]
            if not Path(path).is_file() or digest(path) != sample[key + "_sha256"]:
                raise RuntimeError(f"评估期间原图或标注发生变化：{path}")
    if model_path:
        if not model_sha256 or not Path(model_path).is_file() or digest(model_path) != model_sha256:
            raise RuntimeError("评估期间模型内容发生变化，不能作为有效评估发布。")
