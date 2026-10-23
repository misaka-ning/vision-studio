#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Import a completed model-evaluation snapshot without opening model/data files.

Only the four documented JSON/JSONL products are read; no code from the external
project is imported or executed. All numeric products are freshly calculated by
Vision Studio. Missing preprocessing/backend details never become an applicable
workbench profile.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PureWindowsPath
import stat
import sys

from evaluation_data import canonical_digest, validate_manifest
from evaluation_snapshot import (OPTIONS, hash_string, new_file, regular_bytes,
                                 sha256_bytes, strict_json, strict_json_bytes,
                                 validate_samples)
from evaluation_worker import analyze, atomic_json, export_report, validate_request

SOURCE_FILES = ("status.json", "metrics.json", "inputs_manifest.json", "predictions.jsonl")
SCENES = ("Small", "Occluded", "Blurred", "No Ball")


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " 必须是有效范围内的整数。")
    return value


def number(value, name, minimum=0., maximum=1., positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            value < minimum or value > maximum or (positive and value == minimum)):
        raise ValueError(name + " 必须是有效范围内的有限数。")
    return value


def path_text(value, name):
    # Keep saved paths unchanged: resolving them could follow current data links
    # and would misrepresent the historical source. No file at these paths opens.
    if (not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n") or
            not (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())):
        raise ValueError(name + " 必须是原清单中的有效绝对路径。")
    return value


def model_names(value):
    if isinstance(value, dict):
        if not value or any(str(index) not in value for index in range(len(value))) or set(value) != {str(i) for i in range(len(value))}:
            raise ValueError("外部模型类别表必须使用连续的原始 ID。")
        value = [value[str(index)] for index in range(len(value))]
    if (not isinstance(value, list) or not 1 <= len(value) <= 100000 or
            any(not isinstance(name, str) or not name.strip() or len(name) > 1000 for name in value)):
        raise ValueError("外部报告缺少有效的模型原始类别表。")
    return value


def boxes(value, width, height, name):
    if not isinstance(value, list):
        raise ValueError(name + " 必须是 xyxy 坐标列表。")
    for box in value:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(name + " 必须是 N×4 的 xyxy 坐标。")
        for coordinate in box:
            number(coordinate, name, maximum=max(width, height))
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(name + " 含无效或超出原图范围的框。")
    return value


def flags(value):
    if not isinstance(value, dict) or set(value) != set(SCENES):
        raise ValueError("外部快照必须含完整的四种审核场景标记。")
    for scene in SCENES:
        integer(value[scene], "场景标记 " + scene, maximum=1)
    return value


def validate_source(contents):
    """Return an app manifest/request/rows plus unmodified source provenance."""
    state, summary, source = (strict_json_bytes(contents[name]) for name in SOURCE_FILES[:3])
    if state.get("status") != "complete" or summary.get("status") != "complete":
        raise ValueError("只能导入已完整保存为 complete 的外部评估。")
    if state.get("input_files_verified_unchanged") is not True or summary.get("input_files_verified_unchanged") is not True:
        raise ValueError("来源未记录输入文件完整性验收，拒绝导入。")
    args = source.get("arguments")
    if not isinstance(args, dict) or args != summary.get("arguments"):
        raise ValueError("外部指标与清单中的推理参数不一致。")
    if args.get("dry_run") is not False:
        raise ValueError("外部任务必须是实际推理完成的任务，不能是 dry-run。")
    model = path_text(source.get("model"), "模型路径")
    if model != summary.get("model") or not hash_string(source.get("model_sha256")):
        raise ValueError("外部模型路径或 SHA256 身份不一致。")
    names = model_names(source.get("model_names"))
    class_id = integer(source.get("selected_model_class_id"), "模型原始类别 ID", maximum=len(names) - 1)
    if integer(summary.get("model_class_id"), "指标原始类别 ID") != class_id:
        raise ValueError("外部指标与清单选中的原始模型类别不一致。")
    requested_id = args.get("class_id")
    if requested_id is not None:
        if integer(requested_id, "命令行原始类别 ID") != class_id:
            raise ValueError("外部命令行与已采集类别不一致。")
    elif len(names) != 1:
        raise ValueError("多类别外部模型必须显式声明所选原始类别 ID。")
    split = args.get("split")
    if split not in ("train", "val", "test") or split != summary.get("split"):
        raise ValueError("外部报告的 Train／Val／Test 集合声明不一致。")
    minimum = number(args.get("min_conf"), "候选采集下限", positive=True)
    confidence = number(args.get("conf"), "原分析置信度", minimum=minimum)
    match_iou = number(args.get("match_iou"), "匹配 IoU", positive=True)
    nms_iou = number(args.get("nms_iou"), "NMS IoU", positive=True)
    max_det = integer(args.get("max_det"), "每图最大检测数量", minimum=1)
    requested_batch = integer(args.get("batch"), "请求 batch", minimum=1)
    requested_threads = integer(args.get("threads"), "请求 CPU 线程", minimum=1)
    requested_size = integer(args.get("imgsz"), "请求输入尺寸", minimum=1)
    limit = args.get("limit")
    if limit is not None:
        integer(limit, "调试图片数量上限", minimum=1)
    selected, skipped = source.get("selected"), source.get("skipped")
    if not isinstance(selected, list) or not selected or not isinstance(skipped, list):
        raise ValueError("外部报告缺少固定图片范围或排除名单。")
    available = integer(summary.get("available_before_limit"), "限制前有效图片数", minimum=1)
    partial = summary.get("partial_split")
    if not isinstance(partial, bool) or partial != (len(selected) < available):
        raise ValueError("外部完整／子集声明与限制前图片数不一致。")
    if len(selected) != (min(limit, available) if limit is not None else available):
        raise ValueError("外部 --limit 与选中图片数量不一致，不能提升为完整集合。")
    if integer(summary.get("skipped_images"), "排除图片数量") != len(skipped):
        raise ValueError("外部排除图片数量与名单不一致。")
    skipped_paths = set()
    for item in skipped:
        if (not isinstance(item, dict) or item.get("reason") not in
                ("reviewed_excluded", "review_pending", "not_confirmed_usable")):
            raise ValueError("外部排除名单的原因无效。")
        path = path_text(item.get("image"), "排除图片路径")
        if path in skipped_paths:
            raise ValueError("外部排除名单包含重复图片。")
        skipped_paths.add(path)
    data, metadata = path_text(source.get("data"), "数据集 YAML"), path_text(source.get("metadata"), "审核 metadata")
    if data == metadata:
        raise ValueError("数据集 YAML 与审核 metadata 路径不能相同。")
    if not all(hash_string(source.get(key)) for key in ("dataset_yaml_sha256", "metadata_sha256")):
        raise ValueError("外部清单缺少有效的数据集／审核文件 SHA256。")
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("外部报告缺少真实运行时记录。")
    if not isinstance(runtime.get("device"), str) or not runtime["device"]:
        raise ValueError("外部报告缺少实际设备记录。")
    actual_threads = integer(runtime.get("torch_threads"), "实际 Torch 线程数", minimum=1)
    for key in ("backend_fp16", "backend_end2end"):
        if not isinstance(runtime.get(key), bool):
            raise ValueError("外部报告缺少真实后端模式：" + key)
    shape = runtime.get("actual_input_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("外部实际输入尺寸无效。")
    for value in shape:
        integer(value, "实际输入尺寸", minimum=1)
    if runtime.get("loader_batch_size") is not None:
        integer(runtime["loader_batch_size"], "来源加载器 batch", minimum=1)
    if not isinstance(runtime.get("image_loader"), str) or not runtime["image_loader"]:
        raise ValueError("外部报告缺少实际图片加载器记录。")
    for key in ("predict_wall_seconds", "predict_images_per_second_including_warmup"):
        if key in runtime:
            number(runtime[key], "原始运行耗时／吞吐", maximum=1e30, positive=True)
    if not isinstance(runtime.get("backend_reported_mean_ms", {}), dict):
        raise ValueError("外部运行耗时记录无效。")
    for value in runtime.get("backend_reported_mean_ms", {}).values():
        number(value, "外部后端平均耗时", maximum=1e30)
    code_hashes = source.get("evaluation_code_sha256")
    if (not isinstance(code_hashes, dict) or not code_hashes or
            any(not isinstance(key, str) or Path(key).name != key or not key.endswith(".py") or not hash_string(value)
                for key, value in code_hashes.items())):
        raise ValueError("外部评估代码 SHA256 清单无效。")
    if not isinstance(summary.get("versions"), dict):
        raise ValueError("外部报告缺少原始依赖版本记录。")

    raw_rows = [strict_json_bytes(line) for line in contents["predictions.jsonl"].splitlines()]
    if len(raw_rows) != len(selected):
        raise ValueError("外部预测缓存与选中图片数量不一致。")
    samples, converted, seen = [], [], set()
    for index, (chosen, row) in enumerate(zip(selected, raw_rows)):
        if set(row) != {"image", "width", "height", "gt", "flags", "boxes", "scores"}:
            raise ValueError("外部预测不是受支持的单目标快照格式，不能猜测缺失或额外类别字段。")
        if not isinstance(chosen, dict):
            raise ValueError("外部图片清单项目必须是对象。")
        image = path_text(chosen.get("image"), "图片路径")
        if image in seen or image in skipped_paths or row.get("image") != image:
            raise ValueError("外部图片重复、被排除或预测顺序不一致。")
        seen.add(image)
        label = path_text(chosen.get("label"), "标签路径")
        width = integer(chosen.get("width"), "图片宽度", minimum=1)
        height = integer(chosen.get("height"), "图片高度", minimum=1)
        if integer(row.get("width"), "预测宽度", minimum=1) != width or integer(row.get("height"), "预测高度", minimum=1) != height:
            raise ValueError("外部预测坐标系与图片清单尺寸不一致。")
        if not all(hash_string(chosen.get(key)) for key in ("image_sha256", "label_sha256")):
            raise ValueError("外部图片／标签缺少有效 SHA256。")
        gt = boxes(row.get("gt"), width, height, "GT")
        if integer(chosen.get("gt_boxes"), "GT 框数量") != len(gt):
            raise ValueError("外部 GT 数量与清单不一致。")
        scene = flags(chosen.get("scenes"))
        if flags(row.get("flags")) != scene or scene["No Ball"] != int(not gt):
            raise ValueError("外部审核场景与预测／GT 快照不一致。")
        predicted = boxes(row.get("boxes"), width, height, "预测")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(predicted) or len(scores) > max_det:
            raise ValueError("外部预测框／分数数量或 max_det 不一致。")
        for score in scores:
            number(score, "外部预测分数", minimum=minimum)
        # The source JSONL deliberately omits per-prediction classes: evaluate.py
        # filtered every box to selected_model_class_id before saving. Bind that
        # documented schema to names/args/summary rather than renumbering IDs.
        sample = {"image": image, "label": label, "width": width, "height": height,
                  "gt": [{"class_id": 0, "box": box} for box in gt], "flags": scene,
                  "scene_reviewed": True, "image_sha256": chosen["image_sha256"],
                  "label_sha256": chosen["label_sha256"]}
        result = {**sample, "index": index, "predictions": [
            {"class_id": class_id, "label": names[class_id], "confidence": score, "box": box}
            for box, score in zip(predicted, scores)], "backend": "Ultralytics / imported snapshot",
                  "source_runtime_device": runtime["device"]}
        samples.append(sample)
        converted.append(result)
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or integer(metrics.get("images"), "指标图片数") != len(samples):
        raise ValueError("外部指标图片数与完整预测缓存不一致。")
    gt_total = sum(len(sample["gt"]) for sample in samples)
    predicted_total = sum(sum(score >= confidence for score in row["scores"]) for row in raw_rows)
    if integer(metrics.get("gt_boxes"), "指标 GT 数量") != gt_total or integer(metrics.get("predicted_boxes"), "工作点预测数") != predicted_total:
        raise ValueError("外部指标 GT／工作点预测数量与快照不一致。")
    tp, fp, fn = (integer(metrics.get(key), "外部指标 " + key) for key in ("TP", "FP", "FN"))
    if tp + fn != gt_total or tp + fp != predicted_total:
        raise ValueError("外部 TP／FP／FN 计数不一致。")
    for key, expected in (("confidence", confidence), ("match_iou", match_iou)):
        if key in metrics and number(metrics[key], "来源指标 " + key) != expected:
            raise ValueError("外部工作点指标与来源参数不一致。")
    negative = sum(not row["gt"] for row in raw_rows)
    if "no_ball_images" in metrics and integer(metrics["no_ball_images"], "来源阴性图片数") != negative:
        raise ValueError("外部阴性图片数量与 GT 快照不一致。")

    manifest = {"schema_version": 1, "data_yaml": data, "dataset_yaml": data, "dataset_source": data,
                "root": None, "split": split, "names": ["football"], "metadata_path": metadata,
                "samples": samples, "skipped": skipped,
                "source_hashes": {data: source["dataset_yaml_sha256"], metadata: source["metadata_sha256"]},
                "available_images": available + len(skipped), "selected_images": len(samples),
                "requires_attention": False,
                "validation": {"positive_images": sum(bool(sample["gt"]) for sample in samples),
                               "negative_images": sum(not sample["gt"] for sample in samples),
                               "missing_labels": 0, "excluded": len(skipped)},
                "external_available_before_limit": available,
                "dataset_class_name_source": "model-evaluation single-class football protocol"}
    manifest["fingerprint"] = canonical_digest(manifest)
    validate_manifest(manifest)
    validate_samples(manifest)
    device = runtime["device"]
    resolved = "cpu" if device == "cpu" else "cuda" if device.startswith("cuda:") else "unknown"
    config = {"model_path": model, "labels": names, "model_class_count": len(names),
              "task": "external_detection", "class_filter": [class_id], "confidence": minimum,
              "nms_iou": nms_iou, "max_detections": max_det,
              "cpu_threads": requested_threads, "resolved_cpu_threads": actual_threads,
              "resolved_device": resolved, "external_runtime_device": device,
              "input_size": shape[0] if shape[0] == shape[1] else None,
              "preprocessing_verified_for_workbench": False}
    request = {"schema_version": 1, "model_path": model, "model_sha256": source["model_sha256"],
               "model_class_id": class_id, "model_class_count": len(names), "dataset_class_id": 0,
               "confidence": confidence, "match_iou": match_iou, "score_floor": minimum,
               "scan_start": .05, "scan_end": .95, "scan_step": .01,
               "status": "complete", "full_split": not partial, "sample_images": [sample["image"] for sample in samples],
               "requested_batch_size": requested_batch, "config": config, "allow_apply_profile": False,
               "figure_formats": ["png", "svg", "pdf"], "figure_dpi": 300,
               "batch_notice": "外部原件未逐批记录模型执行大小；加载器 batch 仅保留在 source_runtime，不推断实际模型 batch。"}
    return manifest, request, converted, summary, source


def import_report(source, destination, options):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("导入须新建原报告目录之外的独立目录。")
    if not isinstance(options, dict) or set(options) - OPTIONS:
        raise ValueError("导入只允许调整分析阈值、匹配 IoU、扫描范围和图表格式。")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("导入输出必须是新目录。")
        entries = {item.name for item in destination.iterdir()}
        reserved_files = {"status.json", "analysis-options.json"}
        if entries and entries not in (reserved_files, reserved_files | {"helper.log"}):
            raise ValueError("导入目录已有文件，拒绝覆盖。")
        if entries:
            reserved = strict_json(destination / "status.json")
            if (reserved.get("status") != "analyzing" or reserved.get("completed") != 0 or
                    strict_json(destination / "analysis-options.json") != options):
                raise ValueError("导入目录的预留后台任务不一致。")
            if "helper.log" in entries and not stat.S_ISREG((destination / "helper.log").lstat().st_mode):
                raise ValueError("预留 helper.log 必须是普通文件，不能是符号链接。")
    contents = {name: regular_bytes(source / name) for name in SOURCE_FILES}
    hashes = {name: sha256_bytes(data) for name, data in contents.items()}
    manifest, request, rows, original, original_manifest = validate_source(contents)
    request.update(options)
    request, _ = validate_request(request, manifest)

    def unchanged():
        for name, expected in hashes.items():
            if sha256_bytes(regular_bytes(source / name)) != expected:
                raise RuntimeError("导入期间外部来源快照发生变化，拒绝发布报告。")

    unchanged()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "external-source").mkdir()
    local_hashes = {}
    for name, data in contents.items():
        new_file(destination / "external-source" / name, data)
        local_hashes["external-source/" + name] = hashes[name]
    def encoded(value):
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for name, value in (("manifest.json", manifest), ("request.json", request)):
        data = encoded(value)
        new_file(destination / name, data)
        local_hashes[name] = sha256_bytes(data)
    predictions = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in rows)
    new_file(destination / "predictions.jsonl", predictions)
    local_hashes["predictions.jsonl"] = sha256_bytes(predictions)
    (destination / "samples").mkdir()
    for index, row in enumerate(rows):
        data = encoded(row)
        new_file(destination / "samples" / f"{index}.json", data)
        new_file(destination / "samples" / f"{index}.sha256", (sha256_bytes(data) + "\n").encode("ascii"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(destination / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    report = analyze(request, manifest, destination / "predictions.jsonl", verify_sources=False)
    report.update({"directory": str(destination), "config": request["config"],
                   "model_class_id": request["model_class_id"], "dataset_class_id": 0,
                   "allow_apply_profile": False, "analysis_source": "external_model_evaluation",
                   "analysis_origin": "external_model_evaluation", "source_runtime": original["runtime"],
                   "source_arguments": original["arguments"],
                   "external_provenance": {"format": "model-evaluation/evaluate.py snapshot", "source_directory": str(source),
                       "source_files_sha256": hashes, "evaluation_code_sha256": original_manifest["evaluation_code_sha256"],
                       "versions": original["versions"], "created_at_utc": original.get("created_at_utc"),
                       "original_metrics": original["metrics"], "original_scene_metrics": original.get("scene_metrics"),
                       "available_before_limit": original["available_before_limit"],
                       "partial_split": original["partial_split"], "skipped_images": original["skipped_images"]},
                   "worker_input_hashes": {str(destination / name): local_hashes[name]
                                           for name in ("request.json", "manifest.json")}})
    report["warnings"].extend([
        "本报告只读导入外部脚本的已完成 GT／预测快照，未运行模型或读取当前数据集。",
        "数据集原始 ID 0 按来源工具的单类 football 协议映射；模型原始类别 ID 保持不变，未重新编号。",
        "指标使用本版稳定 IoU 并列匹配重新计算，来源原指标完整保留于 external_provenance；不承诺逐位一致。",
        "外部后端／预处理与本应用未逐项核验，禁止采用为检测配置；未知颜色、通道、均值、缩放和布局不作推断。",
        "错例叠加仍需原图存在且 SHA256 一致；这里的来源哈希只固定所导入原件，不代替重新核验历史模型／数据内容。",
    ])
    if report["source_runtime"]["torch_threads"] != report["source_arguments"]["threads"]:
        report["warnings"].append("来源实际 Torch 线程数与请求不同；分别保留真实记录，不能把请求值当作实测值。")
    pending = destination / ".pending-report.json"
    export_report(pending, report, verify_sources=False)
    unchanged()
    for name, expected in local_hashes.items():
        if sha256_bytes(regular_bytes(destination / name)) != expected:
            raise RuntimeError("导入目标快照在保存期间发生变化：" + name)
    # Verify checksum-protected UI samples against the exact converted snapshot.
    for index, row in enumerate(rows):
        data = regular_bytes(destination / "samples" / f"{index}.json")
        checksum = regular_bytes(destination / "samples" / f"{index}.sha256")
        if strict_json_bytes(data) != row or checksum != (sha256_bytes(data) + "\n").encode("ascii"):
            raise RuntimeError("导入的逐图快照在保存期间发生变化。")
    os.replace(pending, destination / "report.json")
    atomic_json(destination / "status.json", {"schema_version": 1, "status": "complete", "completed": len(rows),
                "total": len(rows), "directory": str(destination), "analysis_source": "external_model_evaluation"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--options", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = import_report(args.source, args.output, strict_json(args.options))
        print(json.dumps({"success": True, "completed_images": report["completed_images"],
                          "allow_apply_profile": False, "output": str(Path(args.output).resolve())}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
