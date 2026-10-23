#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reanalyze a verified evaluation snapshot without opening its model or dataset.

This command creates a new report. Original requests, predictions, GT snapshots
and figures are never edited. Image overlays still need the original images;
numeric analysis deliberately does not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys

from evaluation_data import digest, validate_manifest
from evaluation_worker import (analyze, atomic_json, export_report,
                               load_predictions, validate_request)

OPTIONS = {"confidence", "match_iou", "scan_start", "scan_end", "scan_step",
           "figure_formats", "figure_dpi"}


def regular_bytes(path):
    """Do not follow a snapshot file or sample-directory link outside the report."""
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("快照文件必须是普通文件，不能使用符号链接：" + path.name)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("快照文件必须是普通文件：" + path.name)
        return stream.read()


def strict_json_bytes(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("快照 JSON 包含重复字段：" + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("快照 JSON 包含非有限数：" + value)

    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("快照文件必须是 JSON 对象。")
    return value


def strict_json(path):
    return strict_json_bytes(regular_bytes(path))


def sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def new_file(path, data):
    # Every copied snapshot uses exclusive creation, including after validation.
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def validate_samples(manifest):
    """Validate cached coordinates rather than trusting a recomputable hash alone."""
    names = manifest.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("快照缺少有效的数据集类别表。")
    for sample in manifest["samples"]:
        if (not isinstance(sample, dict) or not isinstance(sample.get("image"), str) or not sample["image"] or
                not isinstance(sample.get("label"), str) or not sample["label"]):
            raise ValueError("快照样本路径无效。")
        if not all(hash_string(sample.get(key)) for key in ("image_sha256", "label_sha256")):
            raise ValueError("快照缺少有效的图片／标签 SHA256。")
        w, h = sample.get("width"), sample.get("height")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (w, h)):
            raise ValueError("快照图片尺寸无效。")
        if not isinstance(sample.get("gt"), list):
            raise ValueError("快照缺少标注框。")
        for gt in sample["gt"]:
            if not isinstance(gt, dict):
                raise ValueError("快照标注框须为对象。")
            cid, box = gt.get("class_id"), gt.get("box")
            if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(names):
                raise ValueError("快照标注类别超出类别表。")
            if (not isinstance(box, list) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
                not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
                raise ValueError("快照标注坐标无效。")
        flags = sample.get("flags", {})
        if (not isinstance(flags, dict) or any(k not in ("Small", "Occluded", "Blurred", "No Ball") for k in flags) or
                any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1) for v in flags.values())):
            raise ValueError("快照场景标记无效。")
        if manifest.get("metadata_path") and set(flags) != {"Small", "Occluded", "Blurred", "No Ball"}:
            raise ValueError("已审核快照缺少完整场景标记。")
        if "scene_reviewed" in sample and not isinstance(sample["scene_reviewed"], bool):
            raise ValueError("快照审核状态必须为布尔值。")


def reanalyze(source, destination, options):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("新报告须使用原报告目录之外的独立目录。")
    if not isinstance(options, dict) or set(options) - OPTIONS:
        raise ValueError("重分析只允许调整分析阈值、匹配 IoU、扫描范围和图表格式。")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("新报告输出必须是目录。")
        entries = {path.name for path in destination.iterdir()}
        # Qt reserves its request/state and an append-only helper log. No other
        # preexisting output, even an old figure or a dangling link, is writable.
        reserved_entries = entries - {"helper.log"}
        if entries and reserved_entries != {"status.json", "analysis-options.json"}:
            raise ValueError("新报告目录中已有文件，拒绝覆盖。")
        if "helper.log" in entries:
            regular_bytes(destination / "helper.log")
        if entries:
            reserved = strict_json(destination / "status.json")
            if (reserved.get("status") != "analyzing" or reserved.get("completed") != 0 or
                    strict_json(destination / "analysis-options.json") != options):
                raise ValueError("新报告目录的预留分析任务不一致，拒绝覆盖。")

    files = [source / name for name in
             ("report.json", "request.json", "manifest.json", "predictions.jsonl", "status.json")]
    if not stat.S_ISDIR((source / "samples").lstat().st_mode):
        raise ValueError("逐图快照目录必须是普通目录，不能使用符号链接。")
    contents = {path: regular_bytes(path) for path in files}
    hashes = {path: sha256_bytes(data) for path, data in contents.items()}
    original, request, manifest, state = (strict_json_bytes(contents[source / name]) for name in
                                         ("report.json", "request.json", "manifest.json", "status.json"))
    if original.get("schema_version") != 1 or state.get("status") not in ("complete", "cancelled"):
        raise ValueError("仅可重分析已完成保存的完整或中断报告。")
    if original.get("status") != state["status"] or request.get("status") != state["status"]:
        raise ValueError("快照任务状态不一致。")
    if original.get("manifest") != manifest or original.get("request") != request:
        raise ValueError("报告与数据集／请求快照不一致。")
    if original.get("prediction_sha256") != hashes[source / "predictions.jsonl"]:
        raise ValueError("预测快照校验失败，拒绝重新分析。")
    validate_manifest(manifest)
    validate_samples(manifest)
    if (not isinstance(original.get("full_split"), bool) or not isinstance(request.get("full_split"), bool) or
            not hash_string(request.get("model_sha256")) or original.get("model_sha256") != request["model_sha256"] or
            original.get("dataset_fingerprint") != manifest["fingerprint"]):
        raise ValueError("快照模型／数据身份或完整集合声明不一致。")
    # Unknown analysis options are rejected above. Other unknown source fields
    # are retained for forward-compatible metadata, without gaining semantics.
    if request.get("schema_version", 1) != 1 or state.get("schema_version", 1) != 1:
        raise ValueError("快照请求或状态版本不支持。")
    request, _ = validate_request(request, manifest)
    strict_rows = [strict_json_bytes(line) for line in contents[source / "predictions.jsonl"].splitlines()]
    samples, rows = load_predictions(source / "predictions.jsonl", manifest, request)
    if strict_rows != rows:
        raise ValueError("读取期间预测快照发生变化。")
    if not rows:
        raise ValueError("报告中没有已完成图片，无法重新分析。")
    completed = original.get("completed_images")
    if (isinstance(completed, bool) or not isinstance(completed, int) or completed != len(rows) or
            isinstance(state.get("completed"), bool) or state.get("completed") != len(rows) or
            original.get("metrics", {}).get("images") != len(rows)):
        raise ValueError("快照已完成图片数量不一致。")
    if (original.get("planned_images") != len(request["sample_images"]) or
            ("total" in state and state["total"] != len(request["sample_images"])) or
            original.get("sample_images") != [row["image"] for row in rows]):
        raise ValueError("快照图片范围或计划数量不一致。")
    complete = request["status"] == "complete" and request["full_split"] and len(rows) == len(manifest["samples"])
    scope = "single_image" if len(rows) == 1 else "full_split" if complete else "subset"
    if original["full_split"] != complete or original.get("scope") != scope:
        raise ValueError("原报告范围与快照不一致，拒绝提升为完整集合成绩。")

    # Cross-bind three independently persisted representations: manifest GT,
    # streamed predictions and the checksum-protected UI review sample.
    for index, (sample, row) in enumerate(zip(samples, rows)):
        if ("index" in row and (isinstance(row["index"], bool) or row["index"] != index)) or any(
                isinstance(row.get(key), bool) or not isinstance(row.get(key), int) for key in ("width", "height")):
            raise ValueError("预测快照的图片索引或尺寸无效。")
        for key in ("image", "label", "width", "height", "gt", "flags", "image_sha256", "label_sha256"):
            if row.get(key) != sample.get(key):
                raise ValueError("预测与标注快照不一致：" + key)
        cached, checksum = source / "samples" / f"{index}.json", source / "samples" / f"{index}.sha256"
        files.extend((cached, checksum))
        contents[cached], contents[checksum] = regular_bytes(cached), regular_bytes(checksum)
        hashes[cached], hashes[checksum] = sha256_bytes(contents[cached]), sha256_bytes(contents[checksum])
        if contents[checksum] != (hashes[cached] + "\n").encode("ascii") or strict_json_bytes(contents[cached]) != row:
            raise ValueError("逐图快照校验失败。")

    def unchanged():
        for path, expected in hashes.items():
            if sha256_bytes(regular_bytes(path)) != expected:
                raise RuntimeError("重分析期间来源快照发生变化，拒绝发布新报告。")

    unchanged()
    request.update(options)
    profile_blocked = (original.get("allow_apply_profile") is False or request.get("allow_apply_profile") is False)
    if profile_blocked:
        request["allow_apply_profile"] = False
    request, _ = validate_request(request, manifest)
    destination.mkdir(parents=True, exist_ok=True)
    new_file(destination / "manifest.json", contents[source / "manifest.json"])
    new_file(destination / "request.json", (json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    new_file(destination / "predictions.jsonl", contents[source / "predictions.jsonl"])
    (destination / "samples").mkdir()
    for index in range(len(rows)):
        for extension in ("json", "sha256"):
            new_file(destination / "samples" / f"{index}.{extension}", contents[source / "samples" / f"{index}.{extension}"])
    if digest(destination / "predictions.jsonl") != hashes[source / "predictions.jsonl"]:
        raise RuntimeError("复制预测快照后校验失败。")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(destination / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    report = analyze(request, manifest, destination / "predictions.jsonl", verify_sources=False)
    report.update({"directory": str(destination), "config": original.get("config", request.get("config", {})),
                   "model_class_id": request["model_class_id"], "dataset_class_id": request["dataset_class_id"],
                   "analysis_source": "verified_snapshot", "source_report": str(source),
                   "source_report_sha256": hashes[source / "report.json"],
                   "snapshot_hashes": {str(p.relative_to(source)): h for p, h in hashes.items()},
                   "worker_input_hashes": {str(destination / n): digest(destination / n)
                                            for n in ("request.json", "manifest.json")}})
    # External imports intentionally lack enough recorded preprocessing detail
    # to adopt a detector profile. Reanalysis must never erase that restriction.
    for key in ("allow_apply_profile", "analysis_origin", "external_provenance", "source_runtime", "source_arguments"):
        if key in original:
            report[key] = original[key]
    if profile_blocked:
        report["allow_apply_profile"] = False
        report["warnings"].append("来源为外部脚本的单类 football GT（数据集 ID 0），模型原始类别 ID 保持不变；原始后端／预处理未完整核验，禁止采用为检测配置。指标按本版稳定 IoU 并列规则重新计算，可能与来源实现存在差异。")
    report["warnings"].append("本报告由历史 GT／预测快照重新分析，未读取原模型或数据集；图片叠加仍需原图存在且校验一致。")
    pending = destination / ".pending-report.json"
    try:
        export_report(pending, report, verify_sources=False)
        unchanged()
        # Export validates before saving its pending JSON. Verify all copied
        # inputs again at publication, including mutations after that callback.
        for path, expected in hashes.items():
            if path.parent.name == "samples" and sha256_bytes(regular_bytes(destination / "samples" / path.name)) != expected:
                raise RuntimeError("新报告逐图快照在分析期间发生变化。")
        inputs = {destination / "predictions.jsonl": hashes[source / "predictions.jsonl"],
                  **{Path(path): expected for path, expected in report["worker_input_hashes"].items()}}
        for path, expected in inputs.items():
            if sha256_bytes(regular_bytes(path)) != expected:
                raise RuntimeError("新报告预测、请求或数据集快照在导出后发生变化。")
        os.replace(pending, destination / "report.json")
    finally:
        if pending.exists():
            pending.unlink()
    atomic_json(destination / "status.json", {"schema_version": 1, "status": request["status"],
                "completed": len(rows), "total": len(request["sample_images"]), "directory": str(destination),
                "analysis_source": "verified_snapshot"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--options", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = reanalyze(args.source, args.output, strict_json(args.options))
        print(json.dumps({"success": True, "completed_images": report["completed_images"],
                          "output": str(Path(args.output).resolve())}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
