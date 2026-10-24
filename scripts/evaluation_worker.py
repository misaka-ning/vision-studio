#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Vision Studio read-only dataset preparation and backend-neutral metrics CLI.

The maintainer's model-evaluation reporting/threshold scanner supplies the
metric definitions. This adapter removes private project paths and inference
dependencies: C++/existing PT backends own inference, this subprocess only
validates snapshots and computes the same metrics for all backends.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from evaluation_data import (assert_unchanged, digest, prepare_dataset, read_json,
                             validate_manifest)
from evaluation_reporting import figure_options, save_report_artifacts, write_csv

ALGORITHM_VERSION = "vision-eval-1/highest-iou-stable-ties"


def finite_number(value, name, minimum=0., maximum=1., positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字。")
    value = float(value)
    if not math.isfinite(value) or value < minimum or value > maximum or (positive and value <= minimum):
        raise ValueError(f"{name} 超出有效范围。")
    return value


def nonnegative_id(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是从 0 开始的整数 ID。")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_request(request, manifest):
    from evaluation_metrics import threshold_grid
    if not isinstance(request, dict):
        raise ValueError("评估请求必须为 JSON 对象。")
    request = dict(request)
    request["model_class_id"] = nonnegative_id(request.get("model_class_id", 0), "模型类别")
    request["dataset_class_id"] = nonnegative_id(request.get("dataset_class_id", 0), "数据集类别")
    if request["dataset_class_id"] >= len(manifest["names"]):
        raise ValueError("数据集类别 ID 超出实际 names 范围。")
    status = request.get("status", "complete")
    if request.get("model_class_count") is not None:
        count = nonnegative_id(request["model_class_count"], "模型类别数量")
        if (not count and status != "cancelled") or (count and request["model_class_id"] >= count):
            raise ValueError("模型类别 ID 超出实际模型类别范围。")
    request["confidence"] = finite_number(request.get("confidence", .25), "分析置信度")
    request["match_iou"] = finite_number(request.get("match_iou", .5), "匹配 IoU", positive=True)
    request["score_floor"] = finite_number(request.get("score_floor", .001), "候选最低置信度", positive=True)
    grid = threshold_grid(request.get("scan_start", .05), request.get("scan_end", .95), request.get("scan_step", .01))
    if min(grid[0], request["confidence"]) < request["score_floor"]:
        raise ValueError("分析／扫描阈值低于候选缓存采集下限，请降低采集下限后重新推理。")
    request["scan_start"], request["scan_end"] = grid[0], grid[-1]
    request["scan_step"] = float(request.get("scan_step", .01))
    request["status"] = status
    if request["status"] not in ("complete", "cancelled"):
        raise ValueError("仅支持已完成或已中断任务的指标汇总。")
    if manifest.get("requires_attention"):
        raise ValueError("数据集存在缺失标签，请补齐标注后重新导入；缺失标签不能作为负样本。")
    for key in ("minimum_precision", "minimum_recall", "maximum_fpr"):
        if request.get(key) is not None:
            request[key] = finite_number(request[key], key)
    request["figure_formats"], request["figure_dpi"] = figure_options(request)
    if not request.get("model_path") or not request.get("model_sha256"):
        raise ValueError("评估请求缺少固定模型路径或 SHA256。")
    all_images = [s["image"] for s in manifest["samples"]]
    known_images = set(all_images)
    expected = request.get("sample_images", all_images)
    if not isinstance(expected, list) or not expected or any(not isinstance(x, str) for x in expected):
        raise ValueError("本次评估必须固定非空图片范围。")
    if os.name == "nt":
        expected = [Path(x).as_posix() if x not in known_images and Path(x).as_posix() in known_images else x for x in expected]
    if len(set(expected)) != len(expected) or any(x not in known_images for x in expected):
        raise ValueError("本次评估范围包含重复图片或不属于数据集清单的图片。")
    request["sample_images"] = expected
    request["full_split"] = bool(request.get("full_split", True)) and expected == all_images
    return request, grid


def load_predictions(path, manifest, request):
    selected = {s["image"]: s for s in manifest["samples"]}
    expected = request["sample_images"]
    rows, samples = [], []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"预测缓存包含空行：第 {number} 行。")
            def no_constant(value):
                raise ValueError(f"预测缓存包含非有限数：{value}")
            row = json.loads(line, parse_constant=no_constant)
            if os.name == "nt" and isinstance(row, dict) and isinstance(row.get("image"), str):
                path = Path(row["image"]).as_posix()
                if row["image"] not in selected and path in selected: row["image"] = path
            if not isinstance(row, dict) or len(rows) >= len(expected) or row.get("image") != expected[len(rows)]:
                raise ValueError(f"预测缓存图片顺序与本次固定范围不一致：第 {number} 行。")
            sample = selected[row["image"]]
            if row.get("width") != sample["width"] or row.get("height") != sample["height"]:
                raise ValueError(f"预测坐标系与原图／标签尺寸不一致：{row['image']}")
            predictions = row.get("predictions")
            if not isinstance(predictions, list):
                raise ValueError("预测缓存缺少 predictions 列表。")
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    raise ValueError("预测框格式无效。")
                nonnegative_id(prediction.get("class_id"), "预测类别")
                if request.get("model_class_count") and prediction["class_id"] >= request["model_class_count"]:
                    raise ValueError("预测缓存的类别超出模型实际类别范围。")
                score = finite_number(prediction.get("confidence"), "预测置信度")
                if score + 1e-8 < request["score_floor"]:
                    raise ValueError("预测缓存包含低于声明采集下限的分数。")
                box = prediction.get("box")
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError("预测坐标必须是四元素 xyxy 数组。")
                box = [finite_number(v, "预测框坐标", minimum=-1e-3,
                                     maximum=max(sample["width"], sample["height"]) + 1e-3) for v in box]
                if box[2] <= box[0] or box[3] <= box[1] or box[2] > sample["width"] + 1e-3 or box[3] > sample["height"] + 1e-3:
                    raise ValueError("预测缓存含无效或超出原图的框。")
            rows.append(row)
            samples.append(sample)
    if request["status"] == "complete" and len(rows) != len(expected):
        raise ValueError("任务声明完成，但预测缓存并未覆盖本次固定范围。")
    return samples, rows


def analyze(request, manifest, predictions_path, *, verify_sources=True):
    from evaluation_metrics import average_precision, masked_ap, operating, prepare_records, recommend, summarize_groups
    import ultralytics

    validate_manifest(manifest)
    request, grid = validate_request(request, manifest)
    if verify_sources:
        assert_unchanged(manifest, request["model_path"], request["model_sha256"])
    prediction_sha256 = digest(predictions_path)
    samples, predictions = load_predictions(predictions_path, manifest, request)
    records = prepare_records(samples, predictions, request["model_class_id"], request["dataset_class_id"],
                              scene_reviewed=bool(manifest.get("metadata_path")))
    ap, pr_curve = average_precision(records)
    metrics, per_image = operating(records, request["confidence"], request["match_iou"], details=True)
    complete_split = request["status"] == "complete" and request["full_split"] and len(samples) == len(manifest["samples"])
    publish_ap = complete_split and len(samples) > 1
    public_ap = ap if publish_ap else masked_ap(ap)
    metrics = {**public_ap, **metrics}
    rows = [{**public_ap, **operating(records, confidence, request["match_iou"])} for confidence in grid]
    groups = summarize_groups(records, request["confidence"], request["match_iou"], publish_ap, all_metrics=metrics)
    reviewed_images = sum(bool(row.get("scene_reviewed")) for row in records)
    best = recommend(rows, request.get("minimum_precision"), request.get("minimum_recall"), request.get("maximum_fpr"))
    can_recommend = request["status"] == "complete" and manifest["split"] == "val" and complete_split and len(samples) > 1
    if len(samples) <= 1:
        scope = "single_image" if len(samples) == 1 else "empty_cancelled"
    else:
        scope = "full_split" if complete_split else "subset"
    warnings = [
        "只评估所选目标类别；mAP 字段表示该目标类别的 AP，不是所有模型类别的总体 mAP。",
        "候选为固定 NMS 后的低分预测；扫描不会改变 NMS IoU、max_det、预处理或推理后端。",
        "COCO ID 32 是 sports ball，涵盖多种球类；映射到 football 不会使模型只识别足球。",
        "无目标图片指该图片没有所选数据集类别的 GT，其他类别标注不计为该目标的 GT。",
    ]
    if not publish_ap:
        warnings.append("单图、子集或未完成任务不显示整集 mAP；partial_metrics 中的 AP 仅作描述性分析。")
    if not can_recommend:
        warnings.append("仅完整 Val 集评估提供部署阈值推荐；Test、Train、子集和中断结果只作描述。")
    if manifest.get("skipped"):
        warnings.append("审核排除或待审核图片未纳入指标；完整集合指清单中实际选入的有效样本。")
    if reviewed_images < len(records):
        warnings.append("未审核的 Small／Occluded／Blurred 不作为已知场景；未知样本不能归入 No difficulty。")
    report = {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION,
              "ultralytics_version": ultralytics.__version__, "status": request["status"],
              "full_split": complete_split, "scope": scope,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "confidence": request["confidence"], "request": request, "manifest": manifest,
              "dataset_fingerprint": manifest["fingerprint"], "model_sha256": request["model_sha256"],
              "prediction_sha256": prediction_sha256,
              "predictions_path": str(Path(predictions_path).resolve()),
              "sample_images": [s["image"] for s in samples], "planned_images": len(request["sample_images"]),
              "completed_images": len(samples), "metrics": metrics, "partial_metrics": ap if not publish_ap else None,
              "rows": rows, "per_image": per_image, "groups": groups, "pr_curve": pr_curve,
              "scene_review": {"reviewed_images": reviewed_images, "unreviewed_images": len(records) - reviewed_images,
                               "source": "metadata" if manifest.get("metadata_path") else "unreviewed"},
              "recommendation": best if can_recommend else None, "descriptive_best_f1": best,
              "warnings": warnings,
              "definitions": {"precision": "TP / (TP + FP); no predictions with GT => 0; empty/empty => null",
                              "recall": "TP / (TP + FN); no target GT => null",
                              "F1": "2TP / (2TP + FP + FN); zero denominator => null",
                              "no_ball_false_positive_rate": "negative images with >=1 target prediction / all confirmed target-negative images",
                              "matching": "IoU desc, ties prediction index then GT index; unique prediction (index order), then unique GT",
                              "AP": "Ultralytics ap_per_class, IoU 0.50:0.05:0.95; selected target only"}}
    if verify_sources:
        assert_unchanged(manifest, request["model_path"], request["model_sha256"])
    if digest(predictions_path) != prediction_sha256:
        raise RuntimeError("指标计算期间预测缓存发生变化，拒绝发布报告。")
    return report


def export_report(path, report, *, verify_sources=True):
    report["artifacts"] = save_report_artifacts(Path(path).parent, report, csv_writer=write_csv)
    # Plot generation can take seconds: validate again at the final publication
    # boundary, rather than trusting the earlier calculation-time check.
    if verify_sources:
        assert_unchanged(report["manifest"], report["request"]["model_path"], report["model_sha256"])
    if digest(report["predictions_path"]) != report["prediction_sha256"]:
        raise RuntimeError("报告导出期间预测缓存发生变化，拒绝发布报告。")
    for source, expected in report.get("worker_input_hashes", {}).items():
        if not Path(source).is_file() or digest(source) != expected:
            raise RuntimeError("报告导出期间任务请求或数据集清单发生变化。")
    atomic_json(path, report)  # Final report is published only after every export succeeds.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--request", required=True)
    prepare.add_argument("--output", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--request", required=True)
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--predictions", required=True)
    analyze_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        request_sha = digest(args.request)
        request = read_json(args.request)
        if digest(args.request) != request_sha:
            raise RuntimeError("读取期间评估请求发生变化。")
        if args.command == "prepare":
            manifest = prepare_dataset(request)
            atomic_json(args.output, manifest)
            print(json.dumps({"success": True, "output": str(Path(args.output).resolve()),
                              "selected_images": len(manifest["samples"]),
                              "requires_attention": manifest["requires_attention"]}, ensure_ascii=False))
        else:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(output.parent / ".ultralytics"))
            os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
            manifest_sha = digest(args.manifest)
            report = analyze(request, read_json(args.manifest), args.predictions)
            if digest(args.manifest) != manifest_sha:
                raise RuntimeError("指标计算期间数据集清单发生变化。")
            report["worker_input_hashes"] = {str(Path(args.request).resolve()): request_sha,
                                              str(Path(args.manifest).resolve()): manifest_sha}
            export_report(output, report)
            print(json.dumps({"success": True, "status": report["status"], "output": str(output),
                              "completed_images": report["completed_images"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
