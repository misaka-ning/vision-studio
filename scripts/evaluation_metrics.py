# SPDX-License-Identifier: AGPL-3.0-only
"""Single-target metrics shared by every Vision Studio inference backend.

Adapted from the maintainer's model-evaluation/eval_metrics.py. Matching follows
Ultralytics' default highest-IoU one-to-one rule (not Hungarian matching). AP is
computed by the installed, pinned Ultralytics implementation; operating-point
P/R are explicitly separate from its maximum-F1 summary. Public class IDs and
prediction indices always remain the original model IDs and cache indices.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import numpy as np

IOUS = np.linspace(.5, .95, 10)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0., center - half), min(1., center + half)]


def threshold_grid(start=.05, end=.95, step=.01):
    try:
        a, b, delta = (Decimal(str(v)) for v in (start, end, step))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("阈值扫描范围必须是有限数字。") from exc
    if not all(x.is_finite() for x in (a, b, delta)) or not 0 <= a <= b <= 1 or delta <= 0:
        raise ValueError("阈值扫描须满足 0 ≤ 起点 ≤ 终点 ≤ 1，且步长大于 0。")
    size = int((b - a) // delta) + 1
    if size > 2001:
        raise ValueError("单次扫描最多支持 2001 个工作点，请增加步长。")
    values = [a + i * delta for i in range(size)]
    if values[-1] != b:
        values.append(b)
    return [float(x) for x in values]


def box_iou(gt, predictions):
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(gt[:, None, 2:], predictions[None, :, 2:]) -
                              np.maximum(gt[:, None, :2], predictions[None, :, :2])).prod(2)
    return intersection / (np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(1)[:, None] +
                           np.maximum(0, predictions[:, 2:] - predictions[:, :2]).prod(1)[None, :] -
                           intersection + 1e-7)


def matching_pairs(iou, threshold):
    pairs = np.array(np.nonzero(iou >= threshold)).T
    if len(pairs) > 1:
        # Stable tie policy shared with the C++ review overlay: highest IoU,
        # then original prediction order, then GT order. Non-tied matching is
        # identical to Ultralytics; its default argsort tie order is unspecified.
        order = np.lexsort((pairs[:, 0], pairs[:, 1], -iou[pairs[:, 0], pairs[:, 1]]))
        pairs = pairs[order]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs


def match(iou, thresholds=IOUS):
    correct = np.zeros((iou.shape[1], len(thresholds)), dtype=bool)
    for index, threshold in enumerate(thresholds):
        pairs = matching_pairs(iou, threshold)
        if len(pairs):
            correct[pairs[:, 1], index] = True
    return correct


def prepare_records(samples, predictions, model_class_id, dataset_class_id):
    result = []
    for sample, row in zip(samples, predictions):
        selected_gt = [(i, gt) for i, gt in enumerate(sample["gt"]) if gt["class_id"] == dataset_class_id]
        selected_pred = [(i, p) for i, p in enumerate(row["predictions"]) if p["class_id"] == model_class_id]
        gt = [p["box"] for _, p in selected_gt]
        boxes = [p["box"] for _, p in selected_pred]
        iou = box_iou(gt, boxes)
        result.append({"image": sample["image"], "gt": gt, "boxes": boxes,
                       "gt_indices": [i for i, _ in selected_gt],
                       "prediction_indices": [i for i, _ in selected_pred],
                       "scores": np.asarray([p["confidence"] for _, p in selected_pred], dtype=float),
                       "flags": sample.get("flags", {}), "iou": iou, "correct": match(iou)})
    return result


def operating(records, confidence, iou_threshold, details=False):
    tp = fp = fn = negative_images = negative_fp_images = negative_fp_boxes = positive_images = positive_missed = 0
    per_image = []
    for row in records:
        kept = np.flatnonzero(row["scores"] >= confidence)
        pairs = matching_pairs(row["iou"][:, kept], iou_threshold)
        matched_pred = set(int(x) for x in pairs[:, 1]) if len(pairs) else set()
        matched_gt = set(int(x) for x in pairs[:, 0]) if len(pairs) else set()
        hits, count, targets = len(pairs), len(kept), len(row["gt"])
        tp += hits
        fp += count - hits
        fn += targets - hits
        if targets:
            positive_images += 1
            positive_missed += hits == 0
        else:
            negative_images += 1
            negative_fp_images += count > 0
            negative_fp_boxes += count
        if details:
            kept_original = [row["prediction_indices"][int(i)] for i in kept]
            matched_original = [kept_original[i] for i in sorted(matched_pred)]
            fp_original = [value for i, value in enumerate(kept_original) if i not in matched_pred]
            per_image.append({"image": row["image"], "TP": hits, "FP": count - hits, "FN": targets - hits,
                              "negative": not bool(targets), "kept_prediction_indices": kept_original,
                              "matched_indices": matched_original, "fp_indices": fp_original,
                              "matched_gt_indices": [row["gt_indices"][i] for i in sorted(matched_gt)],
                              "missed_gt_indices": [index for i, index in enumerate(row["gt_indices"]) if i not in matched_gt]})
    precision = ratio(tp, tp + fp) if tp + fp else (0. if tp + fn else None)
    result = {"confidence": float(confidence), "match_iou": float(iou_threshold), "TP": tp, "FP": fp, "FN": fn,
              "precision": precision, "recall": ratio(tp, tp + fn), "F1": ratio(2 * tp, 2 * tp + fp + fn),
              "images": len(records), "gt_boxes": tp + fn, "predicted_boxes": tp + fp,
              "no_ball_images": negative_images, "no_ball_false_positive_images": negative_fp_images,
              "no_ball_false_positive_rate": ratio(negative_fp_images, negative_images),
              "no_ball_fpr_wilson95": wilson(negative_fp_images, negative_images),
              "no_ball_false_positive_boxes": negative_fp_boxes,
              "no_ball_fp_boxes_per_image": ratio(negative_fp_boxes, negative_images),
              "fp_boxes_per_image": ratio(fp, len(records)), "positive_images": positive_images,
              "positive_images_with_no_matched_ball": positive_missed,
              "positive_image_miss_rate": ratio(positive_missed, positive_images)}
    return (result, per_image) if details else result


def average_precision(records):
    targets = sum(len(r["gt"]) for r in records)
    if not targets:
        return {"mAP50": None, "mAP75": None, "mAP50_95": None, "ap_per_iou": [None] * 10}, None
    from ultralytics.utils.metrics import ap_per_class
    correct = np.concatenate([r["correct"] for r in records], axis=0)
    scores = np.concatenate([r["scores"] for r in records])
    values = ap_per_class(correct, scores, np.zeros(len(scores)), np.zeros(targets),
                          plot=False, names={0: "selected target"})
    ap = values[5][0]
    result = {"mAP50": float(ap[0]), "mAP75": float(ap[5]), "mAP50_95": float(ap.mean()),
              "ap_per_iou": ap.tolist(), "ultralytics_max_f1_precision": float(values[2][0]),
              "ultralytics_max_f1_recall": float(values[3][0])}
    return result, {"recall": values[10].tolist(), "precision_envelope": values[11][0].tolist()}


def recommend(rows, minimum_precision=None, minimum_recall=None, maximum_fpr=None):
    candidates = [r for r in rows if r["F1"] is not None]
    if minimum_precision is not None:
        candidates = [r for r in candidates if r["precision"] is not None and r["precision"] >= minimum_precision]
    if minimum_recall is not None:
        candidates = [r for r in candidates if r["recall"] is not None and r["recall"] >= minimum_recall]
    if maximum_fpr is not None:
        candidates = [r for r in candidates if r["no_ball_false_positive_rate"] is not None and
                      r["no_ball_false_positive_rate"] <= maximum_fpr]
    if not candidates:
        return None
    if minimum_precision is not None or maximum_fpr is not None:
        return max(candidates, key=lambda r: (r["recall"] or 0., r["precision"] or 0., r["F1"], r["confidence"]))
    if minimum_recall is not None:
        return max(candidates, key=lambda r: (r["precision"] or 0., r["recall"] or 0., r["F1"], r["confidence"]))
    return max(candidates, key=lambda r: (r["F1"], r["precision"] or 0., r["recall"] or 0., r["confidence"]))
