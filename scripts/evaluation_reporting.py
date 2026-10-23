# SPDX-License-Identifier: AGPL-3.0-only
"""Paper figures and tables from an evaluation snapshot, without inference."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

FIGURES = ("pr-curve", "threshold-performance", "negative-false-positive-rate", "ap-by-iou", "froc",
           "detection-confusion-matrix", "image-presence-confusion-matrix", "scene-comparison", "tp-fp-fn")
GROUP_NAMES = ("All", "Small", "Occluded", "Blurred", "No Ball", "No difficulty")
METRIC_NAMES = {
    "mAP50": "目标类 AP50", "mAP75": "目标类 AP75", "mAP50_95": "目标类 AP50–95",
    "precision": "Precision", "recall": "Recall", "F1": "F1", "TP": "TP", "FP": "FP", "FN": "FN",
    "images": "已完成图片", "gt_boxes": "目标 GT 框", "predicted_boxes": "工作阈值预测框",
    "positive_images": "目标阳性图片", "no_ball_images": "目标阴性图片",
    "no_ball_false_positive_images": "出现目标误报的阴性图片",
    "no_ball_false_positive_rate": "阴性图片误报率", "no_ball_fpr_wilson95": "误报率 Wilson 95% 描述区间",
    "no_ball_false_positive_boxes": "阴性图片误报框", "no_ball_fp_boxes_per_image": "每阴性图误报框数",
    "fp_boxes_per_image": "FPPI（全部图片平均误报框）",
    "positive_images_with_no_matched_ball": "没有正确定位目标的阳性图片",
    "positive_image_miss_rate": "阳性图片漏检率",
    "ultralytics_max_f1_precision": "官方 AP 扫描最大 F1 点 Precision（非当前工作点）",
    "ultralytics_max_f1_recall": "官方 AP 扫描最大 F1 点 Recall（非当前工作点）",
}


def figure_options(request):
    formats = request.get("figure_formats", ["png", "svg", "pdf"])
    if isinstance(formats, str):
        formats = formats.split(",")
    if (not isinstance(formats, (list, tuple)) or not formats or
            any(not isinstance(fmt, str) or fmt not in ("png", "svg", "pdf") for fmt in formats)):
        raise ValueError("图表格式须为 PNG／SVG／PDF 的非空列表。")
    dpi = request.get("figure_dpi", 300)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 1200:
        raise ValueError("图表 DPI 须为 72 至 1200 的整数。")
    return list(dict.fromkeys(formats)), dpi


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["image", "TP", "FP", "FN", "negative"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def display(value):
    if value is None:
        return "N/A"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display(v) for v in value) + "]"
    return f"{value:.5f}" if isinstance(value, float) else str(value)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def public_ap(report):
    metric = report.get("metrics", {})
    return report.get("status") == "complete" and bool(report.get("full_split")) and metric.get("images", 0) > 1


def csv_tables(output, report, csv_writer):
    metric, rows, images = report["metrics"], report["rows"], report["per_image"]
    # Keep the original filenames and write callback for existing integrations.
    csv_writer(output / "thresholds.csv", rows)
    csv_writer(output / "per-image.csv", images)
    csv_writer(output / "errors.csv", [row for row in images if row["FP"] or row["FN"]])
    names = ["thresholds.csv", "per-image.csv", "errors.csv"]
    write_csv(output / "false-positive-no-ball.csv", [row for row in images if row["negative"] and row["FP"]],
              list(images[0]) if images else None)
    groups = [{"scene": name, **{key: value for key, value in row.items() if key != "sample_images"}}
              for name, row in report.get("groups", {}).items()]
    write_csv(output / "scene-metrics.csv", groups)
    pr = report.get("pr_curve") or {}
    write_csv(output / "pr-curve.csv", [{"recall": recall, "precision_envelope": precision,
                                       "ap_published": public_ap(report)}
              for recall, precision in zip(pr.get("recall", []), pr.get("precision_envelope", []))],
              ["recall", "precision_envelope", "ap_published"])
    write_csv(output / "ap-by-iou.csv", [{"iou": round(.5 + i * .05, 2), "AP": value,
                                         "ap_published": public_ap(report)}
              for i, value in enumerate(metric.get("ap_per_iou", [None] * 10))])
    names += ["false-positive-no-ball.csv", "scene-metrics.csv", "pr-curve.csv", "ap-by-iou.csv"]
    request = report.get("request", {})
    config = request.get("config", report.get("config", {}))
    model = Path(request.get("model_path", "model")).name
    split = report.get("manifest", {}).get("split", request.get("split", "unknown"))
    columns = [tex_escape(model), tex_escape(split), tex_escape(report.get("scope", "unknown")), str(metric.get("images", 0))]
    keys = ("mAP50", "mAP50_95", "precision", "recall", "F1", "no_ball_false_positive_rate")
    columns += [display(metric.get(key)) for key in keys]
    conditions = (f"% confidence={request.get('confidence', report.get('confidence'))}; match IoU={request.get('match_iou')}; "
                  f"candidate floor={request.get('score_floor')}; NMS IoU={config.get('nms_iou', 'N/A')}; "
                  f"max detections={config.get('max_detections', config.get('max_det', 'N/A'))}\n")
    (output / "paper-row.tex").write_text(
        "% Model & Split & Scope & Images & AP50 & AP50--95 & P & R & F1 & Negative-image FPR\n" + conditions +
        " & ".join(columns) + " \\\\\n", encoding="utf-8")
    return names + ["paper-row.tex"]


def plot_figures(output, report, formats, dpi):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows, metric = report["rows"], report["metrics"]
    x = [row["confidence"] for row in rows]
    names = []
    def values(key):
        return [float("nan") if row.get(key) is None else row[key] for row in rows]
    def empty(ax, message):
        ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
    def new(size=(6.8, 4.6)):
        return plt.subplots(figsize=size, layout="constrained")
    def save(fig, name):
        try:
            for fmt in formats:
                path = f"{name}.{fmt}"
                fig.savefig(output / path, dpi=dpi, bbox_inches="tight")
                names.append(path)
        finally:
            plt.close(fig)
    def probability(ax):
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25, linewidth=.6)
    palette = ("#0072B2", "#D55E00", "#009E73")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "legend.fontsize": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = new()
        pr = report.get("pr_curve")
        if pr:
            label = f"Target AP50 = {metric['mAP50']:.4f}" if public_ap(report) and metric.get("mAP50") is not None else "Descriptive snapshot; AP not published"
            ax.plot(pr["recall"], pr["precision_envelope"], color=palette[0], label=label)
            ax.legend(loc="lower left")
        else:
            empty(ax, "Not defined: no target ground truth")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1))
        probability(ax)
        save(fig, "pr-curve")

        for name, keys, ylabel in (
                ("threshold-performance", ("precision", "recall", "F1"), "Target detection score"),
                ("negative-false-positive-rate", ("no_ball_false_positive_rate",), "Target-negative image false-positive rate"),
                ("tp-fp-fn", ("TP", "FP", "FN"), "Object count")):
            fig, ax = new()
            for key, color in zip(keys, palette):
                ax.plot(x, values(key), color=color, linewidth=1.6, label=key)
            ax.axvline(report["confidence"], color=".4", alpha=.6, linestyle="--", linewidth=1,
                       label=f"Operating threshold {report['confidence']:g}")
            ax.set(xlabel="Confidence threshold (fixed NMS candidates)", ylabel=ylabel, xlim=(0, 1))
            if name == "tp-fp-fn":
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.25)
            else:
                probability(ax)
            if name == "negative-false-positive-rate" and not metric.get("no_ball_images"):
                empty(ax, "Not defined: no target-negative images")
            ax.legend(loc="best")
            save(fig, name)

        fig, ax = new()
        ap_values = metric.get("ap_per_iou", [None] * 10)
        if any(value is not None for value in ap_values):
            ax.plot(np.linspace(.5, .95, 10), [np.nan if v is None else v for v in ap_values], marker="o", color=palette[0])
        else:
            empty(ax, "AP not published for this scope" if not public_ap(report) else "Not defined: no target ground truth")
        ax.set(xlabel="Matching IoU", ylabel="Target average precision", xlim=(.49, .96))
        probability(ax)
        save(fig, "ap-by-iou")

        fig, ax = new()
        ax.plot(values("fp_boxes_per_image"), values("recall"), color=palette[2], linewidth=1.6)
        if not metric.get("gt_boxes"):
            empty(ax, "Not defined: no target ground truth")
        ax.set(xlabel="False-positive boxes per image (FPPI)", ylabel="Object recall", xlim=(0, None))
        probability(ax)
        save(fig, "froc")

        presence = metric.get("presence_confusion")
        matrices = (("detection-confusion-matrix", [[metric["TP"], metric["FN"]], [metric["FP"], np.nan]],
                     ("Target", "No detection"), ("Target", "Background")),
                    ("image-presence-confusion-matrix", [[presence["TP"], presence["FN"]], [presence["FP"], presence["TN"]]] if presence else None,
                     ("Target present", "No target"), ("Target present", "No target")))
        for name, matrix, xticks, yticks in matrices:
            fig, ax = new((7.2, 5.4))
            if matrix is None:
                empty(ax, "Not recorded in this historical report")
            else:
                array = np.asarray(matrix, dtype=float)
                maximum = max(1., np.nanmax(array))
                ax.imshow(np.ma.masked_invalid(array), cmap="Blues", vmin=0, vmax=maximum)
                for i in range(2):
                    for j in range(2):
                        value = array[i, j]
                        ax.text(j, i, "N/A" if np.isnan(value) else str(int(value)), ha="center", va="center",
                                color="white" if np.isfinite(value) and value > maximum / 2 else "black", fontsize=14)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=xticks, yticklabels=yticks,
                   xlabel="Predicted", ylabel="Ground truth")
            if name == "image-presence-confusion-matrix":
                ax.set_title("Image presence (does not measure localization)", fontsize=11, pad=12)
            else:
                ax.set_title("Detection counts (background TN is undefined)", fontsize=11, pad=12)
            save(fig, name)

        fig, ax = new((10.8, 5.0))
        groups = report.get("groups", {})
        positions = np.arange(len(GROUP_NAMES))
        for offset, key, label, color in ((-.26, "mAP50", "Target AP50", palette[0]),
                                         (0., "recall", "Recall", palette[1]),
                                         (.26, "no_ball_false_positive_rate", "Negative-image FPR", palette[2])):
            vals = [groups.get(name, {}).get(key) for name in GROUP_NAMES]
            bars = ax.bar(positions + offset, [np.nan if value is None else value for value in vals], width=.24, label=label, color=color)
            for bar, value in zip(bars, vals):
                ax.annotate("N/A" if value is None else f"{value:.2%}",
                            (bar.get_x() + bar.get_width() / 2, .02 if value is None else value),
                            textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=7,
                            rotation=90 if value is None else 0)
        labels = [name + ("\n(unreviewed)" if not groups.get(name, {}).get("available", True) else "") for name in GROUP_NAMES]
        ax.set(xticks=positions, xticklabels=labels, ylabel="Metric value", ylim=(0, 1.18))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
        save(fig, "scene-comparison")
    return names


def markdown_report(report, formats, dpi):
    metric, request, manifest = report["metrics"], report.get("request", {}), report.get("manifest", {})
    config = request.get("config", report.get("config", {}))
    text = ["# Vision Studio 模型评估报告", "",
            f"- 状态：{report['status']}；范围：{report['scope']}；完成 {report['completed_images']}/{report['planned_images']} 张。",
            f"- 数据集：`{manifest.get('data_yaml', manifest.get('dataset_yaml', ''))}`；集合：{manifest.get('split', 'unknown')}。",
            f"- 模型：`{request.get('model_path', '')}`；原始 ID {request.get('model_class_id')} → 数据集 ID {request.get('dataset_class_id')}。",
            f"- 当前置信度：{report['confidence']}；GT 匹配 IoU：{request.get('match_iou')}；候选下限：{request.get('score_floor')}。",
            f"- 推理配置：输入尺寸 {config.get('input_size', config.get('input_width', 'N/A'))}；NMS IoU {config.get('nms_iou', 'N/A')}；完整配置见报告 JSON。",
            f"- 算法：{report.get('algorithm_version', 'unknown')}；Ultralytics {report.get('ultralytics_version', 'unknown')}。",
            f"- 图表：{', '.join(formats)}；PNG {dpi} DPI，SVG/PDF 保留矢量内容。", "",
            "| 指标 | 数值 |", "| --- | ---: |"]
    text += [f"| {title} | {display(metric.get(key))} |" for key, title in METRIC_NAMES.items()]
    text += ["", "AP 来自低置信度候选，随分析工作阈值保持不变；仅映射的目标类 AP，不是多类总体 mAP。",
             "单图、子集或中断任务的主 AP 为 N/A；partial_metrics 中的 AP 仅作描述，不代表完整集合成绩。",
             "PR 表和图保留候选精度包络；未完成范围明确标记为描述性快照。",
             "Wilson 区间仅为图像独立假设下的描述区间；邻帧、左右目或同场景相关性可能使区间过窄。", "",
             "## 场景分组", "", "| 场景 | 审核状态 | 图片 | AP50 | AP50–95 | Precision | Recall | F1 | 阴性图误报率 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in report.get("groups", {}).items():
        values = [name, group.get("review_status", "历史报告未记录")] + [display(group.get(key)) for key in
                  ("images", "mAP50", "mAP50_95", "precision", "recall", "F1", "no_ball_false_positive_rate")]
        text.append("| " + " | ".join(values) + " |")
    review = report.get("scene_review", {})
    text += ["", f"场景已审核 {review.get('reviewed_images', 'N/A')} 张；未审核 {review.get('unreviewed_images', 'N/A')} 张。",
             "Small／Occluded／Blurred 依赖审核 metadata；未审核场景显示 N/A，不能把未知当作 No difficulty。",
             "No Ball 按所选目标类 GT 是否为空确定；No difficulty 仅包括已审核且四项场景标志均为 0 的图片（不含 No Ball）。",
             "分组可重叠，不可相加或平均子组 AP 代替总体。仅完整任务且分组至少两张图片发布分组 AP；更小分组只保留明确标记的描述性 AP。", "",
             "## 混淆矩阵与错误", "",
             "检测矩阵使用目标级 TP／FP／FN，背景 TN 没有定义，图中为 N/A。",
             "图像存在性矩阵仅判断该图是否有目标预测框；预测存在不表示定位正确。",
             "| 图像存在性 | TP | FP | FN | TN |", "| --- | ---: | ---: | ---: | ---: |"]
    presence = metric.get("presence_confusion", {})
    text.append("| 图片数 | " + " | ".join(display(presence.get(key)) for key in ("TP", "FP", "FN", "TN")) + " |")
    text += ["", "阈值明细：`thresholds.csv`；逐图结果：`per-image.csv`；全部错例：`errors.csv`；阴性误报专表：`false-positive-no-ball.csv`。",
             "场景表：`scene-metrics.csv`；PR 坐标：`pr-curve.csv`；AP–IoU 表：`ap-by-iou.csv`；论文 LaTeX 行：`paper-row.tex`。", "",
             "## 图表", ""]
    text += [f"- `{name}`：{', '.join(formats)}。" for name in FIGURES]
    text += ["", *["- " + warning for warning in report.get("warnings", [])], "",
             "模型、数据、预测、来源校验及推理配置保存在同目录报告 JSON；历史结果使用当时的快照。",
             "TP／FP／FN、P／R／F1、FPR、FPPI 与漏检率来自先过滤阈值再重新进行一对一匹配。",
             "扫描固定 NMS 后候选，不改变 NMS、最大候选数、模型头或预处理；不能据此推断不同后端结果相等。",
             "Test 扫描仅作描述，请在完整 Val 上选择部署阈值，再固定到 Test 验证。", ""]
    if report.get("runtime"):
        text += ["运行时间为本次执行观察，不是独立硬件 benchmark：", "", "```json",
                 json.dumps(report["runtime"], ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    if report.get("source_runtime"):
        text += ["外部来源保存的运行观察（不是本次重分析执行推理，也不证明与应用后端等价）：", "", "```json",
                 json.dumps(report["source_runtime"], ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    if report.get("allow_apply_profile") is False or request.get("allow_apply_profile") is False:
        text += ["来源未完整记录应用所需的推理和预处理配置；本报告可用于数值分析，不能采用检测配置。", ""]
    return "\n".join(text)


def save_report_artifacts(output, report, csv_writer=write_csv):
    """Write artifacts only; the caller owns input verification and publication."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    formats, dpi = figure_options(report.get("request", {}))
    names = csv_tables(output, report, csv_writer)
    names += plot_figures(output, report, formats, dpi)
    (output / "REPORT.md").write_text(markdown_report(report, formats, dpi), encoding="utf-8")
    report["export_options"] = {"formats": formats, "dpi": dpi}
    names.append("REPORT.md")
    return {name: name for name in names}
