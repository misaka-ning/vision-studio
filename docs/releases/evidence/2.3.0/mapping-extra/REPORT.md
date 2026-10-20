# Vision Studio 模型评估报告

- 状态：complete；范围：full_split；完成 4/4 张。
- 数据集：`/tmp/vision-studio-evaluation-public-230/coco8-qa.yaml`；集合：val。
- 模型原始 ID 32 → 数据集 ID 0。
- 当前置信度：1.0；GT 匹配 IoU：0.5；候选下限：0.001。
- 算法：vision-eval-1/highest-iou-stable-ties；Ultralytics 8.4.173。

| 指标 | 数值 |
| --- | --- |
| mAP50 | 0.00000 |
| mAP50_95 | 0.00000 |
| precision | 0.00000 |
| recall | 0.00000 |
| F1 | 0.00000 |
| TP | 0 |
| FP | 0 |
| FN | 10 |
| no_ball_images | 1 |
| no_ball_false_positive_rate | 0.00000 |

- 只评估所选目标类别；mAP 字段表示该目标类别的 AP，不是所有模型类别的总体 mAP。
- 候选为固定 NMS 后的低分预测；扫描不会改变 NMS IoU、max_det、预处理或推理后端。
- COCO ID 32 是 sports ball，涵盖多种球类；映射到 football 不会使模型只识别足球。
- 无目标图片指该图片没有所选数据集类别的 GT，其他类别标注不计为该目标的 GT。

阈值表：`thresholds.csv`；逐图结果：`per-image.csv`；错例：`errors.csv`。
模型、数据和推理配置及哈希保存在同目录报告 JSON；历史报告使用当时的快照。
