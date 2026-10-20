# Vision Studio 2.3.0 验收证据

本目录只记录实际执行结果。源码输入冻结于 `9dba95c0081cf1d1616c6537f6e41ad58d07722a`，最终安装包与完整对应源码的大小、SHA256 见 [final-assets.json](final-assets.json)。本版发布状态与远端核验另以 GitHub Release／发布回执为准。

| 检查 | 真实结果 | 证据 |
| --- | --- | --- |
| 有效软件回归 | 18 套，Qt 286 项、Python 86 项；失败／跳过为 0 | [ctest-qa.json](ctest-qa.json) |
| 工作台集成 | 23 项，原生数据集选择、61 张三范围、停止前缀、阈值／错例／采用配置、历史与导出 | [集成日志](final-ui/integration.log)、[实际退出码和 ELF](final-ui/run-qa.json) |
| 原界面及小窗口 | 38 项；1260×820 画布被父容器完整包含，深色页签可读 | [UI 日志](final-ui/ui-tests.log)、[视觉检查](final-ui/visual-qa.json) |
| 评估后台与数据指标 | Worker 18 项，Python 34 项；手算与官方 AP 实现对照，缓存篡改拒绝 | [evaluation-qa.json](evaluation-qa.json) |
| 最终 DEB | 结构、依赖模拟、只读 CPU／Netron、ONNX／TorchScript 转换通过，无警告 | [qa-report.json](qa-report.json) |
| 实际离线安装 | 私有命名空间安装、启动、推理、卸载通过，宿主只读 | [install-qa.json](install-qa.json) |
| 最终包 CPU／CUDA | 实际 ONNX／PT 四条路径通过，另有 C1／C3、左右目视频与录制回归 | [gpu-qa.json](gpu-qa.json) |
| 最终包数据集评估 | 6 个预训练模型场景，3 组同权重对照，每项 4 张、91 阈值、三格式导出；实际 910 个 CUDA 节点事件 | [packaged-evaluation-qa.json](packaged-evaluation-qa.json) |
| 对应源码 | 4019 份应用文件逐字节核验、18 份上游归档、59 份 runtime METADATA、最终 ELF 与安装源码映射 | [source-bundle-qa.json](source-bundle-qa.json) |
| 图像解码 | 最终包 30 张报告曲线、原生标注和界面截图可完整解码 | [packaged-image-decode-qa.json](packaged-image-decode-qa.json) |

预训练评估使用官方 COCO8 的四张 Val 真标注，只用于验证数据／模型／后端链路，不能作为足球模型能力测试。ONNX 与 PT 各自 CPU／CUDA 指标在本次样本上相同；PT 与导出的 TorchScript 的 AP50–95 有约 0.00667 差异，按公开容差通过，不声称不同后端逐比特一致。首版 AP 仅针对映射的一个目标类别；COCO ID32 是 sports ball，不等于足球专用类别。合成 Conv／GAP／Reshape 模型用于独立几何与状态断言，明确不作为真实模型准确率宣传。

统计、输入哈希、类别映射、工作点、预处理、设备、候选上限和固定 NMS 口径均随报告保存。完整与子集／中断任务分开标识；只有完整 Val 可推荐阈值。历史回看校验原图及样本缓存，原图改变时不叠加旧预测。未测试 Windows 2.3.0、物理摄像头或新显卡驱动安装；本版未新增运行依赖。

[packaged-raw-manifest.json](packaged-raw-manifest.json) 绑定 68 份包内原始日志；[QA-ARTIFACTS.zip](QA-ARTIFACTS.zip) 包含实际曲线、CSV／Markdown 与选中结果数据，原始数据照片、模型生成物留在 Git 之外。原始回归失败保留在 regression-01／integration-01 等目录，通过项只采用 ctest-qa 中明确列出的有效复验，不重复累加。最初的画布布局问题、文件选择自动化时序问题和 sandbox loopback 限制均保留原因及复验映射。

QA 工具与最终验收资料在源码冻结后补入 Git，不改变已构建的运行代码、DEB 或完整对应源码。发行和清理回执会在远端核验之后另作维护提交，已发布标签和附件保持不变。
