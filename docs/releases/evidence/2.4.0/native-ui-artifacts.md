# 2.4.0 原生 UI 专项证据

此归档保存 UI 开发阶段的原始日志、报告和最终原生窗口截图。正式发行验收以同版本完整测试和最终安装包报告为准；这里的专项测试与正式 18 套测试重复覆盖，**不得把下列通过数量再次累加到正式总数**。

- 归档：[native-ui-artifacts.zip](native-ui-artifacts.zip)，4,707,864 bytes，SHA256 `7cb321845937ec8ffb16af773ecccd096a85fa45dd67e9907d3b7ed95ccb17da`。
- 逐项大小、SHA256、PNG 解码尺寸及实测 Qt 统计：[native-ui-artifacts.manifest.json](native-ui-artifacts.manifest.json)。
- 归档、清单与本文的校验值：[native-ui-artifacts.SHA256SUMS](native-ui-artifacts.SHA256SUMS)。
- ZIP 逐项重新读取后已核对大小与 SHA256，CRC 检查通过；共 132 个文件，其中 25 张 PNG 已使用本机 `/opt/VisionStudio/runtime/bin/python` 的 Pillow 完整解码。

## 最终专项结果

`ui-layout-final/` 中保存最终两套完整专项日志、JUnit、QA JSON、数据集评估报告和全部 25 张 PNG。实际结果：

| 套件 | 通过 | 失败 | 跳过 | 时长（QA runner） |
| --- | ---: | ---: | ---: | ---: |
| vision-evaluation-panel-tests | 24 | 0 | 0 | 0.626 s |
| vision-evaluation-integration-tests | 33 | 0 | 0 | 61.700 s |

这 57 个 Qt 通过条目包含每套的 init/cleanup，不能视作 57 个独立产品功能，也不能与后续正式运行重复计数。完整启动命令、二进制哈希及执行环境见归档内 `ui-layout-final/detection-regression-qa.json` 与对应原始日志。

最终 8 张 `compact-{1260|1380}-{cpu|cuda}-{collapsed|expanded}.png` 均为 Qt 原生完整主窗口。默认工具区实测 96 px；展开全部高级设置为 182 px。1260 × 820 与 1380 × 820 下，折叠画布高 369 px、展开高 283 px；测试逐项检查首行输入按钮、CPU/GPU 设备选择器及第二行控件不越界，画布同时完整包含于父容器与主窗口。PNG 规格逐项记录在清单中。

专项还覆盖：启动空状态、移除普通图片/文件夹/示例入口、Ctrl+O 目录导入、目录/YAML 拖入、拒绝松散图片拖入、三种检测范围、61 张队列、停止后回看、阈值与类别映射、导出、长导入错误摘要和完整详情可访问性。具体覆盖以测试源码和原始断言为准。

## 早期失败与修复过程

| 原始目录 | 实际结果与含义 |
| --- | --- |
| ui-layout-preflight | UI 39 通过、Panel 23 通过、Batch 5 通过；Integration 32 通过、1 失败。失败断言仍要求启动空队列显示空指标卡，与新版空状态契约不符。测试改用真实清空按钮验证空队列隐藏指标，并验证选择摄像头后恢复流式指标；未连接摄像头硬件。 |
| ui-layout-targeted | 长错误定向检查 3 通过；紧凑布局 2 通过、4 失败。4 个窗口/设备组合均发现展开高级参数后画布被父容器裁切。 |
| ui-layout-targeted-geometry | 原始诊断重跑 2 通过、4 失败。工具区高 244 px，画布在父框中从 y=43 起高 280 px，但父框只有 287 px，画布底部超过父框 36 px。原始失败未改写。 |
| ui-layout-targeted-compact-r2 | 将审核文件与扫描参数合并一行并缩减设置区间距后，工具区降为 182 px。长错误定向检查 3 通过、紧凑布局检查 6 通过；随后执行上表最终完整两套检查。 |

定向统计也包含 init/cleanup；2 通过、4 失败不表示已有两种布局通过。早期日志/JUnit/报告保留原始内容，早期截图不重复收录。最终测试保留了画布至少 280 px 和父框包含断言，没有放宽断言掩盖布局问题。

## 证据边界

CPU/CUDA 命名的四个紧凑布局条目仅切换设备选择器，**不执行 GPU 推理，也不证明本机显卡或 CUDA 环境可用**。真实 CPU/GPU 模型及安装包验收见正式发行证据，不能由布局截图代替。

集成测试使用程序生成的合成图片、标签和真实执行的 Conv → GlobalAveragePool → Reshape ONNX 测试网络，验证匹配、任务范围、缓存、停止、导出等确定行为。蓝色图片中的 GT 框是真实测试标注，截图中的测试指标不代表公开 YOLO 权重的实际准确率，也不应作为性能宣传数据。

归档未复制隔离设置、缓存、模型权重、runtime、用户数据集或其他个人文件。部分原始日志包含本仓库路径和临时测试目录路径，用于追溯执行，不含用户提供的数据集内容。
