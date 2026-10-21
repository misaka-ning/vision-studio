# Vision Studio 2.5.0 验收报告

本报告记录 2026-10-21 实际完成的 Linux amd64 2.5.0 验收。本版统一目录／YAML 导入选择器，增加可见缩略图与原图的后台加载，将「图像预览」「评估与阈值」分为两个完整工作区，并把数据集设置移至悬浮卡片。C++17／Qt 6.8.3、Fluent 深色界面、既有推理后端与个人资料约定保留。

**验收、正式发布和保留两版清理已完成，见[独立发布凭证](publication.md)。** 当前隔离安装已实际通过，宿主仍为 2.4.0，本次升级等待系统 Polkit 管理员验证，不将隔离安装等同宿主安装。发行说明见 [2.5.0](../../2.5.0.md)。

## 有效回归与初轮失败

[ctest-qa.json](ctest-qa.json)记录 **18 套有效结果、305 项 Qt、101 项 Python，通过且无剩余失败、跳过或 blacklist**。这些数字来自原完整执行中通过的 17 套，加上修正异步回看等待后的 scope 单套 CPU 复验，各套只计一次；**不是一次完整 18 套全部通过**。

初轮完整执行含真实 CUDA；scope 套的 **14 项通过、24 项失败**全部保留在[原始日志](qa-full/vision-scope-filter-ui-tests.log)、[JUnit](qa-full/vision-scope-filter-ui-tests.xml)和[原 runner](qa-full/detection-regression-qa.json)，不改写为通过。旧测试在选图后同步读取画布，而本版回看已经异步。修正仅在 `tests/scope_filter_ui_tests.cpp` 插入 **35 行**等待与额外真实像素断言，保留每一条旧源码行及全部原断言，不删除、替换、跳过案例或修改模型 fixture。[源码原件](async-review-repair/original-scope_filter_ui_tests.cpp.gz)、[完整差异](async-review-repair/scope_filter_ui_tests.diff)与 SHA256 随报告保存。正式 CPU 复验的[原始日志](qa-scope-async-rerun/vision-scope-filter-ui-tests.log)与[JUnit](qa-scope-async-rerun/vision-scope-filter-ui-tests.xml)中，相同顺序的全部 **38 项 PASS**；该次未追加 GPU 套件。

[严格聚合工具](summarize_async_review_repair.py)核验[前快照](production-inputs-before.json)、[修正后快照](production-inputs-after-async.json)、完整执行结束与复验的时间顺序、准确失败断言位置及完整案例列表。前后仅上述测试源码与 `vision-scope-filter-ui-tests` 二进制改变；生产应用、其他原生测试、其余已快照源码、模型资源、CMake 配置、运行时元数据和依赖锁的 SHA256 相同。外部 runtime 整棵目录未在该快照中独立逐文件重算，不扩大此证明范围。

| 套件 | 实际通过 | 原始证据 |
| --- | ---: | --- |
| vision-core-tests | 39 | [原始日志](qa-full/vision-core-tests.log) |
| vision-inference-delivery-tests | 6 | [原始日志](qa-full/vision-inference-delivery-tests.log) |
| vision-recording-tests | 15 | [原始日志](qa-full/vision-recording-tests.log) |
| vision-pt-tests | 36 | [原始日志](qa-full/vision-pt-tests.log) |
| vision-gpu-tests | 11 | [原始日志](qa-full/vision-gpu-tests.log) |
| vision-ui-tests | 39 | [原始日志](qa-full/vision-ui-tests.log) |
| vision-batch-responsiveness-tests | 6 | [原始日志](qa-full/vision-batch-responsiveness-tests.log) |
| vision-scope-filter-ui-tests | 38 | [原始日志](qa-scope-async-rerun/vision-scope-filter-ui-tests.log) |
| vision-modelviewer-tests | 16 | [原始日志](qa-full/vision-modelviewer-tests.log) |
| vision-conversion-ui-tests | 8 | [原始日志](qa-full/vision-conversion-ui-tests.log) |
| vision-conversion-inference-tests | 14 | [原始日志](qa-full/vision-conversion-inference-tests.log) |
| vision-model-conversion-tests | 20 | [原始日志](qa-full/vision-model-conversion-tests.log) |
| vision-netron-server-tests | 11 | [原始日志](qa-full/vision-netron-server-tests.log) |
| vision-gpu-runtime-tests | 21 | [原始日志](qa-full/vision-gpu-runtime-tests.log) |
| vision-evaluation-worker-tests | 18 | [原始日志](qa-full/vision-evaluation-worker-tests.log) |
| vision-evaluation-panel-tests | 25 | [原始日志](qa-full/vision-evaluation-panel-tests.log) |
| vision-evaluation-python-tests | 49 | [原始日志](qa-full/vision-evaluation-python-tests.log) |
| vision-evaluation-integration-tests | 34 | [原始日志](qa-full/vision-evaluation-integration-tests.log) |

## 本版 UI 与大队列专项

[ui-queue-qa.json](ui-queue-qa.json)的七项检查均为 true：目录／YAML 共用选择器、非法输入行内错误、Popup 关闭与忙时锁定／几何、长错误完整可读、完整工作区切页保持配置、异步队列的心跳／像素／过期取消，以及四组布局。它解析实际生产原生测试日志；绑定最终 DEB 的哈希用于追溯，**不是另一次已安装包的专项 UI 执行**。

- [Panel 原始日志](qa-full/vision-evaluation-panel-tests.log)：一个选择器实际接受目录和 YAML／YML；单击目录确认与双击浏览、取消、非法类型行内拒绝均有真实检查。Popup 再次点击、外部点击和 Esc 收起，忙时锁定，长错误可滚动、复制，配置不变。
- [集成原始日志](qa-full/vision-evaluation-integration-tests.log)：1260／1380 × 820，CPU／CUDA 选择器四组布局均通过。任务条为 96px；1260 宽时画布为 422 × 369，1380 宽时为 542 × 369；打开 620 × 309 的 Popup 前后工具栏和画布几何不变。另一个全宽工作区案例记录 tab 宽 1120、分析区宽 1118，切页模式与配置不变、4 项缓存保留、范围索引联动为 2。选择器的 CUDA 布局本身不证明 GPU 推理。
- [队列原始日志](qa-full/vision-batch-responsiveness-tests.log)：标准 YOLO 格式的 **2000 张合成小图**，实际观察到分片填充 96 项、队列填充时编辑锁定；在 96 项处停止，停止调用耗时 3ms。初始只加载 6 张缩略图，心跳 136 次、最大间隔 66ms；快速选图最终索引 1998，源路径／真实像素与所选图片一致，清空后剩余任务 0，并检查灰度及过期回调拒绝。以上是本次环境的观察，不能推广为大尺寸图像、磁盘或其他机器的固定速度。

三种检测范围、超过 50 张、停止后已完成结果回看与当前图导出、类别筛选、灰度 C1／C3、左右目／录像、全局模型、模型显示、转换与持久报告继续由适用套件覆盖。合成 fixture 用于调度与 UI 验证，不提供模型准确率结论。

## 最终 DEB、真实模型与数据

- [qa-report.json](qa-report.json)：最终 DEB 权限、依赖、ELF、许可与只读 CPU／PT／结构／转换 smoke，通过 APT 依赖模拟。
- [install-qa.json](install-qa.json)：只读宿主、隔离网络的 bubblewrap 命名空间内，实际安装 `2.5.0-1`、普通用户运行、卸载与资料保留通过；宿主 dpkg 和桌面集成保持原样。这与宿主安装是不同步骤。
- [packaged-gpu-qa.json](packaged-gpu-qa.json)：最终包 ONNX CPU／CUDA、PT CPU／CUDA 四条实际路径通过。[gpu-qa.json](gpu-qa.json)的七项门禁全为 true，覆盖真实 ONNX CUDA 节点、PT CUDA、两条包内 GPU 路径、CPU 回归、灰度 C1／C3 与双目录像；硬件为 NVIDIA GeForce RTX 4060。两份原生 CUDA profile 各含 364 次节点事件、182 个不同节点，不把它们写成 FPS 成绩。
- [packaged-evaluation-qa.json](packaged-evaluation-qa.json)：公开 COCO8 四张 Val 图完成 ONNX CPU、PT CPU、TorchScript CPU、ID32 零匹配 CPU、ONNX CUDA、PT CUDA **六个真实用例**；PT／TorchScript CPU、ONNX CPU／CUDA、PT CPU／CUDA **三组同权重比较**按报告容差通过。保留 91 个阈值点、曲线、逐图结果及当前图 PNG／JSON／CSV，ONNX CUDA 评估 profile 记录 **910 次 CUDA 节点事件**。[evaluation-qa.json](evaluation-qa.json)绑定上述矩阵和开发回归，不再执行推理。
- [conversion-qa.json](conversion-qa.json)：真实原生／Python 转换回归和最终只读包的两条转换、推理 smoke 通过。
- [folder-package-qa.json](folder-package-qa.json)：目录来源、唯一 YAML、旧绝对根／名单的受限只读迁移、公开四图 CPU 评估等十项专项通过。此 CLI 检查不替代真实选择器／Popup 的 UI 测试。
- [packaged-private-precheck.json](packaged-private-precheck.json)：用户 Val **1223 张**，813 张有标注框、410 张空标签、缺失和排除均为 0；整个数据根 **16667 个文件**前后哈希不变，原 YAML 不变，源码与包 helper 的清单相同。仅进行只读预检查，**未执行模型推理，不是足球能力评估**。私人图片、标签、配置正文、路径和完整文件清单不入库。

公开原始资料已分别归档为 `evaluation-artifacts.zip`（191 文件／30 PNG）、`folder-artifacts.zip`（34／5）、`namespace-artifacts.zip`（49／12），ZIP CRC、逐成员 SHA256 与完整 PNG 解码通过，实际窗口已经目检。这些是现有验收材料的保存，不另加测试数量；排除 runtime、模型和私人资料。QA 归档在仓库证据中保存，不增加 Release 手动附件。

[原生 UI 专项归档](native-ui-artifacts.md)另保留 16 张真实原生测试窗口与 3 份早期诊断日志，完整解码与 ZIP CRC／成员 SHA256 已核验。合成网络仅用于 UI 和调度验证，早期失败不计为通过；测试固定的旧应用标题在清单中说明，截图未修图。

README 的 [CPU 图像预览](../../../screenshots/workbench-2.5.0.png)与 [CUDA 评估页](../../../screenshots/evaluation-2.5.0.png)来自最终只读 DEB 的实际 2.5.0 窗口和公开 COCO8 Val 数据，演示界面与管线。工作台页脚的 GPU 验证状态属于该页实时状态，不能代替报告中的实际设备证明；画面耗时也不作为部署性能宣传。

## 对应源码与最终文件

[source-freeze.json](source-freeze.json)记录本版干净应用源码的冻结过程，包括保留初始输入与仅修正 scope 测试的追溯关系。[source-bundle-qa.json](source-bundle-qa.json)逐成员核验最终对应源码、18 份原始上游归档、59 个 runtime 发行包、暂存树及 DEB；4434 份冻结应用文件字节一致，2733 份嵌入源码和 2794 份 DEB 嵌入文件通过 SHA256 核验。源码排除四份无关用户训练文件，保留在用户本地，名单见报告。GitHub 自动源码 ZIP 不能替代完整对应源码包。

最终源码另与主线提交 `949727adf14692e2793a31470e1befeb5861d90e`、树 `0a04b5067ec93bd28479d4be89c730f3b078dfcd`逐文件匹配；这是源码对应性记录，不预填正式发行标签的最终提交。源码冻结后的验收报告、截图及发布凭证独立入库，不据此改写包内生产输入。

[final-assets.json](final-assets.json)记录以下最终三个手动附件：

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `vision-studio_2.5.0-1_amd64.deb` | 387201252 | `9abc153c5699852ec40b16cb7b261f12796a103642601be4dda47f464d07fe03` |
| `vision-studio-2.5.0-complete-source.tar.xz` | 883232864 | `6cda2f9be5dbda8e06e07868cf0225d30a1fe2d7c8f87a5fd83913da812b990f` |
| `SHA256SUMS` | 207 | `2066875a7fd67e577f330a3af478ce72230da59c3aa3f02eee9a5879d2052024` |

## 范围限制与后续凭证

COCO8 仅验证导入、实际推理、指标计算和同权重后端一致性；它很小，不能作为足球准确率、部署性能或泛化能力基准。原始 ID32 是 sports ball，包含多种球类。评估仍是单目标类别 AP，不提供多类总体 mAP、训练、分类评估或视频在线 mAP。本版未验收 Windows 2.5.0、物理摄像头、TensorRT 或其他新后端。

宿主升级尚待 Polkit。GitHub 正式标签／Latest、三个附件远端大小与 SHA256、精确清理和主检出同步均已实际完成，详见[发布凭证](publication.md)。本地保留 2.5.0 与 2.4.0，当前工作树／Git、独立 QA、私人备份、共享 Qt SDK/runtime 和用户资料保留；既有标签与资产未覆盖。
