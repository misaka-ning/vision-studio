# Vision Studio 2.3.1 验收报告

本报告记录 2026-10-20 实际执行的 Linux amd64 2.3.1 验收，重点为数据集文件夹／YAML 导入、受限只读路径迁移、失败后重试及已有数据集恢复。最终 DEB、完整对应源码和适用测试均已完成本地核验；随后完成的宿主升级见下方独立凭证。GitHub 发布状态以随后保存的真实发布核验凭证为准。

## 源码、构建与最终文件

[source-freeze.json](source-freeze.json) 固定主线源码提交 `cfe21880b386f09dc4ef20ac6bc38d0e90dcfd8b`；用于 `git archive` 的提交 `40418b7f74f6d2674fc923c8c6e487ad3e066933` 与其 Git 树完全相同，树 ID 为 `f60d04239cc526766b66cd712712926e7de243d5`。干净源目录保留实际 runtime-info，排除四份无关的用户训练文件。最终 QA 与发布凭证在源码冻结后独立入库，不改包内生产输入。

构建为 C++17／Qt 6.8.3 Release，生产资源根 `/opt/VisionStudio`。完整回归执行前保存 [production-inputs-before.json](production-inputs-before.json)，聚合时再次核对生产源码、资源、示例模型、CMake 配置、应用与测试二进制哈希；[ctest-qa.json](ctest-qa.json) 绑定未 strip 应用 SHA256 `d8db18d7444e431598e2c2a512843334e61ae052047a003f5be54d978b3df667`。最终包内应用 SHA256 为 `17f426bc0c84775b9364fb1f005ffb869772db52ff03f4c690cf004179107f59`，与最终暂存 ELF 一致；strip 后哈希不同于开发构建属正常过程，分别保留验证。

| 最终文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `vision-studio_2.3.1-1_amd64.deb` | 387,141,452 | `1e959057337832f2dac8d950d738426ad426ea124aee7cefc925286792d19758` |
| `vision-studio-2.3.1-complete-source.tar.xz` | 847,507,304 | `bcf9e2dcfb27e96db3c411dfa71da3e7dc248ed81ed1adb0c554ee08ba7d29e6` |
| `SHA256SUMS` | 207 | `f68adff3fd49fbad4bd73c1562a8b28ceaa904721c87a67f037fa56ac2a1e24e` |

文件身份另见 [final-assets.json](final-assets.json)、[SHA256SUMS](SHA256SUMS) 与 [DEB 构建清单](vision-studio_2.3.1-1_amd64.deb.manifest.json)。原始打包日志：[DEB](build-deb.log)、[完整源码](build-source.log)。

## 一次完整回归执行

[ctest-qa.json](ctest-qa.json) 独立复核同一完整 runner 的 **18 套测试、293 项 Qt 结果与 101 项 Python 测试，0 失败、0 跳过、0 黑名单**。Qt 数量包括各套初始化／清理及数据行，不等同于新增功能数。执行元信息与选定套数见 [原始 runner 报告](qa-full/detection-regression-qa.json)，原始发现清单见 [ctest-discovery.json](qa-full/ctest-discovery.json)。每套都有真实退出码、完整结果汇总和 JUnit，未以早期诊断或子集重跑替代完整结果。

| 测试套 | 实际通过 | 原始证据 |
| --- | ---: | --- |
| `vision-core-tests` | Qt 39 | [日志](qa-full/vision-core-tests.log) · [JUnit](qa-full/vision-core-tests.xml) |
| `vision-inference-delivery-tests` | Qt 6 | [日志](qa-full/vision-inference-delivery-tests.log) · [JUnit](qa-full/vision-inference-delivery-tests.xml) |
| `vision-recording-tests` | Qt 15 | [日志](qa-full/vision-recording-tests.log) · [JUnit](qa-full/vision-recording-tests.xml) |
| `vision-pt-tests` | Qt 36 | [日志](qa-full/vision-pt-tests.log) · [JUnit](qa-full/vision-pt-tests.xml) |
| `vision-gpu-tests` | Qt 11 | [日志](qa-full/vision-gpu-tests.log) · [JUnit](qa-full/vision-gpu-tests.xml) |
| `vision-ui-tests` | Qt 38 | [日志](qa-full/vision-ui-tests.log) · [JUnit](qa-full/vision-ui-tests.xml) |
| `vision-batch-responsiveness-tests` | Qt 5 | [日志](qa-full/vision-batch-responsiveness-tests.log) · [JUnit](qa-full/vision-batch-responsiveness-tests.xml) |
| `vision-scope-filter-ui-tests` | Qt 38 | [日志](qa-full/vision-scope-filter-ui-tests.log) · [JUnit](qa-full/vision-scope-filter-ui-tests.xml) |
| `vision-modelviewer-tests` | Qt 16 | [日志](qa-full/vision-modelviewer-tests.log) · [JUnit](qa-full/vision-modelviewer-tests.xml) |
| `vision-conversion-ui-tests` | Qt 8 | [日志](qa-full/vision-conversion-ui-tests.log) · [JUnit](qa-full/vision-conversion-ui-tests.xml) |
| `vision-conversion-inference-tests` | Qt 14 | [日志](qa-full/vision-conversion-inference-tests.log) · [JUnit](qa-full/vision-conversion-inference-tests.xml) |
| `vision-model-conversion-tests` | Python 20 | [日志](qa-full/vision-model-conversion-tests.log) · [JUnit](qa-full/vision-model-conversion-tests.xml) |
| `vision-netron-server-tests` | Python 11 | [日志](qa-full/vision-netron-server-tests.log) · [JUnit](qa-full/vision-netron-server-tests.xml) |
| `vision-gpu-runtime-tests` | Python 21 | [日志](qa-full/vision-gpu-runtime-tests.log) · [JUnit](qa-full/vision-gpu-runtime-tests.xml) |
| `vision-evaluation-worker-tests` | Qt 18 | [日志](qa-full/vision-evaluation-worker-tests.log) · [JUnit](qa-full/vision-evaluation-worker-tests.xml) |
| `vision-evaluation-panel-tests` | Qt 22 | [日志](qa-full/vision-evaluation-panel-tests.log) · [JUnit](qa-full/vision-evaluation-panel-tests.xml) |
| `vision-evaluation-python-tests` | Python 49 | [日志](qa-full/vision-evaluation-python-tests.log) · [JUnit](qa-full/vision-evaluation-python-tests.xml) |
| `vision-evaluation-integration-tests` | Qt 27 | [日志](qa-full/vision-evaluation-integration-tests.log) · [JUnit](qa-full/vision-evaluation-integration-tests.xml) |

导入专项包括 49 项评估 Python 测试、22 项面板 Qt 结果、27 项工作台集成 Qt 结果：根目录唯一约定／自定义 YAML、歧义与缺配置、默认文件夹选择器与直接 YAML、缺少 Val 后显式 Train 重试、已有数据集导入失败恢复、迁移后集合切换，以及既有三种范围、超过 50 张、停止后回看、阈值与错例、配置采用和导出。适用普通检测、转换、模型显示、录制、CPU／CUDA 后端均在上表完整套数中保留。

首轮集成测试曾错误假设失败导入会清空启动时普通演示队列；断言已修正为队列应保持原样。原始失败和重测日志完整保留于 [diagnostics/](diagnostics/README.md)，有效结论来自后续完整 18 套执行。额外 [发布工具 48 项](github-release-tests.log) 和 [版本元数据 11 项](diagnostics/vision-studio-231-release-metadata-docs.log) 为工具测试，不计入上述 18 套或产品结果。

## 最终包、真实模型与对应源码

| 核验 | 实际结果与范围 | 报告与原始日志 |
| --- | --- | --- |
| 最终 DEB | 普通用户解包、系统依赖／APT 模拟、离线 CPU 推理、模型显示与缓存、两种转换后真实推理；无错误或警告 | [报告](qa-report.json) · [日志](qa-report.log) |
| 隔离安装与卸载 | 私有 dpkg 安装／purge、断网普通用户运行、安装树只读、用户结果保留；宿主 dpkg／集成入口保持不变 | [报告](install-qa.json) · [日志](install-qa.log) |
| 最终包 CPU／CUDA | ONNX CUDA、PT CUDA、ONNX CPU、PT CPU 四场景；实际设备与后端核对 | [包内报告](packaged-gpu-qa.json) · [日志](packaged-gpu-qa.log) · [GPU 汇总](gpu-qa.json) |
| 模型转换 | 完整原生／Python 转换回归，加最终只读包 ONNX／TorchScript 转换与真实推理 | [汇总](conversion-qa.json) · [原生原始日志](qa-full/vision-conversion-inference-tests.log) · [Python 原始日志](qa-full/vision-model-conversion-tests.log) |
| 完整对应源码 | 4176 个冻结应用源码逐文件／字节一致，2733 个部署源码文件、59 项实际 runtime 元数据、18 份原始上游材料；实际 DEB 成员映射与输入不变核对 | [报告](source-bundle-qa.json) · [日志](source-bundle-qa.log) · [源码清单](SOURCE-INVENTORY.json) · [内部校验](SOURCE-SHA256SUMS) |

GPU 实测设备为 **NVIDIA GeForce RTX 4060**。完整 GPU 套的两份 ORT profile 各含 364 个实际 `CUDAExecutionProvider` 节点事件，分别保留在 [彩色 profile](qa-full/artifacts/onnx-cuda-0_2026-10-20_18-24-59.json)、[灰度 profile](qa-full/artifacts/onnx-cuda-1_2026-10-20_18-24-59.json)。[gpu-qa.json](gpu-qa.json) 的七项检查分别来自这些实际 trace、灰度 C1／C3、真实 PT CUDA、左右目与录像及四轮最终包验收；事件数不是吞吐量或性能保证。

### 公开数据集真实评估

[packaged-evaluation-qa.json](packaged-evaluation-qa.json) 与 [原始日志](packaged-evaluation-qa.log) 使用公开 COCO8 Val 的 4 张图片和原始 GT，完成 ONNX CPU、PT CPU、TorchScript CPU、ID32 零结果 CPU、ONNX CUDA、PT CUDA **六个最终包场景**；PT／TorchScript CPU、ONNX CPU／CUDA、PT CPU／CUDA **三组同权重比较**按报告中的数值容差通过。报告与预测缓存、阈值表、错例、选图 PNG／JSON／CSV 导出以及实际设备均核对；最终 ONNX CUDA 评估 trace 含 910 个实际 CUDA 节点事件。汇总见 [evaluation-qa.json](evaluation-qa.json)。

[folder-package-qa.json](folder-package-qa.json)／[日志](folder-package-qa.log) 另外验证最终只读 DEB 的文件夹导入：复制公开 4 张图片与标签并逐字节核对，构造失效绝对根与绝对名单路径、保留另一个训练 YAML，确认选择唯一约定配置、保留目录来源和迁移提示、实际 ONNX CPU 评估、91 个阈值点及当前图片三格式导出，共 **10 项检查通过**。这条专项不计为额外一套 CTest，也不作为真实足球能力测试。

### 用户目录只读预检查

[private-dataset-precheck-summary.json](private-dataset-precheck-summary.json)／[日志](private-dataset-precheck.log) 对用户实际目录只读预检查：Val 1223 张，813 张有框、410 张空标签，缺标签和排除均为 0。失效绝对根路径按明确选择的目录迁移，源 YAML 与共 2448 个输入文件的聚合哈希保持不变，没有自动选择 metadata，没有模型推理。

[packaged-private-precheck.json](packaged-private-precheck.json)／[日志](packaged-private-precheck.log) 使用最终包暂存的实际 helper 复核，三个 helper 与冻结源码一致，清单匹配源预检查且绑定最终 DEB。这里只证明解析、图片／标签验证和只读迁移；没有对 1223 张执行准确率或模型能力评估。公开证据仅保留统计、哈希与检查范围，私人原图、标注、文件名、路径与完整清单未入库。

## 截图、曲线与小型原始素材

- [namespace-artifacts.zip](namespace-artifacts.zip)、[逐项 SHA 清单](namespace-artifacts.json)、[说明](namespace-artifacts.md)：49 个隔离安装原始小文件，12 张 PNG 全部 Pillow 完整解码，另目检工作台与 ONNX 层级树两帧。只含公开 bus 示例 QA 输出，排除运行环境、模型与个人历史。
- [evaluation-artifacts.zip](evaluation-artifacts.zip)、[逐项 SHA 清单](evaluation-artifacts.json)、[说明](evaluation-artifacts.md)：192 个公开 COCO8 评估文件，30 张 PNG 全部解码，另目检 CUDA 分析页与 PT 标注预览；包含 provenance、原始日志、候选、报告、曲线、三格式导出和真实 profile，不复制大型解包树／模型／缓存。
- [native-ui-artifacts.zip](native-ui-artifacts.zip)、[逐项 SHA 清单](native-ui-artifacts.json)：冻结 Qt 集成测试二进制的两个专项场景及 3 张实际 offscreen 窗口截图。该次日志为 **4 项通过（含初始化／清理），不与完整 27 项 Integration 或 293 项 Qt 重复累加**；1380×940 两张、1260×820 一张均完整解码并目检。合成图片与标注仅验证确定性交互，不代表模型准确率。

- [folder-artifacts.zip](folder-artifacts.zip)、[逐项 SHA 清单](folder-artifacts.json)：最终包的文件夹迁移专项原始请求、stdout／stderr、候选、报告、曲线与导出，以及公开测试目录的配置与名单；不包含私人数据。

四个 ZIP 仅是随仓库保存的 QA 证据，不额外上传为零散 Release 附件。CRC、逐项 SHA256 及图像检查范围分别保留于清单和包内说明。

## 已知限制与结论边界

- 本轮没有 Windows 2.3.1 构建／安装或物理摄像头验收。灰度、左右目和录像结果只证明实际已执行测试条件下的行为。
- COCO8 很小，覆盖公开管线集成与同权重后端一致性，不是部署基准或足球准确率。ID32 为 sports ball，零 GT 场景不证明足球专用能力；合成 UI 标注同样不能作为能力宣传。
- 1380×940 的专项截图中，既有右侧检测摘要区域部分顶部文字受到压缩裁切；1260×820 截图该摘要可见。新增数据集输入行、箭头与迁移提示可见。该既有布局观察已如实记录在原始视觉检查中，本轮没有修改冻结 UI 以掩盖它，不能将全部面板布局宣称为无保留通过。
- 只读迁移仅限明确选择文件夹、旧绝对根不存在、原路径不存在且位于旧根内的路径；直接选择 YAML 仍严格解释原路径，迁移不自动修改配置或数据。
- 隔离命名空间安装与随后真实宿主升级分别记录；GitHub 发布结果由独立凭证确认。历史发行标签和资产未被本轮验收覆盖或移动。

## 随后实际宿主升级

[host-install-qa.json](host-install-qa.json)、[DPKG 原始日志](host-install.log) 与 [版本输出](host-version.log) 记录本机从 `2.3.0-1` 升级到 `2.3.1-1`。`vision-studio --version` 实际输出 `Vision Studio 2.3.1`，9 个安装代码／模型文件匹配最终 DEB，5 个已有用户文件不变；小型设置备份仅本地保留。安装后的 helper 再次只读预检查私人 Val 1223 张，清单与安装前一致，没有执行私人数据集模型推理。私人备份与完整清单不入库。
