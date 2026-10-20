# Vision Studio 2.4.0 验收报告

本报告记录 2026-10-20 实际执行的 Linux amd64 2.4.0 验收。本版退出独立图片／普通图片文件夹入口，保留标准 YOLO 数据集、视频与摄像头；重新设计紧凑任务条和折叠数据集设置。C++17／Qt 6.8.3、Fluent 深色界面、既有模型后端与个人资料保留。

## 冻结源码与最终文件

[source-freeze.json](source-freeze.json) 固定主线源码提交 `9d57e96860a2b7388594fb402ea52eaa363fa886`，其树与打包输入提交 `9d57e96860a2b7388594fb402ea52eaa363fa886` 完全一致，树 ID `96a8cbec0c10fb2da226c1588117a4d499fc32df`。完整源码来自该干净 Git 导出和实际 runtime-info，排除四份无关的用户训练文件。验收证据、README 实拍截图和发布凭证在源码冻结后独立提交，不改包内生产输入。

生产二进制未 strip SHA256：`5813c1bf7a3970b5c51746479a78c5acb0fc81048be8f5a36ef4574f23d6484e`；包内实际 ELF SHA256：`3f83fbdc4f5fe277f13ea08332a88bbfd8adb4f889342ec75e0873f6b2bfcbd1`。完整对应源码、18 份上游归档、59 个运行时发行包及嵌入源码映射已由[source-bundle-qa.json](source-bundle-qa.json)绑定最终文件核验；不能用 GitHub 自动源码 ZIP 替代完整对应源码。

| 手动附件 | 字节 | SHA256 |
| --- | ---: | --- |
| `vision-studio_2.4.0-1_amd64.deb` | 387150092 | `a49aac15daf9733d42126f84d7c6dc64256dbac46da21cb4d9d8ed58d2effe68` |
| `vision-studio-2.4.0-complete-source.tar.xz` | 862570864 | `5450efd8c22eabaad294fa4f04bcf574fd1ccf1eaeca00644a133006a3a0b14c` |
| `SHA256SUMS` | 207 | `c86d51207583310322dbee8eb9165197903045ce90755ff973bdc28ba40335f7` |

## 完整回归

[ctest-qa.json](ctest-qa.json) 独立解析本轮原始日志与 JUnit，记录 **18 套、302 项 Qt、101 项 Python**；有效结果无剩余失败、跳过或 blacklist。有效结果由原完整执行中的 17 套通过结果，加上修正测试输入后的转换集成 suite 组成。原执行明确保留该 suite 的 12 项失败，均为未准备输入导致 starts.size()==0；修正后十二个相同数据行逐一 PASS。新增一次 GPU 回归由 runner 强制附加，仅单列验证而不重复计数。前后仅一个测试源码与其测试二进制改变，应用、其他测试、模型资源、CMake、运行时元数据与依赖锁哈希均未改变；[严格聚合工具](summarize_fixture_repair.py)验证两个快照、实际失败原因及案例集合。真实 NVIDIA 硬件为 RTX 4060；CUDA profiles 随证据保留。辅助发布工具另有 [48 项实际单测日志](github-release-tests.log)，不加入上述计数。

| 套件 | 实际通过 | 原始证据 |
| --- | ---: | --- |
| vision-core-tests | 39 | [原始日志](qa-full/vision-core-tests.log) |
| vision-inference-delivery-tests | 6 | [原始日志](qa-full/vision-inference-delivery-tests.log) |
| vision-recording-tests | 15 | [原始日志](qa-full/vision-recording-tests.log) |
| vision-pt-tests | 36 | [原始日志](qa-full/vision-pt-tests.log) |
| vision-gpu-tests | 11 | [原始日志](qa-full/vision-gpu-tests.log) |
| vision-ui-tests | 39 | [原始日志](qa-full/vision-ui-tests.log) |
| vision-batch-responsiveness-tests | 5 | [原始日志](qa-full/vision-batch-responsiveness-tests.log) |
| vision-scope-filter-ui-tests | 38 | [原始日志](qa-full/vision-scope-filter-ui-tests.log) |
| vision-modelviewer-tests | 16 | [原始日志](qa-full/vision-modelviewer-tests.log) |
| vision-conversion-ui-tests | 8 | [原始日志](qa-full/vision-conversion-ui-tests.log) |
| vision-conversion-inference-tests | 14 | [原始日志](qa-conversion-fixture-rerun/vision-conversion-inference-tests.log) |
| vision-model-conversion-tests | 20 | [原始日志](qa-full/vision-model-conversion-tests.log) |
| vision-netron-server-tests | 11 | [原始日志](qa-full/vision-netron-server-tests.log) |
| vision-gpu-runtime-tests | 21 | [原始日志](qa-full/vision-gpu-runtime-tests.log) |
| vision-evaluation-worker-tests | 18 | [原始日志](qa-full/vision-evaluation-worker-tests.log) |
| vision-evaluation-panel-tests | 24 | [原始日志](qa-full/vision-evaluation-panel-tests.log) |
| vision-evaluation-python-tests | 49 | [原始日志](qa-full/vision-evaluation-python-tests.log) |
| vision-evaluation-integration-tests | 33 | [原始日志](qa-full/vision-evaluation-integration-tests.log) |

## 本版专项与真实界面

[native-ui-artifacts.md](native-ui-artifacts.md)、[逐项清单](native-ui-artifacts.manifest.json)和 [ZIP](native-ui-artifacts.zip)保留实际 Qt 窗口、日志及早期失败诊断。初轮空队列指标断言已按新设计更新；高级参数展开导致的画布裁切通过合并审核文件／扫描行与收紧间距修复，原始失败证据保留。最终独立 Panel 24／Integration 33 项通过，但不与完整回归重复累加。

1260／1380 × 820、CPU／CUDA 选择器布局：折叠顶栏 96px，画布高 369px；展开高级设置顶栏 182px，画布高 283px，窗口和父容器均完整包含。长错误限制首行摘要及 44px 高度，完整信息保留 tooltip／辅助阅读；无效评估保持禁用。数据集目录／YAML 拖入、Ctrl+O、图片拖入拒绝、启动空画布及旧 demo CLI 退出均有实际检查，[公开 CLI 原始结果](public-entry-qa/public-entry-qa.json)另存。

原有数据集三范围、61 张队列、停止／保存响应、已完成结果回看、当前图导出、映射、类别筛选、灰度／双目／录像、全局模型、转换及历史报告由相应完整回归覆盖。UI 合成图片只证明交互与几何，不代表准确率；CUDA 选择器布局本身也不证明 GPU 推理。

README 两张截图来自最终只读 DEB 的真实 COCO8 CPU 预览与 CUDA 分析页，已目检；相关素材见下方公开矩阵。

## 最终 DEB 与安装

- [qa-report.json](qa-report.json)：最终包权限、依赖、ELF、原始许可及只读 CPU／PT／结构／转换等真实 smoke，APT 依赖模拟通过。
- [install-qa.json](install-qa.json)：只读宿主中的 bubblewrap 命名空间实际安装、普通用户运行、卸载及资料保留验收；非宿主安装的替代声明。
- [packaged-gpu-qa.json](packaged-gpu-qa.json)：最终包 ONNX CUDA、PT CUDA、ONNX CPU、PT CPU 四条实际路径，记录所请求和实际设备；[gpu-qa.json](gpu-qa.json)另绑定完整回归中的灰度 C1／C3 与双目录像。
- [packaged-evaluation-qa.json](packaged-evaluation-qa.json)：公开 COCO8 四张 Val 图，六个真实用例、三组同权重后端比较、91 个阈值点、曲线与当前图三格式导出，CUDA profile 含 910 个节点事件。
- [folder-package-qa.json](folder-package-qa.json)：最终包目录来源、唯一配置、失效路径迁移和列表、真实四图 CPU 评估等 10 项专项通过。
- [packaged-private-precheck.json](packaged-private-precheck.json)：用户 Val 1223 张，813 张含框、410 张空标签，缺失和排除 0；冻结源码与包 helper manifest 完全相同，整个数据根 16667 个文件前后哈希不变。此项仅只读预检查，无模型推理，不是能力评估。私人清单、路径、原图和标签不入库。
- [宿主升级状态](host-install-pending.json)：宿主升级等待系统 Polkit 管理员验证，当前不声称完成宿主安装；隔离命名空间安装已实际通过。私人快照与设置备份仅本地保留。

[namespace-artifacts.md](namespace-artifacts.md)、[evaluation-artifacts.md](evaluation-artifacts.md)、[folder-artifacts.md](folder-artifacts.md)说明三个原始公开资料 ZIP：共 274 个文件、47 张 PNG，逐项 SHA／ZIP CRC 与完整解码通过；排除 runtime、模型、提取源码和私人数据。QA ZIP 留在仓库，Release 仅三个手动附件。

## 边界与发布状态

没有验收 Windows 2.4.0、物理摄像头、TensorRT 或新模型架构。COCO8 很小，只证明管线、指标和同权重后端一致性，不是足球精度或部署性能基准；ID32 为 sports ball，不等于仅足球。历史 Release 标签和资产不覆盖。

GitHub 发布、Latest、三个资产大小与 SHA256 以及保留最新两版的结果，以下游独立发布凭证为准。旧 2.3.0 的[清理预审](cleanup-old-230-preview.json)是只读计划；远端未核验时不执行删除，个人资料、其余 QA、共享 SDK/runtime 始终保护。

## 正式发布与本地保留

正式标签 `v2.4.0` 固定为 `ba0e38604ba81b430e9234ea122c208439a7b1cd`。官方工具的[预检查](release-dry-run.json)、[发布凭证](release-publish.json)、[独立复核](release-verify.json)及 [Latest 原始响应](latest-release.json)确认：发行已公开、非预发布、Latest 为 2.4.0，只有三个手动附件，大小与 SHA256 全部吻合上表。

新旧远端核验通过后，官方工具分别[预览](prune-worktree-preview.json)并[执行工作区发行清理](prune-worktree-executed.json)，[主目录同样完成](prune-primary-executed.json)。两处发行副本只保留 2.4.0、2.3.1。源码／构建／暂存另经[逐成员复核与精确清理](old-source-cleanup.json)：只删除已审计的三个 2.3.0 目录，独立 QA、用户备份、Git、SDK 与 CPU／GPU runtime 保护。两个未发布的重复源码诊断归档也按记录 SHA 精确删除，原校验、日志及 Git 修订保留。

[最终本地状态](final-local-state.json)记录对应版本目录及当时宿主状态。本机升级仍待 Polkit 管理员验证；安装包与源码已公开，隔离安装验收已经完成。这里的随后凭证以维护提交保存，不改冻结标签或资产。
