# MainWindow 重构行为基线与权限矩阵

记录日期：2026-10-29。本文记录重构开始前的 **2.10.0 行为与数据所有权**，不是新架构已经实现的声明。产品生产源码为 `cecd6ea12552e247b686dff090e2dbdffd4b4c93`；最初只读盘点 HEAD 为 `c61690bbde639928e15372da9fe9185d443034b5`。随后同步到 `c459ba4` 的远端变化是历史发行说明；本轮工作分支为 `codex/mainwindow-architecture`。实施中的新文件、类型和枚举可以调整，以下外部行为需保持。

## 证据等级与执行状态

- **静态核实**：阅读源码、CMake、测试和既有报告，确认下面的状态、守卫和契约；静态字符串或源码依赖检查不能证明应用运行、响应速度或模型输出正确。
- **历史真实执行**：2.10.0 原始 `ctest-qa.json` 记录 26 套、436 项 Qt、159 项 Python，失败／跳过／blacklisted 均为零。它只属于旧版本，不能替代本次重构回归。
- **本次重构前行为基线**：`output/v2.10.1/baseline` 首轮因沙箱中 ModelViewer 本地端口操作返回 EPERM 导致 batch suite 失败，原件保留；隔离宿主重跑的 `output/v2.10.1/baseline-host/detection-regression-qa.json` 已实际报告 success、errors 为空，六套 core、范围筛选、批量响应、输入拖入、数据集恢复、评估集成测试均退出 0、无跳过。此报告记录的产品版本仍为 2.10.0，是重构前基线，不是新架构完整验收；后续通过不抹去首轮失败。
- **阶段一实际构建**：由新快照 `output/v2.10.1/stage1-source` 配置全新 `output/build/2.10.1-stage1`，`vision-studio` 目标构建成功，原始日志为 `output/v2.10.1/stage1-clean-build.log`。这只证明该目标的实际构建，不表示全部测试目标或最终包已经验收。
- **本次实际静态检查**：`tests/architecture_tests.py` 的 4 项只读检查已执行通过；阶段一独立重跑日志为 `output/v2.10.1/stage1-architecture-tests.log`，再次 4 项通过，不将两次结果叠加为 8 项。检查依赖方向及既有 CLI 名称词面，没有运行应用／模型或生成用户数据；不能与 Qt／模型功能测试数量混计，也不证明重构功能通过。
- **尚未完成**：新架构全部阶段的完整回归、最终包、安装及发布验收，应在各阶段与同版最终证据中另记，本文不预填通过。

## 当前状态权威与写入边界

这里的“权威”是行为由谁最终决定，不代表所有数据都应该继续存放在 MainWindow。后续迁移必须一次转移到唯一拥有者；旧字段只能暂时作为委托接口，不能长期保留可写副本。

| 领域 | 2.10.0 的权威状态／来源 | 当前写入与派生边界 |
| --- | --- | --- |
| 全局模型 | `modelPath_`，有序 `models_`，名称／备注／转换配置／评估配置四个 map | `setModel` 统一影响工作台、模型显示与转换来源；批量导入先全批次校验，规范路径去重，复用旧 key，按序追加，仅选用去重后最后一项。显示名不改变真实文件名。 |
| 模型类别与身份 | `modelClassCount_`、`modelInputChannels_`、`nativeLabels_`、`inspectedModelIdentity_`、检查错误／pending／generation／取消令牌 | 后台 `VisionEngine::inspectModel` 结果只有在 generation、取消状态和当前文件身份仍匹配时提交；不能把元数据推断成已完成真实推理。 |
| 模型结构 | `ModelViewer` 持有当前模型解析、图／树／参数／YAML 和服务缓存 | MainWindow 只给选用模型并展示状态；相同未变化文件不应重启解析，旧异步结果不得覆盖新选择。 |
| 当前输入与队列 | `sourceKind_`、`files_`、`streamPath_`、队列行和选择 | 队列顺序由输入会话决定；视频／摄像头不使用图片范围。临时图片属于当前队列，不属于磁盘数据集、GT 或恢复记录。 |
| 数据集 | `datasetManifest_`、`datasetSamples_`、导入 generation／取消状态 | `EvaluationWorker` 预检查后才提交 manifest；分片建队列、异步图片读取。失败或取消不能把旧有效集合错误改名或写成新恢复记录。 |
| 启动恢复 | `DatasetSessionStore` 的独立 JSON，以及用户交互／恢复 generation | 只有完整成功导入后原子保存原始目录／YAML、split 和可选 metadata；恢复不自动推理。手动选择／取消／停止优先于延迟恢复。 |
| 待执行配置 | `currentConfig()` 从控件、实际模型类别及已采用配置派生 | UI 当前值不等于已运行的模型或设备；实际后端反馈另存。类别映射、类别筛选与标签名称是不同概念。 |
| 推理任务快照 | `JobRequest`、`taskRangeFiles_`、首尾索引、`activeModelIdentity_`、`activeImageFingerprint_`、每图提交身份 | 点击开始固定本次范围和顺序；worker 不从正在变化的控件取参数。进度只计算任务范围。 |
| 普通检测结果 | `imageResults_` 中的结果／配置／源身份／无损快照；当前 `lastResult_` | 接收结果后先可回看，再后台保存；不能用后来磁盘版本给已完成结果重新贴身份。模型／参数指纹失效后不能显示旧预测；零匹配仍是有效结果。 |
| 评估任务与报告 | `EvaluationRequest`、`evaluationRunConfig_`／指纹、`evaluationReport_`／目录、已评估索引 | 报告与原始候选是已执行事实；`EvaluationPanel` 持有显示副本、分析阈值和筛选 UI，不拥有全局模型选择或推理调度。 |
| 错例与阈值浏览 | 当前分析阈值／筛选、已完成样本索引、review generation／取消令牌 | 阈值变化复用候选重匹配，不重新推理；错例仅在已完成且可校验的原数据集样本中统计；浏览与推荐不能自动采用检测配置。 |
| 线程、任务与保存 | `InferenceWorker`／`workerThread_`、`EvaluationWorker`、`resultIoPool_`／pending、busy 与停止状态 | GUI 请求停止不等待推理／保存队列排空；图片结果 ticket 背压控制投递，保存回调需正确释放／确认 ticket。析构清理另有等待，不能与交互停止混淆。 |
| 录像与回放 | 录制 requested／active／stopping、worker 的线程安全录制命令、录制目录；独立播放 capture | 录制的是实际处理后的颜色与单目画面；切换路由会停止非录像页回放，不应停止正在执行的推理。 |
| 用户设置与历史 | `preferences.ini`、`history.json`、数据集 session、用户模型／录像／评估目录 | 设置写入是持久化出口，不可成为第二份运行状态权威；历史报告打开不自动切换全局模型。重构不迁移／删除／改写用户原件。 |
| GPU 环境 | probe／setup 进程及其反馈、CPU／GPU runtime helper | 设备偏好与实际执行设备分别记录；准备环境不更改驱动，取消／退出清理保持。 |
| 路由与布局 | `pages_`／导航、工作台子页及分析 tabs；控件对象名 | 路由改变只处理导航／必要显示与回放；不能因切页推理、改阈值或清缓存。布局层不直接写业务状态或磁盘。 |

## 权限谓词：记录现状，不暗中统一行为

记号均表示 2.10.0 的实际字段条件：

```text
B = busy_
Q = queuePopulating_
D = droppedImageImportActive_
E = evaluationWorker_ != nullptr
G = gpuSetupProcess_ != nullptr
I = resultIoPending_ != 0
C = closing_
X = modelConversion_ && modelConversion_->isBusy()
M = QApplication::activeModalWidget() != nullptr
P = modelInspectionPending_
L = B || Q || D || E || G || I || C || X || M
```

`L` 是原 `previewDropLocked()`，不是所有操作共同采用的唯一守卫。实现集中权限时，要分别保留“控件能否操作”“命令入口守卫”“异步提交重验证”，不能把 `!L` 一次替换所有方法而悄悄改变行为。

| 操作／层级 | 原始必要条件与二次校验 | 拒绝或停止后的保证 |
| --- | --- | --- |
| 预览外部拖入 | `!L`；URL 必须为本地文件，交给 `classifyPreviewDrop` 区分数据集／视频／图片并全批次验证；drop 阶段由 `handlePreviewDrop` 再检查忙状态 | 混合类别不能部分提交；已有数据集时按替换／临时追加／取消处理。原预览入口没有模型库独有的 host／query／fragment 与 only-move 检查，不能误记成同一个验证器。 |
| 模型库外部拖入／模型文件导入入口 | `!L`；URL 有复制 action，本地 URL 无远端 host／query／fragment；全批次可读、非空且实际后缀受支持；drop 时重新验证 | 无效混合批次不能部分提交；only-move／only-link 拒绝，源文件不搬动；QListWidget 自身内部 MIME 留给内部排序处理。按钮文件选择器共用文件校验，不走 URL MIME 判断。 |
| 全局 `setModel` | 方法入口只拒绝 `B`；非空路径须是文件；PT 环境准备可取消；相同路径且有效身份走缓存分支 | 不应把这个较窄入口守卫误记为 `!L`；正常用户入口还受外部控件／导入守卫约束。 |
| 名称／备注／移除 | `!B && models_.contains(path)`；模态编辑确认后再次检查 busy 和条目仍存在 | 菜单位置动作可在 busy 时使用；修改动作禁用。显示重命名不改文件，移除不删原文件。 |
| 启动按钮通用 ready | `!B && !Q && !G && !I && !modelPath_.isEmpty() && rangeError.isEmpty() && !X` | 图片后两种范围缺少实际选中行时禁用，不回退全队列。 |
| 预览页开始按钮 | 通用 ready 且 `classFilterError_.isEmpty()` | 分类任务不应用检测类筛选；筛选基于实际类别信息，保留原始 ID。 |
| 评估按钮 | 通用 ready，加图片来源／非空 manifest／有效非空评估范围／`EvaluationPanel::canEvaluate()`／`!P` | 临时图片无 GT，不进入评估分母；不能因有队列就视作可评估数据集。 |
| `startWorkbenchTask` | 入口拒绝 `B || Q`；评估子页额外检查 `canEvaluate()`；按子页分发 | 切换子页本身不启动任务。 |
| `startInference` | 入口拒绝 `B || G || X`；评估模式转交评估；PT runtime 就绪；拒绝 `I`、类别错误、范围错误、空模型或图片来源空队列；自动导出目录须可创建 | 固定请求快照；已完成且同指纹的范围外结果保留。入口条件不与按钮表达式简单等同。 |
| 数据集导入 | `!(B || E || G || X) && !C`，runtime 就绪；提交检查 worker／generation／取消／closing | 只在完整成功载入后更新独立 session；失败／停止保留旧有效记录。 |
| `startEvaluation` | runtime 就绪；`!(B || E || I)`、panel 可评估；类别检查完成且身份当前；评估范围有效 | 冻结配置和样本；不完整／中断任务不冒充完整集合指标。 |
| 重新分析／导入报告 | runtime 就绪；前者 `!(B || E)` 且报告及目录非空；后者 `!(B || E)` 且来源非空 | 单独生成结果，不改写来源报告；导入未知后端信息继续未知。 |
| 显式采用评估配置 | `!B` 且报告非空、`allow_apply_profile` 允许；当前模型绝对路径及 SHA256 与报告请求一致 | 只有明确采用才改检测配置；分析阈值变化／推荐定位／错例浏览不修改全局配置。 |
| 录制按钮／命令 | `B && sourceKind_ != Images && !recordingStopping_ && !inferenceStopping_ && !failed_` | 只提交轻量录制命令；结束录制不阻塞 GUI 保存视频。 |
| 手动导出 | 按钮／快捷键还受 idle 限制；`exportResult` 方法本身只检查图像非空且非 demonstration | 导出必须对应当前选中结果与其配置；不能只检查画布有图就导出其他图片的预测。 |
| GPU 检查／准备 | 拒绝 `B || gpuSetupProcess_ || gpuProbeProcess_`；准备先确保 CPU runtime | 保持环境进度、取消与失败行内反馈；不把偏好 CUDA 当作实际 CUDA。 |
| 启动恢复 | 未请求过且无需下载 runtime；默认开关开启；延迟回调再次检查 generation／用户操作／`C,B,Q,E,G,X,M`，还须图片来源且队列及 manifest 为空 | 用户选择优先；无任务自动执行；关闭恢复开关保留记录。注意这个谓词没有直接使用 `L`。 |
| 关闭窗口 | 先取消延迟恢复／相关异步读取；转换中取消并推迟关闭，GPU setup 中取消并推迟关闭；`B || I` 时停止或等保存完成后关闭；否则 persist 后接受 | 不能直接析构活跃线程，也不能因按钮 stop 等待磁盘而卡主窗口。 |

`EvaluationPanel::canEvaluate()` 精确等价于 `validationMessage().isEmpty()`。后者逐项检查 manifest、缺失／无效标签、样本与类别名称、检测任务、模型路径及实际类别数、映射范围、扫描起止／步长／最多 1001 点、候选下限、ONNX 线程限制及至少一个图表格式；不能简化成“已载入数据集”。

### 停止的分支顺序

`stopInference` 总是先取消延迟恢复。`!B` 时返回；临时图片追加中撤销本次追加；队列建立中使 generation 失效并取消建队列；评估中只调用 `requestStop()` 并保留已完成样本；普通推理中设 stopping、撤销录制请求并调用 worker 的轻量 `requestStop()`。主线程不在交互停止路径调用 `wait()` 或等待保存池。完整退出时析构会等待线程／保存池，这属于另一个生命周期。

## 必须保留的接口和数据契约

- 公共运行入口、`startRequested(vision::JobRequest)`、应用版本、命令行退出语义与已支持的 `VISION_STUDIO_*`／XDG 隔离方式保持；委托给新控制器不能绕过真实流程。
- `preferences.ini` 的模型列表、activeModel、名称／备注、转换／评估 profile、预处理／设备／自动导出／类别筛选键，以及默认开启的 `restoreDatasetOnStartup` 保持；独立 `dataset-session.json` 仍由 `DatasetSessionStore` 原子读写。
- 模型／源文件身份、任务快照、普通检测缓存、评估候选／报告快照不能混为一种缓存。普通检测回看与当前阈值下评估回看使用各自配置和身份。
- 保留祖先作用域 `imagePreviewPage`、`modelLibraryPage`；以及 `globalModelList`、`modelLibraryDropHint`、`modelStructureViewer`、`activeModelName`、`conversionSourcePath`、`inputQueue`、`evaluationWorkspacePage` 等 objectName。若换成新的 Qt 页面类型，先保持事件作用域与可查询对象；新增 typed accessor 后再逐步减少 friend，不为“测试通过”留下第二份业务状态。
- 旧路由语义：0 工作台、1 模型库、2 模型显示、3 运行记录、4 使用指南、5 录制视频、6 模型转换、7 更多。侧栏视觉排序为工作台／模型库／模型显示／运行记录／录制视频／模型转换／使用指南／更多。工作台子页为预览 0／评估 1。可改 typed enum，不能因枚举编号调整改变用户入口。
- MainWindow 作为 QObject 连接上下文时承担自动断开与过期回调保护。把 lambda 移到服务后必须明确接收者生命周期、线程归属和取消令牌。`InferenceWorker` 的 ticket 确认／背压不能被遗漏。

### CLI 与 smoke

| CLI | 既有结果文件／主要范围 |
| --- | --- |
| `--screenshot path` | 实际窗口截图后退出。 |
| `--smoke directory`、`--smoke-pt directory` | `smoke-report.json`，真实 ONNX／PT 推理及 PNG／JSON／CSV。 |
| `--smoke-model directory`，可配 `--display-model path` | `model-display-report.json`，实际模型结构／树／参数／可用 YAML 与缓存。 |
| `--smoke-conversion directory`，配 `--conversion-source`／`--conversion-format` | `conversion-smoke-report.json`，真实转换及推理验证。 |
| `--smoke-evaluation directory --evaluation-request path` | `evaluation-smoke-report.json`，真实数据集评估、配置／阈值／回看与导出；可含现有侧栏检查。 |
| `--smoke-dataset-session directory` | `dataset-session-smoke-report.json`，正常启动恢复，不推理。 |
| `--smoke-preview-drop directory --preview-drop-request path` | `preview-drop-smoke-report.json`，真实 Qt 拖入、临时图片、原集合评估。 |
| `--smoke-model-library-drop directory` | `model-library-drop-smoke-report.json`；固定 21 项观察、14 组事件、ONNX／PT／本轮真实 TorchScript 解析、选中 ONNX CPU 实际推理。 |
| `--smoke-cpu-runtime directory` | `cpu-runtime-setup-report.json`，环境准备；不能在纯重构回归中悄悄触发下载。 |

保留 `--smoke-input-root`、`--device auto|cpu|cuda` 与 `--gpu-index`。smoke 的 JSON 名称、检查名、导出命名和实际／请求设备区分是包验收输入，不能只保留 CLI 名字而用假结果替代。模型库专项 `restart_executed=false`；原生 UI 的关闭后重建 MainWindow 也不能冒充独立进程重启。

## 关键测试与迁移门禁对应

| 改动领域 | 应保留／运行的现有测试与实际关注点 |
| --- | --- |
| 全局模型、模型库 | `vision-ui-tests`：`modelLibraryExternalDropsAppendAtomicallyAndSelectOnce`、`modelLibraryDropsRejectInvalidBatchesWithoutMutations`、`busyModelLibraryDropsCannotReplaceCurrentTask`、`modelLibraryDialogCancelKeepsLibraryAndSelection`、`modelLibraryInternalMovesPreserveGlobalSelectionAndPersistAfterRestart`；另有备注／名称／移除／context／结构缓存测试。 |
| 图片任务与缓存 | `vision-scope-filter-ui-tests`：`fixedRangeAndProgress`、`filtersAndScope`、`validCacheOutsideRangeAndChangedParameters`、`stoppedRangePreservesCompletedResultsAndExport`、外部模型／源替换身份测试；`vision-ui-tests` 包含超过 50 张、停止后回看、后续读图失败和零匹配回看。 |
| 异步响应／保存 | `vision-batch-responsiveness-tests`：大队列懒加载／过期预览、高清自动保存时停止、磁盘失败后的恢复；结果 ticket 背压与停止不能只测按钮状态。 |
| 预览拖入与 GT | `vision-input-drop-ui-tests`：MIME 顺序、混合整批拒绝、替换／临时追加／取消、忙时拒绝、临时项无 GT、评估只含原标注集、取消追加恢复旧队列。 |
| 数据集恢复 | `vision-dataset-session-tests` 的原子存储；`vision-dataset-session-ui-tests` 的三种 split、失败／取消保留旧记录、用户选择优先、开关取消延迟恢复、导航目标保持。 |
| 评估控制与采用 | `vision-evaluation-integration-tests`：范围／指标、当前阈值错例、零错不扩范围、当前选中导出、报告关联不改模型、未知信息保留、真实 PT batch／尾批／采用、停止前缀回看。 |
| 独立界面组件 | `vision-modelviewer-tests`、`vision-evaluation-panel-tests`、`vision-conversion-ui-tests`；保持模块自己的加载／缓存／交互与格式规则。 |
| 预处理、设备、录制、转换 | `vision-core-tests`、`vision-pt-tests`、`vision-gpu-tests`、`vision-recording-tests`、`vision-conversion-inference-tests`；真实模型检查不能由静态边界测试替代。 |
| 最终发布 | 同版完整回归、最终只读包 CPU／CUDA、各 smoke、源码对应、隔离安装与远端核验；Qt 单元测试或旧包通过不能代替新包验收。 |

初期迁移保持这些断言的语义。字段移到控制器时，测试应跟随唯一状态拥有者或面向稳定行为接口；不要删断言、增加可写 shadow 字段或给生产 API 填伪造状态来维持旧 friend 测试。

## 构建与 QA 复用

已有 Qt 6.8.3 SDK、系统 OpenCV、CPU runtime 与 GPU runtime 可复用，不需要重装：Qt 为 `$HOME/Qt/6.8.3/gcc_64`，CPU 为 `/opt/VisionStudio/runtime/bin/python`，GPU 为 `$HOME/.local/share/vision-studio/gpu-runtime`。现有 `output/build/2.10.0-release`、`output/v2.10.0/qa-full` 与只读 DEB 树是旧版原件，不覆盖。

新开发目录示例（本段仅给命令，不表示已执行）：

```bash
cmake -S . -B output/build/mainwindow-refactor-dev -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64" \
  -DVISION_RESOURCE_FALLBACK=/opt/VisionStudio \
  -DVISION_STUDIO_TEST_PYTHON=/opt/VisionStudio/runtime/bin/python
cmake --build output/build/mainwindow-refactor-dev --parallel 2
python3 -B scripts/qa_detection_scope_220.py \
  --build-dir output/build/mainwindow-refactor-dev \
  --output-dir output/qa/mainwindow-refactor-stage-example \
  --python /opt/VisionStudio/runtime/bin/python \
  --test vision-ui-tests --test vision-scope-filter-ui-tests \
  --test vision-batch-responsiveness-tests
```

runner 要求新输出目录，并隔离 app data／XDG／YOLO／Matplotlib，保留原始 log 和 JUnit。它的默认集合只有七套，**不等于完整 26 套**；完整回归必须显式提供所有本版注册测试并核对发现清单。GPU 检查需显式 `--with-gpu --gpu-runtime "$HOME/.local/share/vision-studio/gpu-runtime"`，使用已准备环境，不下载。Windows 的 WebEngine GUI 测试使用 native windows QPA，不能简单套用 Linux offscreen。

旧 `output/v2.10.0/qa-tools/summarize_regressions.py` 和最终门禁可作为工具基线阅读，但其版本、输入路径及原件冻结规则不能直接复用成新版本通过；应在本轮 Git 外新目录适配后真实执行。构建仍需让新实现进入应用与 UI 测试共享 target，并保持资源注册、AUTOMOC 和运行路径回退。

## 第一版架构检查的范围

`tests/architecture_tests.py` 目前只建立三个稳定边界：core 不反向依赖 UI／Widgets／WebEngine；生产源码不引入 QtTest 或测试目录；生产实现不通过 `#include .cpp` 拼装。另盘点旧 smoke CLI 名称词面，防止搬文件时无意丢失入口。

它**不**限制 MainWindow 行数、当前 friend 数量、控制器数量、当前字段位置、未来模块文件名或具体 CMake target 拆分，不会提前判定尚未完成迁移的新结构不合格。后续禁止新增反向依赖的规则可逐阶段收紧，但必须先有明确模块权限设计与运行验证。
