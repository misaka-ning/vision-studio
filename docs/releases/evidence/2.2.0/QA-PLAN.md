# Vision Studio 2.2.0 验证计划与可复用发行输入

本文件是验证计划，**不是通过报告**。测试结果以本目录后续实际执行生成的报告、原始日志及最终文件 SHA256 为准。未执行项不视为通过；2.1.0 的旧记录不充当 2.2.0 验收。

## 只读输入调查

已于 2026-10-19 在 Ubuntu 22.04.5 LTS amd64 调查原发行材料。未修改用户设置、模型、录像、系统安装、GPU 环境或旧发行文件。

| 输入 | 本机位置 | 已核实范围 |
| --- | --- | --- |
| Qt SDK | `/home/misaka/Qt/6.8.3/gcc_64` | 6.8.3，含 WebEngine、WebChannel 和 QtWebEngineProcess |
| QtImageFormats 补充插件 | `/home/misaka/文档/ChatGPT/yolo/build/qt-imageformats-prefix/plugins` | 原始 icns、tga、tiff、wbmp、webp 插件仍在 |
| CPU runtime | `/opt/VisionStudio/runtime` | 已安装 2.1.0 环境，Python 3.10、59 项发行依赖，Torch 2.9.1+cpu、Netron 9.3.1、ONNX 1.17.0、protobuf 6.33.0 |
| 原始上游归档与来源 | `/home/misaka/文档/ChatGPT/yolo/output/releases/2.1.0/sources` | 18 个文件、794,239,087 bytes；逐份重新计算 SHA256，全部与原 2.1.0 SOURCE-INVENTORY 一致 |
| 当前已准备 GPU runtime | `/home/misaka/.local/share/vision-studio/gpu-runtime` | ready.json 存在；此输入调查未执行 CUDA，不据此宣称 GPU 通过 |
| 原发行回退副本 | `/home/misaka/文档/ChatGPT/yolo/output/releases/2.1.0` 及相邻 `2.0.0` | 原包和完整对应源码仍在，保持只读，不覆盖 |

逐份归档实测哈希及 CPU 元数据见 [input-reuse-proof.json](input-reuse-proof.json)。这些值仅证明旧发行输入可复用，**不是 2.2.0 新安装包校验**。无须重新下载这 18 份归档，也不能把旧应用源码归档作为 2.2.0 应用源码。源码快照必须从当前本次提交的 Git 跟踪文件与明确授权输入重新整理，不带其他任务文件、训练输出或用户数据。

## 新功能验收矩阵

### 文件夹范围

新集成 target 为 `vision-scope-filter-ui-tests`，在真实 Qt 界面与真实 ONNX forward 上验证。预期数据行包括：

- 整个文件夹／从当前向后／仅当前，每种分别选择首张、中间、末张；整个文件夹始终从索引 0 开始。
- 单张图片三种范围均只执行一次；61 张队列三种范围校验范围、顺序、进度分母、完成数量，无 50 张上限。
- 后两种没有选中项时禁止开始并显示行内提示；不会隐式扩大范围。视频／摄像头隐藏范围控件。
- 开始后请求中固定文件列表和顺序，禁用范围及相关队列修改，检测完成后恢复；允许回看时不能改变已固定请求。
- 范围外未处理图片不标记完成；相同配置的已完成缓存保留，新结果逐张完成后原子更新，取消未完成的新计算不抹掉仍有效的旧结果。进度只计算本次范围，变化模型、设备、阈值、预处理或类别条件后不能显示失效结果。
- 中间范围停止后不再调度下一图，已完成图片能回看标注、目标表与指标；手动导出 source 与当前选图一致。
- 高分辨率随机图自动保存期间停止：心跳、停止派发耗时和已接受数量使用原 `vision-batch-responsiveness-tests` 实测，不把 CPU forward 当作可强杀操作。
- 自动保存失败、图片读失败、任务取消后再次启动与过期确认信号继续由现有批量交付回归覆盖。

UI 集成测试将输出 `detection_scope_evidence:` JSON 行；报告须保留实际数据行和文件顺序，不仅记录请求数量。若最终函数名或覆盖行改变，按实际源码与日志核对，不以计划文字补填。

最终 UI 测试函数为 `fixedRangeAndProgress`（15 行，包含 mode 0～2 与首／中／末、单图、61 张）、`filtersAndScope`（12 行，3 范围 × 4 筛选输入）、`selectionRequiredAndStreamingVisibility`、`invalidFilterAndModelChange`、`validCacheOutsideRangeAndChangedParameters`、`stoppedRangePreservesCompletedResultsAndExport`、`equivalentFiltersAndChangedSourceFile`、`externalModelReplacementDuringTaskCannotRelabelCache`、`submittedSourceIdentityIsNotReplacedByLaterDiskVersion`、`workbenchScopeFilterVisualArtifact`。最后一项在 64 图队列选择第 51 张、仅当前、类别 0 与 5，实际执行 CPU 推理并向独立 artifact 目录保存工作台截图；原文件／模型在任务中被外部替换的边界由前两项验证。这份清单不预设运行结果。

### 类别筛选

- 空值全部、32、多个 ID、中文逗号／空格、去重；与三种范围组合。
- 负数、非整数、分隔符与去重等语法边界及越界 ID；行内错误与禁止启动。切换到类别数不同的模型后重新校验，不固定为 80 类。
- 原始 class_id 保留，例如 32 仍是 32。类名标签映射不改变筛选语义。
- 0 匹配仍是有效已完成结果，目标数 0、预览、列表、自动保存、手动导出、录制统一为空。
- 分类禁用筛选并解释原因；检测和分类切换不误用旧缓存。
- 实际 CPU ONNX、CUDA ONNX、现代 PT、旧版 PT 及标准 TorchScript 对照同一条件；比较类别集合与目标输出，容差沿用已有后端一致性检查。
- 录制消费最终过滤后的 InferenceResult；通过实际视频帧与导出结果检查，不能只证明画框隐藏。
- 使用真实公开 YOLO 全模型验证常规输出；小型真实 ONNX／TorchScript 网络用于可重复的多类及 32 非空结果。生成图的固定输出只用于功能路径验证，不当作真实场景识别能力或性能结果。

COCO 的 ID 32 是 `sports ball`，覆盖多种运动球。筛选它不等同于模型专门识别足球，也不扩展模型已训练的类别。

core／PT／GPU 新增测试名如下，列出的是拟运行代码，实际通过数和设备以执行日志为准：

- `vision-core-tests`：`classFilterParsingAndValidation`、`classFilterUsesActualOnnxClassCountAndOriginalIds`、`classFilter32IsNotRenumberedAndZeroMatchSucceeds`、`realOnnxClassFilterMatchesUnfilteredSubset`、`classificationRejectsDetectionClassFilter`。
- `vision-pt-tests`：`realPtClassFilterMatchesOriginalIds` 的 modern-yolov8／legacy-yolov5 数据行、`torchScriptClassFilterUsesModelMetadataAndRejectsClassification`、`torchScriptClass32RemainsOriginalId`、`modelInspectionIgnoresUserLabelsAndFiltering`。33 类 TorchScript 用于真正命中 ID 32 并核对其不重编号，不仅验证零匹配。
- `vision-gpu-tests`：`classFilterCpuAndCudaKeepIdenticalOriginalIds`，覆盖 ONNX、现代 PT、旧版 PT、真实 TorchScript 的 CPU／CUDA 实际输出对照。
- `vision-recording-tests`：`realClassFilteredResultIsRecordedWithOriginalIdMetadata`，官方 YOLOv5n 与 bus 图片筛选原 ID 5，实际录制并完整解码 3 帧，核对 JSON 的 `class_filter=[5]` 与 `model_class_count=80`。

## 可重复执行与证据边界

统一开发构建目录为 `/tmp/vision-studio-2.2.0-build`。新增 runner 不联网、不安装依赖，要求新输出目录，保存每个真实 CTest 的完整日志和 JUnit、应用二进制 SHA256。Qt 内部的 skipped／blacklisted 不会因 CTest 进程返回 0 被写成全部通过。`VISION_STUDIO_QA_ARTIFACTS` 设置到该新输出目录下的 `artifacts/`，保留原始 ORT profile；新 GPU 测试输出的 `CLASS_FILTER_EVIDENCE` 和 `ORT_PROFILE_EVIDENCE` 与 UI 的 JSON 证据同时进入报告，已有文件不覆盖。

Python unittest 的 `OK (skipped=1)` 也会返回进程 0，而 CTest 的套件级 JUnit 可能没有 `<skipped>`；runner 因此还读取原始日志中的正数 `skipped=N`、`... skipped` 用例行与正数 `N skipped` 摘要。`skipped=0` 与 Qt 的 `0 skipped` 不会误判。原始只含报告的跳过记录不会被改成通过，补齐实际缺失条件后的复验须另存新目录并保留初次日志。

```bash
# CPU/集成套件；具体选择按最终测试名核对。
# 原 recording suite 含真实 GPU 行，因此若不提供 GPU，整套的 skip 会明确使严格 runner 不通过。
python3 scripts/qa_detection_scope_220.py \
  --build-dir /tmp/vision-studio-2.2.0-build \
  --output-dir /tmp/vision-studio-2.2.0-regression-cpu \
  --python /opt/VisionStudio/runtime/bin/python \
  --test vision-scope-filter-ui-tests --test vision-core-tests \
  --test vision-inference-delivery-tests --test vision-pt-tests \
  --test vision-ui-tests --test vision-batch-responsiveness-tests

# 使用已存在的 GPU 环境；不执行 setup，不改驱动或用户 runtime。
python3 scripts/qa_detection_scope_220.py \
  --build-dir /tmp/vision-studio-2.2.0-build \
  --output-dir /tmp/vision-studio-2.2.0-regression-gpu \
  --python /opt/VisionStudio/runtime/bin/python \
  --with-gpu --gpu-runtime /home/misaka/.local/share/vision-studio/gpu-runtime
```

runner 报告仅表示所选测试的实际执行，不替代完整 CTest、安装树验证、包绑定 CUDA 证据或源码归档验收。因沙箱硬件、子进程或 Chromium 命名空间限制未执行的项目，须保留实际错误并在可用权限环境重试，不能静默禁用测试后宣称通过。

CPU Python 参数保留虚拟环境入口的绝对路径，**不解析解释器软链接**。`/opt/VisionStudio/runtime/bin/python` 的目标是系统解释器，但通过该入口运行才能识别相邻的 `pyvenv.cfg` 与本版 site-packages；把它 `resolve()` 成 `/usr/bin/python3.10` 会丢失 CPU runtime。2026-10-19 已只读执行核对该入口的 `sys.executable`、`sys.prefix` 与 Torch 模块位置，分别为 `/opt/VisionStudio/runtime/bin/python`、`/opt/VisionStudio/runtime` 与其 site-packages 下的 Torch。该检查仅验证解释器选择，不表示模型测试通过。

## 最终发行构建顺序（待执行）

1. 完成源码及版本／文档变更与全部适用测试。整理干净输入树，将实际 `/opt/VisionStudio/runtime-info.json` 作为本版明确输入，并核对现有许可快照与 59 项运行依赖一致。保持生产资源 fallback 为 `/opt/VisionStudio`。将旧 2.1.0 `sources/` 原归档及 DOWNLOAD-PROVENANCE **复制**到新 `output/releases/2.2.0/sources/`，不编辑旧数据；先运行新工具 `build_source_archive.py --inventory-only`，将新版本 SOURCE-INVENTORY 与发行锁更新到干净源码树，核对后再冻结打包输入。这一步不需要 DEB，可避免包内继续携带旧版清单。
2. 使用原始 Qt SDK、补充插件及已部署 CPU runtime；只修改新 staging 内副本。下列 `CLEAN_SOURCE` 指新的本次干净源码绝对路径，须由发布负责人赋值；示例不是已执行记录。

```bash
python3 packaging/build_deb.py \
  --source-root "$CLEAN_SOURCE" \
  --binary /tmp/vision-studio-2.2.0-build/bin/vision-studio \
  --qt-prefix /home/misaka/Qt/6.8.3/gcc_64 \
  --extra-qt-plugins /home/misaka/文档/ChatGPT/yolo/build/qt-imageformats-prefix/plugins \
  --runtime /opt/VisionStudio/runtime \
  --stage /tmp/vision-studio-2.2.0-deb-stage \
  --output "$PWD/output/releases/2.2.0/vision-studio_2.2.0-1_amd64.deb"
```

3. DEB 生成后，运行新源码的 `build_source_archive.py` 生成新的应用源码、完整对应源码和两个层级清单。它会重新校验全部原始归档并重写同版 inventory／锁，核对仍与先前冻结输入一致。任何影响已打包输入的变化都须重建包及其证据，不在旧发行目录使用 `--overwrite`。
4. 对最终文件执行：

```bash
python3 packaging/verify_deb.py output/releases/2.2.0/vision-studio_2.2.0-1_amd64.deb \
  --smoke --apt-simulate --source-dir output/releases/2.2.0 \
  --report output/releases/2.2.0/qa-report.json --keep-workdir
python3 packaging/verify_install_namespace.py \
  output/releases/2.2.0/vision-studio_2.2.0-1_amd64.deb \
  --report output/releases/2.2.0/install-qa.json --keep-workdir
python3 packaging/verify_gpu_deb.py \
  output/releases/2.2.0/vision-studio_2.2.0-1_amd64.deb \
  --runtime-dir /home/misaka/.local/share/vision-studio/gpu-runtime \
  --report output/releases/2.2.0/packaged-gpu-qa.json --keep-workdir
```

5. 原 verifiers 验证既有推理、结构图／树／表、转换与只读系统安装树。新范围／过滤另需在该新包解压树／安装树中实测，不用旧应用的演示输出充当证据。通过报告绑定新 DEB SHA256，完整对应源码另做成员哈希与 Git 冻结源码比对。
6. `scripts/github_release.py` 要求 qa-report、install-qa、ctest-qa、gpu-qa 和 source-bundle-qa 实际成功。GPU 聚合中每项 `onnx_cuda_nodes`、`pt_cuda`、`packaged_onnx_cuda`、`packaged_pt_cuda`、`cpu_regression`、`grayscale_c1_c3`、`stereo_recording` 都须有本版真实证据，不能复制旧报告只改版本号。保留完整日志到本目录并随源码提交。
7. 发布负责人统一提交、推送 main 和带说明的 v2.2.0 标签；本次只发布 DEB、complete-source、SHA256SUMS 三个手动附件。所有文件必须小于 2 GiB，远端核验状态、大小和 SHA256 后再执行精确目录的保留两版清理。本版未请求制作 Windows 安装包，不覆盖现有 2.1.0 Windows 附件或标签。

源码归档校验和最终包验收加入仓库时，区分归档冻结后的 QA 补充文件，避免递归自校验。已发表 2.1.0／2.0.0 的应用、标签、资产和原始证据均保持不变。

## 发布门禁的准确报告结构

以下是对现有 `scripts/github_release.py::validate_bundle` 的只读审计结果，用于避免错误复用或改名报告；不降低 AGENTS 的真实验收要求。

| 报告 | 工具直接核验的字段 | 额外必须保留的实际依据 |
| --- | --- | --- |
| `qa-report.json` | `success=true`、`errors` 空、`details.deb_sha256` 等于新 DEB | 新包结构、依赖、只读运行、APT 模拟、真实模型与结构页 |
| `install-qa.json` | 同上 | 隔离 DPKG 安装／启动／转换／卸载；宿主数据库与用户资料保持不变 |
| `ctest-qa.json` | `success=true`、`errors` 空；若有 `version` 必须为 2.2.0 | 本版最终构建的真实完整测试日志、实际通过／失败／跳过数与测试二进制身份 |
| `gpu-qa.json` | 同上，且 `schema_version=1`、`details.deb_sha256`、7 个 `checks` 全 true | 本版 CUDA 节点 trace、PT CUDA、新包 ONNX／PT CUDA、CPU、灰度 C1／C3、左右目及录制；每项都有来源 |
| `source-bundle-qa.json`（或 `source-qa.json`） | `success=true`、`errors` 空；若有 `version` 必须为 2.2.0 | 完整包与拆分包 SHA、18 个上游归档 SHA、应用成员与干净冻结源码逐文件比对、外来文件排除、包内 helper 身份 |

`verify_gpu_deb.py` 的原始报告是 **`schema: 1`、顶层 `deb_sha256`、3 项 checks**，因此只保存为 `packaged-gpu-qa.json`，不能直接改名为 `gpu-qa.json`；必须在保留原结果的基础上真实聚合上述 7 项来源。新版范围／筛选证据另归档，不能仅因旧 verifier 的通用 smoke 通过就宣称新功能包验收完成。

所有存在的可选 `desktop-qa.json`、`metadata-boundary-qa.json`、`home-qa.json`、`source-qa.json` 等也会被门禁检查，故不复制 2.1.0 的报告到 2.2.0 目录。发布目录只从旧版复用原始 `sources/`，所有新 QA 按实际执行生成。

`SHA256SUMS` 恰好列出新 DEB 与新 complete-source；`SOURCE-SHA256SUMS` 必须包含应用拆分归档、inventory、DOWNLOAD-PROVENANCE 与 18 个上游文件且哈希真实一致。三个公开附件都需小于 2 GiB；当前 Linux `validate_bundle` 尚未自动拦截该大小，必须在发布前独立检查实际字节（Windows 补充发行已有大小检查，不能混为 Linux 已检查）。发布还要求本地标签的提交等于 HEAD、已跟踪工作树干净，远端同名标签提交一致；草稿恢复只能对同内容使用 `--resume-draft`，不覆盖已公开资产。

## 发布成功后的只读清理候选调查

以下仅列候选，**尚未删除、复制或远端核验**。调查仅限原项目明确版本目录。2.2.0 完整发布和远端校验前，继续保留现有 2.1.0 与 2.0.0。

| 路径（均以 `/home/misaka/文档/ChatGPT/yolo/output/` 为前缀） | 当前内容／体积 | 后续处理 |
| --- | --- | --- |
| `releases/2.0.0` | 原正式版，独立 inode 文件逻辑总计 2,001,391,134 bytes，31 个文件全部具有外部硬链接 | 2.2.0 与 2.1.0 远端核验成功后，可由发布工具预览；单独删除此副本不会释放这 31 个文件的数据块，不能把逻辑体积当成释放空间 |
| `deb-stage/2.0.0-1` | release-info 明确 `2.0.0-1`，约 1.66 GB 文件逻辑大小，约 1.74 GB 数据块 | 明确版本的旧 staging 候选；含预期 Python／Qt 软链接，不能交给只接受无软链接整版目录的 `prune`，须另行按精确路径核验后处理 |
| `v2.0-stable/source-inputs` | CMake 明确 2.0.0，约 87.90 MB | 冻结源码副本候选，先与原源码归档核对并保存必要未归档记录 |
| `v2.0-stable/package-source-inputs` | CMake 明确 2.0.0，约 87.88 MB | 同上 |
| `v2.0-stable/review/delivery-build` | CMakeCache 明确 2.0.0，约 6.75 MB | 明确旧版本的独立验证构建候选 |
| `releases/2.1.0` | 原正式版与上游输入，独立 inode 文件逻辑总计 2,063,459,335 bytes | 保留，作为最新两版之一与源码复用来源 |
| `v2.1.0`、`v2.1.0-conversion` | 2.1.0 源码／构建／原始验证材料 | 本次保留，不因目录名字近似而批量删除 |

`v2.0-stable` 整树还含 generated QA 的 preferences、history 与未统一归档的诊断材料，虽然属于旧测试输出，也不因版本前缀就整树删除。发布工具的整版清理限制拒绝受保护名字和软链接；源码／staging 清理只能在必要证据保存与远端核验后，对上述精确路径另行评估。

存在跨目录硬链接，`du` 对多个参数的计数会去重，故本节使用逐目录独立 inode 统计，并明确区分逻辑字节、磁盘数据块与真正可释放空间。当前工作树、Git、Qt SDK、已装 `/opt/VisionStudio`、GPU runtime、gpu-wheelhouse、用户模型、设置和录像全部不在清理候选内。

若新 2.2.0 发行放在当前工作树的 `output/releases/2.2.0`，而旧两版在文档项目的另一个 releases 根，**不能在旧根直接 `prune --execute` 期待它识别外部 2.2.0**：它只按该根中存在的版本排序，当前仍会保护 2.1.0 和 2.0.0。需先按规范在同一个正式发行根整理准确的新版本副本并重新核验，或对旧副本执行单独、明确版本的核验与清理流程；不移动用户工作树或修改工具门禁。
