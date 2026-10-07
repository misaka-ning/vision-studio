# 更新日志

所有后续版本的用户可见变更在这里记录，版本号采用 `主版本.次版本.修订版本`。下载文件和逐版验收材料见 [GitHub Releases](https://github.com/misaka-ning/vision-studio/releases)。

1.1.0–1.5.0 的条目根据已冻结的本地源码归档、发行说明和同版本验收报告于 **2026-10-07** 补录。下列历史日期是有日志证据的**本地交付／验收日期**，不代表首次上传 GitHub 的日期；公开发布时间以 GitHub Release 页面为准。归档文件保持原样，不把后来补写的文档冒充历史原文件。

## 未发布

暂无变更；后续新变更在此记录，发行时移入对应版本。

## [1.6.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.6.0) — 2026-10-07

### 新增

- 工作台增加「自动 · 优先 NVIDIA GPU」「CPU」「NVIDIA GPU」和 GPU 编号选择；自动模式说明不可用原因并回退 CPU，显式 GPU 失败时报告错误。
- ONNX GPU 使用 C++ ONNX Runtime 1.23.2 CUDA；PT 的 Ultralytics、兼容 YOLOv5 和 TorchScript 使用独立 PyTorch `2.9.1+cu128` 环境，CPU 继续使用原 OpenCV／PyTorch 路径。
- 「更多」增加 GPU 检查、准备、进度日志和取消。完整版本／官方 SHA256 锁一次配置到用户目录，实际 CUDA 自检后原子发布；未知目录不覆盖，取消／失败保留原环境，不修改驱动。
- 状态、导出与历史保存请求设备、实际设备、GPU 编号／名称和回退原因；切换设备保留灰度通道与左右目选择。
- 保留离线 CPU 安装包；可选 GPU 环境独立保存原始许可及第三方通知，CUDA 二进制不加入发行附件。README 展示实际 RTX 4060 CUDA 检测与本版模型结构图。

### 仓库与发布管理

- 建立 GitHub 源码与逐版 Release 管理，补录 1.1.0–1.5.0 的历史更新日志。
- 补充 GitHub README、贡献指南和版本管理规范，约定不可变标签、配套源码交付与远端核验后本地保留最新两版。
- 增加带草稿续传、远端 SHA256 与标签校验的发布工具、清理安全测试、只读 CI 和问题／PR 模板；启用主线及版本标签保护。
- 按维护者要求将每版 Release 整理为 DEB、完整源码包、`SHA256SUMS` 三个手动附件，保留 GitHub 自动生成的 Source code (zip) 与 Source code (tar.gz) 两个入口；同版 QA 和原始说明迁入仓库 `docs/releases/evidence/X.Y.Z`，保持冻结安装包、源码包及原始校验值不变。
- 补齐初次仓库发布的产品截图，并更新为 V1.6 的实际 NVIDIA GPU 检测与模型结构图。

历史 1.1.0–1.5.0 的冻结应用、安装包与原始校验保持不变；上述仓库维护代码和文档随 1.6.0 源码交付。

### 验证与交付

- 8 组 CTest 在 CPU／GPU 两阶段实际执行并通过：CPU 7 组 118.22 秒，GPU 3 组 62.52 秒；GPU 阶段 0 失败、0 跳过。配置安全与许可测试 21 项通过。
- RTX 4060／驱动 595.91.07 完成 78 项锁定依赖与候选／最终环境 CUDA 自检；GPU 后端 QtTest 报告 10 passed，覆盖实际 ONNX CUDA 节点、三类 PT、彩色／灰度 C1／C3、取消与恢复。
- GPU 右目灰度录像的 8 个已处理源帧全部编码且可播放；内容为 160 × 96，原左右拼接源为 320 × 96，元数据记录实际 CUDA 设备。
- 候选 DEB 已通过普通用户 CPU 推理、模型三视图缓存复用、断网安装／卸载，以及包内 ONNX／PT 的 CUDA／CPU 四轮启动器检查。正式包重新构建后，必须用其最终 SHA256 绑定安装、源码及 GPU 报告；最终证据保存在同版仓库目录。
- 继续提供 Ubuntu 22.04 amd64 DEB、完整对应源码包及 `SHA256SUMS` 三个手动附件；保留 GitHub 自动生成的两个源码入口。源码包含 GPU 配置脚本、版本／哈希锁、ORT C API 来源与 MIT 原通知，以及原有 15 份上游源码与构建输入。

依据：[本版发行说明](docs/releases/1.6.0.md) 和 [同版验收证据](docs/releases/evidence/1.6.0)。上述日期记录本版功能验收，公开发布时间以 GitHub Release 为准；物理摄像头仍需实际设备验证。

## [1.5.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.5.0) — 2026-10-07

### 新增

- 模型显示提供结构图、可展开层级树、参数表三种模式，共用模型库的全局选择和同一次解析缓存，切换视图或页面不重新加载模型。
- 层级树和参数表支持搜索与只读详情，可查看名称、类型、路径、张量形状和元素数；元素数使用精确整数排序。
- 普通 PT 模块展示包含关系，裸权重按参数名称分组；保留 TorchScript 计算节点及 ONNX 子图中的独立权重。
- 大模型显示达到深度、条数或元数据大小上限时给出截断提示，无法可靠计算的元素数显示为空缺标记。

### 修复

- 修复 Netron 嵌套对象的白底浅字问题，统一深色可读背景，保留不同层类型的颜色，并使用青绿色选中描边。
- 切换模型时清除旧搜索、旧内容与详情，保留所选展示模式。
- 避免超长元数据撑大界面，避免循环引用、重复共享张量与不可靠维度造成错误统计。

### 验证与交付

- 六组 CTest 全部通过，另有 11 个元数据边界场景通过；正式包完成实际 ONNX／PT 三模式、缓存复用、普通用户离线安装与卸载验收。
- 继续提供 Ubuntu 22.04 amd64 DEB、完整对应源码包和 SHA256 校验，安装目录为 `/opt/VisionStudio`，已有设置与运行记录保留。

依据：本版源码包中的 `docs/release/发行说明.md`、`CTest-1.5.0-2026-10-07.log`，以及 [同版验收证据](docs/releases/evidence/1.5.0) 中的 `ctest-qa.json`、`metadata-boundary-qa.json`、`qa-report.json` 和 `install-qa.json`。

## [1.4.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.4.0) — 2026-10-07

### 新增

- 增加内嵌 Netron 9.3.1 的「模型显示」页，通过 Qt WebEngine 查看本地模型结构。
- 模型库选择成为全局模型，同时决定检测工作台和模型显示使用的文件；选择后在后台异步预加载，进入页面直接使用缓存。
- 对不支持、损坏或没有可显示图的文件提供中文说明与重试；非 ONNX 文件显示导出 ONNX 的提示。
- 模型页面由本机回环地址提供，模型文件留在本机，结构查看不执行 `torch.load`。

### 交付

- 发行环境加入 Netron、Qt WebEngine 及其辅助进程与资源，完整源码包补充对应的 Qt 模块、Netron 和第三方通知。
- 保留灰度、左右目、录像与结果导出；完成模型图、HTTP 服务、安装及离线推理验收。

依据：本版源码包中的 `docs/release/发行说明.md`、`CTest-1.4.0-2026-10-07.log` 和 details 日志，以及 [同版验收证据](docs/releases/evidence/1.4.0) 中的 `qa-report.json`、`desktop-qa.json`、`install-qa.json` 和 `source-qa.json`。原本独立发行目录缺少 README，本次历史日志根据归档内说明补齐。

## [1.3.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.3.0) — 2026-10-06

### 新增

- 增加「录制视频」和末尾的「更多」页；运行示例和手动导出从工作台移至「更多」。
- 视频与摄像头检测支持开始／结束录制，保存当前所选单目或完整画面、颜色模式及检测标注，输出 MJPEG AVI 和同名 JSON。
- 录制视频页支持列表、预览、播放、导出和打开存储目录；停止检测、视频结束及正常关闭时完成录像保存。

### 修复

- 灰度模式的预览与 PNG 底图显示实际灰度；切回彩色恢复原色并清除旧预测。
- 补齐 16 位 PT 灰度输入、奇数尺寸录像、录制帧率与停止过程中有效尾帧等处理。

### 验证

- 四组 CTest 通过，包含真实推理、六页导航、录像启停、实际解码、播放与导出；正式包完成离线安装与卸载验收。

依据：本版原始发行说明、源码包中的说明，以及 [同版验收证据](docs/releases/evidence/1.3.0) 中的验收报告和 `CTest-2026-10-06.log`。

## [1.2.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.2.0) — 2026-10-06

### 新增

- 支持彩色 RGB／BGR 与灰度输入；灰度按模型的固定输入通道适配：1 通道直接输入灰度，3 通道复制同一灰度信息。
- ONNX、兼容的 Ultralytics／YOLOv5 PT 和具有必要元数据的 TorchScript 路径支持 1／3 通道灰度模型。
- 单设备水平左右拼接摄像头及视频可选择完整画面、左目或右目，检测和导出坐标相对所选画面，JSON 记录原始帧尺寸。
- 保存颜色与画面选择设置，兼容已有默认彩色／完整画面配置；图片队列继续完整处理。

### 验证

- 三组 CTest 通过，覆盖真实灰度像素、单／三通道模型、合成左右目视频、坐标和设置迁移；正式包完成离线推理与安装验收。

依据：本版原始发行说明、源码包中的说明，以及 [同版验收证据](docs/releases/evidence/1.2.0) 中的验收报告和 `CTest-2026-10-06.log`。

## [1.1.0](https://github.com/misaka-ning/vision-studio/releases/tag/v1.1.0) — 2026-10-05

这是当前保留下来的最早完整发行包；不据此推断不存在更早的开发版本。

### 已提供功能

- C++17／Qt 6.8.3 桌面工作台，支持图片、批量图片、视频和摄像头，目标检测与单标签分类。
- C++／OpenCV DNN CPU 执行 ONNX；常驻本地 Python／PyTorch CPU 后端执行兼容的 Ultralytics、YOLOv5 和 TorchScript PT 模型。
- 模型库、运行记录、检测框与类别显示、缩放和平移、PNG／JSON／CSV 导出，以及可取消的后台推理。
- Ubuntu 22.04 amd64 系统级 DEB，安装到 `/opt/VisionStudio`，随包提供独立 CPU runtime、示例模型、中文文档、许可及完整对应源码。

### 验证

- 三组 CTest、真实 YOLO 推理与导出、普通用户离线安装／运行／卸载验收通过，用户数据和宿主系统状态保留。

依据：本版源码包中的 `docs/release/发行说明.md`、`CTest-2026-10-05.log`，以及 [同版验收证据](docs/releases/evidence/1.1.0) 中的 `qa-report.json` 和 `install-qa.json`。
