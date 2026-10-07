# Vision Studio

使用本机 **C++17 / Qt 6.8.3** 构建的桌面视觉推理工作台。ONNX 由 C++ / OpenCV DNN 执行，`.pt` 由常驻的本地 PyTorch 进程执行，均使用 CPU。支持目标检测、单标签图像分类、图片批量处理、视频与摄像头输入，并通过内嵌 Qt WebEngine 查看 Netron 模型结构。

## 启动

正式发行包面向 **Ubuntu 22.04 amd64**，包含 Qt 6.8.3 / WebEngine、独立 CPU PyTorch 环境、Netron 9.3.1 和本地示例模型。安装后从应用菜单启动 **Vision Studio**，也可运行 `vision-studio`：

```bash
sudo apt install ./vision-studio_1.5.0-1_amd64.deb
vision-studio
```

程序安装到 `/opt/VisionStudio`。设置、运行记录和默认导出写入 `${XDG_DATA_HOME:-$HOME/.local/share}/vision-studio`，普通用户无需向 `/opt` 写入文件。发行版 Python 环境使用无 GUI 的 `opencv-python-headless 4.11.0.86`，原生 ONNX 后端仍使用系统 OpenCV 4.5.4。安装无需下载 Python 模型依赖；APT 可能需要安装系统共享库。

家目录中的便携版可通过 `VisionStudio`、`VisionStudio.desktop` 或 `run.sh` 启动。首次使用可打开 **更多 → 运行示例**，查看真实 YOLO 模型的推理结果，再导入自己的模型。正式 DEB 与早期 `scripts/package.sh` 生成的便携目录不同：DEB 已携带 CPU runtime，便携打包脚本需要另行安装该环境。

发行说明见 [发行说明](docs/release/发行说明.md)。发布时将 DEB 与 `vision-studio-1.5.0-complete-source.tar.xz` 一同提供，并附校验这两个主文件的 `SHA256SUMS`。完整源码包包含本项目源码归档、`sources/` 下的上游源码和校验内部文件的 `SOURCE-SHA256SUMS`。重建方法和 Qt 替换接口见 [对应源码与重建](docs/release/对应源码与重建.md)。

完整操作说明见 [使用指南](docs/使用指南.md)，界面设计说明见 [UI 设计](docs/UI设计.md)，早期版本的验证记录见 [验证报告](docs/验证报告.md)。本版结果以发行目录内的验收报告与对应 CTest 日志为准。

![工作台真实 YOLO 检测](docs/preview.png)

## 功能

- **推理工作台**：模型和标签配置、图片队列、视频与摄像头输入、置信度与 NMS 阈值调节。
- **视觉结果**：检测框、类别和置信度叠加，缩放、平移、适应视图，目标表与性能数据。
- **模型库**：导入本地 `.onnx` 或 `.pt`；现代 Ultralytics YOLO `.pt` 自动读取类别名称，配置保存到用户设置。
- **模型显示**：结构图、层级树、参数表三种视图共用全局模型和一次解析缓存；可搜索并检查详情，深色界面覆盖嵌套对象和参数内容。
- **结果导出**：保存标注 PNG、结构化 JSON 和 CSV；图片批量任务可逐张自动导出，视频与摄像头任务在结束后保存最后一帧。
- **运行记录**：查看处理来源、模型、目标数量、推理耗时和执行时间。
- **后台任务**：推理在工作线程中运行，可中止任务，界面保持可操作。
- **灰度输入**：工作台可选择彩色或灰度；ONNX 自动识别固定的 1 / 3 输入通道，单通道直接输入灰度，三通道复制同一灰度值。
- **左右拼接输入**：单设备摄像头和视频可选择完整画面、左目或右目；推理及导出使用所选画面，图片队列保持完整。
- **视频录制**：推理视频或摄像头时可开始 / 结束录制，保存所选画面、实际颜色模式与检测标注；录制视频页可浏览和播放本地录像。
- **更多**：集中提供运行示例和手动导出结果，保留原有快捷键。

七页导航依次为：检测工作台、模型库、模型显示、运行记录、使用指南、录制视频、更多。

## 模型显示

在「模型库」选择模型后，它同时用于工作台检测与模型显示。应用在后台异步预加载结构；完成后进入「模型显示」直接查看缓存的图，切换页面不重复加载。预加载尚未完成时页面显示当前加载状态。模型库可前往检测或模型显示，页面导航本身不切换模型。

1.5.0 提供三种互补视图：**结构图**检查节点与连接，**层级树**查看可解析的模块包含关系或权重分组，**参数表**集中查看参数和张量信息。三种模式使用同一次解析结果，切换不重新读取模型；可搜索内容并查看详情。嵌套对象、张量和详情面板沿用深色背景与清晰文字，修复浅色底与浅色字叠加的问题。

结构图使用 Netron 内置查找；层级树可搜索名称、类型和路径，参数表可搜索路径、dtype、shape、元素数与备注。表中包含文件内的权重、常量和缓冲，不能当作可训练参数总量或运行时激活统计。大型模型达到显示限额时会提示截断；选择新模型会清除旧搜索和详情，并保留当前展示模式。

Netron 9.3.1 通过随包 Python 环境在本机回环地址的随机端口提供页面，由 Qt WebEngine 内嵌显示；模型文件留在本机，不上传远程服务，查看结构不执行 `torch.load`。ONNX 展示文件中声明的运算图；普通 `.pt` 只展示能解析的模块包含关系、权重分组及静态信息，不据此补造 forward 连接。模型是否能被 Netron 解析，与它能否用于本应用推理分别判定。格式不识别、文件损坏或没有可显示结构时，界面显示中文说明并提供重试入口。

`.pt` 等文件可能只保存权重、模块或部分静态信息，完整结构通常需要导出 ONNX。选择任何非 ONNX 文件时，页面显示以下提示：

想看完整结构，建使用导出的 ONNX。

## 模型兼容范围

ONNX 后端使用 **OpenCV 4.5.4 DNN / CPU**，支持能被该版本 OpenCV 导入且符合下列输出约定的模型。

内置可直接运行的 YOLOv5 Nano 示例模型（固定 640 输入），在「更多」点击「运行示例」即可体验。实际用户模型支持情况以执行结果为准。

| 任务选择 | 预期主要输出 | 处理方式 |
| --- | --- | --- |
| YOLOv5 检测 | `[1, N, 5 + 类别数]` | `cx, cy, w, h, objectness, class scores`，置信度为 objectness × class score |
| YOLOv8 / YOLO11 检测 | `[1, 4 + 类别数, N]` | `cx, cy, w, h, class scores` |
| 图像分类 | `[1, 类别数]` 或 `[1, 类别数, 1, 1]` | 概率直接使用；logits 转为 softmax 概率，显示 top-5 |

检测采用保持比例的 letterbox 预处理，输出坐标映射回原图后进行裁剪和按类别 NMS，每张图最多保留 300 个目标。自定义检测模型须提供与输出类别数一致的标签；默认标签为 COCO 80 类。分类输入缩放到正方形；颜色通道、缩放系数和 RGB 均值可配置。输入宽高使用相同的「输入尺寸」值，需与模型匹配。旧版 YOLOv5 附带的三个辅助检测头可与主要输出一起存在。

ONNX 图像输入需为 NCHW，通道数固定为 1 或 3。**彩色模式**要求 3 通道，并使用原有 RGB / BGR 设置；**灰度模式**按亮度转换，1 通道模型接收单通道，3 通道模型接收三份相同灰度。灰度使用统一均值，G / B 均值不参与，检测补边仍为 114。灰度预览和 PNG 标注底图也显示实际灰度；切回彩色恢复原色并清除旧预测，需要重新运行。固定 4 通道、动态通道、非 NCHW 四维图像输入及多个运行时输入目前不支持；形状为动态宽高的模型仍需满足 OpenCV 兼容要求。

1.5.0 保留旧版本默认彩色设置；已有 3 通道模型无需更改。载入 1 通道模型时须明确选择灰度，程序不会静默改变输入模式。灰度转换必须与训练预处理一致，把普通 RGB 模型切为灰度也可能影响识别结果。

对单设备输出的水平左右拼接视频，在摄像头或视频输入中选择「完整画面」「左目」或「右目」。完整画面使用原帧；左目取前 `floor(宽度 / 2)` 列，右目取其余列，奇数宽度时右目多一列。摄像头保留设备默认分辨率。检测坐标和 PNG 导出相对所选画面，JSON 同时记录原始帧尺寸。图片输入继续完整处理，不受所选左右目影响。

**暂不支持**分割、姿态、旋转框、多标签分类、带内置 NMS 的检测输出、多输入模型及量化模型的专用预处理。模型导入成功不代表所有算子都能执行；运行失败会在界面显示错误信息。CUDA / TensorRT 推理后端尚未提供。

### 直接运行 .pt

支持现代 **Ultralytics YOLOv8 / YOLO11 的完整 `.pt` 检查点**，彩色模式使用模型原生预处理和后处理。应用直接载入 `.pt`，每个任务保留一个后台 Python 进程和模型实例，连续处理图片或视频帧。

检测任务支持输入尺寸、置信度与 NMS IoU，输入尺寸须为 32 的倍数；类别名称从检查点读取。自行导入标签可覆盖名称，数量必须与模型类别数匹配。RGB / BGR、缩放系数和均值属于 ONNX 预处理配置，`.pt` 后端使用模型原生配置。

灰度 `.pt` 支持兼容模型的 1 / 3 通道输入，按实际输入层识别通道；TorchScript 无法可靠识别时需在归档内的 `config.txt` JSON 元数据中提供 `input_channels`。检测使用灰度转换、114 补边和 `1 / 255`；灰度分类使用 Ultralytics 分类图像变换，明确设定所有通道 `mean=0`、`std=1`，再保留模型所需的通道。手工均值保持零，彩色模式仍要求三通道。

同时支持与随附 `vendor/yolov5` v7.0 框架兼容的完整 YOLOv5 检查点，内置 `models/yolov5n.pt` 可直接运行。带任务和类别元数据的 TorchScript `.pt` 支持标准 YOLO 原始检测输出或单标签分类输出。

只有 `state_dict` 的 `.pt` 文件还需要网络架构，当前版本不能仅凭权重字典推理。自定义模型类仍需要原项目代码；分割、姿态和旋转框 `.pt` 暂不支持。

直接 `.pt` 推理使用交付目录的独立 `runtime/bin/python`：Python 3.10.12、PyTorch 2.9.1+cpu、Torchvision 0.24.1+cpu、Ultralytics 8.4.173。运行环境信息见 `runtime-info.json`，依赖记录见 `requirements-pt.txt`。`VISION_STUDIO_PYTHON` 可指定另一个具有这些依赖的解释器绝对路径。模型和依赖准备完成后，推理可离线执行。检查点载入可能通过 pickle 执行 Python 代码，请使用自己训练或来自可信来源的 `.pt` 文件。

## 从源代码构建

依赖：CMake ≥ 3.21、C++17 编译器、Qt 6.8.3 Core / Gui / Widgets / Svg / Network / WebChannel / WebEngineWidgets / Test、OpenCV Core / Imgproc / Imgcodecs / Videoio / DNN。Qt WebEngine 同时需要 SDK 中兼容的 Quick / Qml / Positioning 模块及辅助进程、资源和语言文件。

直接 `.pt` 后端还需要本地 Python、CPU PyTorch、Torchvision 和 Ultralytics；模型显示需要同一环境中的 Netron 9.3.1。它们在独立 runtime 中，与 C++ 编译过程分开。正式 DEB 已捆绑这些依赖，源码与便携目录需要自行准备；实际发行锁为 `docs/release/requirements-release.lock.txt`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

也可以运行 `./scripts/build.sh` 一次完成构建与检查。`./scripts/package.sh /绝对路径/新的目录` 可生成包含源码、模型和可执行文件的交付目录；不会携带原工程的 CMake 缓存、偏好设置或临时测试目录。`docs` 保留历史验证报告与测试日志；当前版本结果以本版实际验收为准。

本机主文件夹中的安装已准备 `.pt` runtime。单独生成的交付包不携带约 1.5 GB 的 Python 环境；需要直接运行 `.pt` 时，准备 Python 3.10 或 3.11，在该交付目录执行 `./scripts/setup_pt.sh` 安装独立 CPU runtime。安装按 `requirements-pt.lock.txt` 约束依赖版本，需要下载依赖；完成后图片、视频和模型推理均可离线执行。

GUI 自检使用本地示例模型执行推理并导出结果，`--smoke` 使用 ONNX，`--smoke-pt` 使用 PyTorch 检查点。`--smoke-model` 等待 Netron 真正解析图节点，依次保存结构图 `model-display.png`、层级树 `model-hierarchy.png`、参数表 `model-parameters.png` 与 `model-display-report.json`：

```bash
./build/bin/vision-studio --smoke /tmp/vision-studio-smoke
./build/bin/vision-studio --smoke-pt /tmp/vision-studio-pt-smoke
./build/bin/vision-studio --smoke-model /tmp/vision-studio-model-smoke
./build/bin/vision-studio --smoke-model /tmp/vision-studio-pt-model-smoke \
  --display-model "$PWD/models/yolov8n.pt"
./build/bin/vision-studio --screenshot /tmp/vision-studio.png
```

模型显示报告记录 `nodes`、`hierarchy_items`、`parameter_rows`、`display_modes` 和 `cache_preserved`。验收需检查三种模式的真实内容与同服务、同浏览器缓存复用，不能仅以页面打开或截图文件存在作为成功。

无显示器环境可设置 `QT_QPA_PLATFORM=offscreen`，WebEngine 自检可另用 `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu` 选择软件渲染。正常启动器保留 Chromium sandbox。桌面环境应直接启动应用以获得正常字体和窗口尺寸。

## 验证方法

`tests/core_tests.cpp` 在临时目录生成小型 ONNX 网络，通过真实 OpenCV DNN forward 验证检测坐标映射、越界裁剪、同类 NMS、异类保留、YOLOv5 objectness、分类概率、RGB/BGR 及均值/缩放预处理与错误报告。工作线程测试覆盖批量输入、取消与重新运行、短视频 EOF 最后一帧以及输入错误，不依赖 Python 或在线服务。真实示例模型的 GUI 自检覆盖模型载入、图片推理、界面更新与导出。摄像头硬件需在实际设备上验证。

灰度测试使用对输入像素敏感的真实小网络：红、绿、蓝像素分别验证灰度亮度，检查单通道、三通道复制、统一均值、114 补边和通道错误，并保留旧 ONNX initializer 同时列为 graph inputs 的兼容性。左右拼接测试通过实际视频与推理验证两眼不同信息、奇数宽度、所选图像尺寸及坐标，并检查图片保持完整。新版本完整测试和发行包执行结果以对应版本验证报告为准。

`tests/pt_backend_tests.cpp` 使用官方 YOLOv8n 与 YOLOv5n `.pt` 验证直接加载、自动类别名称和连续图片请求；旧版模型另覆盖改名与同目录其他权重的干扰。同时验证带元数据的 TorchScript 分类，以及解释器缺失、无效检查点、裸权重字典和不兼容预处理的错误。后台进程协议测试覆盖载入取消、推理取消、非法响应、模型身份不一致和载入超时。这些 `.pt` 测试需要已准备的本地 runtime。

`tests/ui_tests.cpp` 覆盖真实 `.pt` 工作台推理、自动类别与后端信息、PNG / JSON / CSV 自动导出，以及七页导航和「更多」入口。灰度用例检查实际灰度显示、原色恢复和清除旧检测；录制用例使用本地短视频，检查开始 / 结束录制、AVI / JSON 文件、录像列表及播放预览。

内置示例模型与图片的来源、授权和预处理方式记录在 [模型说明](docs/模型说明.md)。

## 开源与维护

维护者：**misaka_ning <1468549029@qq.com>**。新编写的 Vision Studio 代码按 **AGPL-3.0-only** 发布，完整正文见 [LICENSE](LICENSE)。Qt、Python 依赖、模型和其他上游资源保留各自许可，正文与元数据保存在 `packaging/licenses/`；摘要见 [第三方许可清单](docs/release/第三方许可清单.md)。发行目录同时提供对应源码，转发发行版时请保留这些配套文件。
