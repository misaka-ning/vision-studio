# Vision Studio

使用本机 **C++17 / Qt 6.8.3** 构建的桌面视觉推理工作台。ONNX 由 C++ / OpenCV DNN 执行，`.pt` 由常驻的本地 PyTorch 进程执行，均使用 CPU。支持目标检测、单标签图像分类、图片批量处理、视频与摄像头输入。

## 启动

正式发行包面向 **Ubuntu 22.04 amd64**，包含 Qt 6.8.3、独立 CPU PyTorch 环境和本地示例模型。安装后从应用菜单启动 **Vision Studio**，也可运行 `vision-studio`：

```bash
sudo apt install ./vision-studio_1.1.0-1_amd64.deb
vision-studio
```

程序安装到 `/opt/VisionStudio`。设置、运行记录和默认导出写入 `${XDG_DATA_HOME:-$HOME/.local/share}/vision-studio`，普通用户无需向 `/opt` 写入文件。发行版 Python 环境使用无 GUI 的 `opencv-python-headless 4.11.0.86`，原生 ONNX 后端仍使用系统 OpenCV 4.5.4。安装无需下载 Python 模型依赖；APT 可能需要安装系统共享库。

家目录中的便携版可通过 `VisionStudio`、`VisionStudio.desktop` 或 `run.sh` 启动。首次使用可点击 **运行示例**，查看真实 YOLO 模型的推理结果，再导入自己的模型。正式 DEB 与早期 `scripts/package.sh` 生成的便携目录不同：DEB 已携带 CPU runtime，便携打包脚本需要另行安装该环境。

发行说明见 [发行说明](docs/release/发行说明.md)。发布时将 DEB 与 `vision-studio-1.1.0-complete-source.tar.xz` 一同提供，并附校验这两个主文件的 `SHA256SUMS`。完整源码包包含本项目源码归档、`sources/` 下的上游源码和校验内部文件的 `SOURCE-SHA256SUMS`。重建方法和 Qt 替换接口见 [对应源码与重建](docs/release/对应源码与重建.md)。

完整操作说明见 [使用指南](docs/使用指南.md)，界面设计说明见 [UI 设计](docs/UI设计.md)，已执行检查见 [验证报告](docs/验证报告.md)。

![工作台真实 YOLO 检测](docs/preview.png)

## 功能

- **推理工作台**：模型和标签配置、图片队列、视频与摄像头输入、置信度与 NMS 阈值调节。
- **视觉结果**：检测框、类别和置信度叠加，缩放、平移、适应视图，目标表与性能数据。
- **模型库**：导入本地 `.onnx` 或 `.pt`；现代 Ultralytics YOLO `.pt` 自动读取类别名称，配置保存到用户设置。
- **结果导出**：保存标注 PNG、结构化 JSON 和 CSV；图片批量任务可逐张自动导出，视频与摄像头任务在结束后保存最后一帧。
- **运行记录**：查看处理来源、模型、目标数量、推理耗时和执行时间。
- **后台任务**：推理在工作线程中运行，可中止任务，界面保持可操作。

## 模型兼容范围

ONNX 后端使用 **OpenCV 4.5.4 DNN / CPU**，支持能被该版本 OpenCV 导入且符合下列输出约定的模型。

内置可直接运行的 YOLOv5 Nano 示例模型（固定 640 输入），点击“运行示例”即可体验。实际用户模型支持情况以执行结果为准。

| 任务选择 | 预期主要输出 | 处理方式 |
| --- | --- | --- |
| YOLOv5 检测 | `[1, N, 5 + 类别数]` | `cx, cy, w, h, objectness, class scores`，置信度为 objectness × class score |
| YOLOv8 / YOLO11 检测 | `[1, 4 + 类别数, N]` | `cx, cy, w, h, class scores` |
| 图像分类 | `[1, 类别数]` 或 `[1, 类别数, 1, 1]` | 概率直接使用；logits 转为 softmax 概率，显示 top-5 |

检测采用保持比例的 letterbox 预处理，输出坐标映射回原图后进行裁剪和按类别 NMS，每张图最多保留 300 个目标。自定义检测模型须提供与输出类别数一致的标签；默认标签为 COCO 80 类。分类输入缩放到正方形；颜色通道、缩放系数和 RGB 均值可配置。输入宽高使用相同的「输入尺寸」值，需与模型匹配。旧版 YOLOv5 附带的三个辅助检测头可与主要输出一起存在。

**暂不支持**分割、姿态、旋转框、多标签分类、带内置 NMS 的检测输出、多输入模型及量化模型的专用预处理。模型导入成功不代表所有算子都能执行；运行失败会在界面显示错误信息。CUDA / TensorRT 推理后端尚未提供。

### 直接运行 .pt

支持现代 **Ultralytics YOLOv8 / YOLO11 的完整 `.pt` 检查点**，检测与分类使用模型自带的预处理和后处理。应用直接载入 `.pt`，每个任务保留一个后台 Python 进程和模型实例，连续处理图片或视频帧。

检测任务支持输入尺寸、置信度与 NMS IoU；类别名称从检查点读取。自行导入标签可覆盖名称，数量必须与模型类别数匹配。RGB / BGR、缩放系数和均值属于 ONNX 预处理配置，`.pt` 后端使用模型原生配置。

同时支持与随附 `vendor/yolov5` v7.0 框架兼容的完整 YOLOv5 检查点，内置 `models/yolov5n.pt` 可直接运行。带任务和类别元数据的 TorchScript `.pt` 支持标准 YOLO 原始检测输出或单标签分类输出。

只有 `state_dict` 的 `.pt` 文件还需要网络架构，当前版本不能仅凭权重字典推理。自定义模型类仍需要原项目代码；分割、姿态和旋转框 `.pt` 暂不支持。

直接 `.pt` 推理使用交付目录的独立 `runtime/bin/python`：Python 3.10.12、PyTorch 2.9.1+cpu、Torchvision 0.24.1+cpu、Ultralytics 8.4.173。运行环境信息见 `runtime-info.json`，依赖记录见 `requirements-pt.txt`。`VISION_STUDIO_PYTHON` 可指定另一个具有这些依赖的解释器绝对路径。模型和依赖准备完成后，推理可离线执行。检查点载入可能通过 pickle 执行 Python 代码，请使用自己训练或来自可信来源的 `.pt` 文件。

## 从源代码构建

依赖：CMake ≥ 3.21、C++17 编译器、Qt 6.8 Core / Gui / Widgets / Svg / Test、OpenCV Core / Imgproc / Imgcodecs / Videoio / DNN。

直接 `.pt` 后端还需要本地 Python、CPU PyTorch、Torchvision 和 Ultralytics；它们在独立 runtime 中，与 C++ 编译过程分开。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

也可以运行 `./scripts/build.sh` 一次完成构建与检查。`./scripts/package.sh /绝对路径/新的目录` 可生成包含源码、模型和可执行文件的交付目录；不会携带原工程的 CMake 缓存、偏好设置或临时测试目录。`docs` 保留此次验证报告与测试日志。

本机主文件夹中的安装已准备 `.pt` runtime。单独生成的交付包不携带约 1.5 GB 的 Python 环境；需要直接运行 `.pt` 时，准备 Python 3.10 或 3.11，在该交付目录执行 `./scripts/setup_pt.sh` 安装独立 CPU runtime。安装按 `requirements-pt.lock.txt` 约束依赖版本，需要下载依赖；完成后图片、视频和模型推理均可离线执行。

GUI 自检使用本地示例模型执行推理并导出结果，`--smoke` 使用 ONNX，`--smoke-pt` 使用 PyTorch 检查点：

```bash
./build/bin/vision-studio --smoke /tmp/vision-studio-smoke
./build/bin/vision-studio --smoke-pt /tmp/vision-studio-pt-smoke
./build/bin/vision-studio --screenshot /tmp/vision-studio.png
```

无显示器环境可设置 `QT_QPA_PLATFORM=offscreen`。桌面环境应直接启动应用以获得正常字体和窗口尺寸。

## 验证方法

`tests/core_tests.cpp` 在临时目录生成小型 ONNX 网络，通过真实 OpenCV DNN forward 验证检测坐标映射、越界裁剪、同类 NMS、异类保留、YOLOv5 objectness、分类概率、RGB/BGR 及均值/缩放预处理与错误报告。工作线程测试覆盖批量输入、取消与重新运行、短视频 EOF 最后一帧以及输入错误，不依赖 Python 或在线服务。真实示例模型的 GUI 自检覆盖模型载入、图片推理、界面更新与导出。摄像头硬件需在实际设备上验证。

`tests/pt_backend_tests.cpp` 使用官方 YOLOv8n 与 YOLOv5n `.pt` 验证直接加载、自动类别名称和连续图片请求；旧版模型另覆盖改名与同目录其他权重的干扰。同时验证带元数据的 TorchScript 分类，以及解释器缺失、无效检查点、裸权重字典和不兼容预处理的错误。后台进程协议测试覆盖载入取消、推理取消、非法响应、模型身份不一致和载入超时。`tests/ui_tests.cpp` 覆盖真实 `.pt` 工作台推理、自动类别与后端信息，以及 PNG / JSON / CSV 自动导出。这些 `.pt` 测试需要已准备的本地 runtime。

内置示例模型与图片的来源、授权和预处理方式记录在 [模型说明](docs/模型说明.md)。

## 开源与维护

维护者：**misaka_ning <1468549029@qq.com>**。新编写的 Vision Studio 代码按 **AGPL-3.0-only** 发布，完整正文见 [LICENSE](LICENSE)。Qt、Python 依赖、模型和其他上游资源保留各自许可，正文与元数据保存在 `packaging/licenses/`；摘要见 [第三方许可清单](docs/release/第三方许可清单.md)。发行目录同时提供对应源码，转发发行版时请保留这些配套文件。
