# Vision Studio

**基于 C++17 / Qt 6.8 的本地 YOLO 推理与模型可视化工具。** 支持 ONNX / PT 模型、图片与视频检测、摄像头输入、灰度与左右目处理、标注视频录制，以及结构图、层级树和参数表。

[下载发行版](https://github.com/misaka-ning/vision-studio/releases) · [更新日志](CHANGELOG.md) · [使用指南](docs/使用指南.md) · [报告问题](https://github.com/misaka-ning/vision-studio/issues)

当前正式版本：**[2.0.0](https://github.com/misaka-ning/vision-studio/releases/tag/v2.0.0)**。2.0 保留 [Fluent-Qt](https://github.com/calvinhxx/Fluent-Qt) 风格的深色界面，并集中改进参数输入、文件夹任务停止、保存目录入口与模型库排序。CPU／NVIDIA GPU、ONNX／PT、灰度和左右目处理沿用现有约定。发行平台：**Ubuntu 22.04 LTS amd64**。应用代码采用 **AGPL-3.0-only**。

![V2.0 Fluent 风格检测工作台：RTX 4060 实际 CUDA 推理](docs/preview.png)

## 功能

- **检测工作台**：图片队列、批量图片、视频和摄像头输入，YOLO 检测与单标签图像分类，置信度和 NMS IoU 阈值可调；输入尺寸、置信度和 NMS IoU 不受鼠标滚轮误改，文件夹结果保存放在后台，停止时不受保存队列阻塞。批量完成后可点击队列图片回看本轮标注、目标列表和指标，并导出所选图片的结果。
- **模型库**：导入 ONNX / PT，可拖动排序并在重启后保留；重排保持当前全局模型，统一决定工作台和模型显示使用的文件。
- **模型显示**：Netron 结构图、可展开层级树、参数表；后台预加载并缓存，切换页面与模式复用解析结果，树与表支持搜索和详情。
- **灰度与双目输入**：灰度按模型的 1 / 3 通道适配；单设备水平左右拼接视频或摄像头可只处理左目或右目。预览和导出显示实际所选画面与颜色模式。
- **标注与录制**：检测框、类别、置信度、缩放和平移；保存 PNG / JSON / CSV，自动保存选项下方可直接打开结果目录。视频和摄像头可开始／结束录制，在「录制视频」页浏览、播放和导出。
- **运行记录与更多**：保留处理记录、模型和耗时；「更多」集中提供运行示例与手动导出。
- **计算设备**：工作台可选自动、CPU 或 NVIDIA GPU；「更多」检查并准备独立 GPU 环境。自动模式在 GPU 不可用时显示原因并使用 CPU，显式 GPU 失败时报告错误；结果记录实际设备。

![V2.0 ONNX 模型结构图：Netron 节点与运算连接](docs/model-hierarchy.png)

普通 PT 的层级树表示文件可解析的模块包含关系，裸权重按名称分组；参数表列出权重、常量和缓冲，不能直接作为可训练参数总量。ONNX 更适合查看完整运算连接。不支持或损坏的模型会显示文字说明。详细范围见 [模型说明](docs/模型说明.md)。

## 下载与安装

从 [GitHub Releases](https://github.com/misaka-ning/vision-studio/releases) 选择版本。每版只上传 **3 个手动附件**：安装包 `.deb`、`vision-studio-X.Y.Z-complete-source.tar.xz`、`SHA256SUMS`。

页面另外显示 GitHub 自动提供的 **Source code (zip)** 和 **Source code (tar.gz)** 两个源码入口，它们由平台生成，不能从 Release 中移除。下载三个手动附件后，先校验，再安装：

```bash
sha256sum -c SHA256SUMS
sudo apt install ./vision-studio_2.0.0-1_amd64.deb
vision-studio
```

正式包的 Debian 版本为 `2.0.0-1`，高于 Beta.2 的 `2.0.0~beta.2-1`，可用以上 APT 命令直接升级。先结束检测与录像，导出需要永久保存的结果，退出旧程序再安装；升级保留个人设置、模型、运行记录、导出、录像与已准备的 GPU 环境。本轮临时回看缓存不会跨程序重启保留。

也可从应用菜单启动 **Vision Studio**。首次体验可打开「更多 → 运行示例」，再导入自己的模型。

程序安装到 `/opt/VisionStudio`，设置、运行记录、导出和录像位于 `${XDG_DATA_HOME:-$HOME/.local/share}/vision-studio`。普通用户无需向安装目录写入文件。`VISION_STUDIO_DATA_DIR` 可指定另一数据目录。卸载应用使用 `sudo apt remove vision-studio`，个人数据由用户保留或自行清理。

正式 DEB 包含 Qt 6.8.3 / WebEngine、独立 CPU PyTorch 环境、Netron 9.3.1 与本地示例模型，模型准备完成后可以离线使用；APT 可能需要安装包声明的系统共享库。Git checkout 与便携打包脚本生成的目录需要自行准备 runtime，不能直接当作完整 DEB 安装环境。

V1.6 的 NVIDIA GPU 支持首次需在「更多」点击「准备 GPU 支持」，从官方源下载约 4–5 GB 锁定依赖，建议预留至少 20 GB 空间。配置写入个人数据目录的 `gpu-runtime/`，可以取消；只有实际 CUDA 自检成功才发布环境，失败保留原来的 CPU／GPU 环境。不安装或修改显卡驱动，不在 APT／DPKG 安装过程中下载 CUDA。GPU 准备完成后模型推理可离线执行。

## 推理架构与兼容范围

| 用途 | 实现 | 当前设备 |
| --- | --- | --- |
| 界面、输入、结果显示、录制与导出 | C++17 / Qt 6.8.3 | 本机 |
| ONNX CPU 推理 | C++ / OpenCV DNN 4.5.4 | CPU |
| ONNX GPU 推理 | C++ / ONNX Runtime 1.23.2 CUDA Execution Provider | NVIDIA GPU，可选环境 |
| PT 推理 | 常驻本地 Python / PyTorch 2.9.1 / Ultralytics 8.4.173 | CPU；可选 CUDA 12.8 |
| 模型结构显示 | Qt WebEngine / Netron 9.3.1，本机回环服务 | 本机 |

V1.6 增加 NVIDIA CUDA 后端，TensorRT、AMD 与 Intel GPU 尚未提供。GPU 环境固定为 Python 3.10／3.11、PyTorch `2.9.1+cu128`、Torchvision `0.24.1+cu128`、CUDA 12.8、cuDNN 9.10.2 与 ONNX Runtime 1.23.2；推荐 Linux NVIDIA 驱动 ≥ 570.26。`nvidia-smi` 的 CUDA Version 表示驱动支持上限，不代表已安装 CUDA Toolkit。应用使用官方运行库 wheel，无需全局 Toolkit。[PyTorch 版本](https://pytorch.org/get-started/previous-versions/)、[ORT CUDA 兼容](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[CUDA 12.8 驱动要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。

模型可视化不参与检测推理，结构查看不会执行 `torch.load`；PT 推理加载检查点可能执行其 Python 代码，请使用自己训练或可信来源的模型。

ONNX 支持符合本项目输入输出约定的 YOLOv5、YOLOv8 / YOLO11 原始检测输出与单标签分类；CPU 与 GPU 的模型算子分别以 OpenCV DNN、ONNX Runtime CUDA 支持范围为准。图像输入需为 NCHW，固定 1 或 3 通道；灰度直接输入 C1 或复制到 C3，彩色要求 C3。带内置 NMS、多输入、分割、姿态、旋转框和量化专用预处理不在当前兼容范围。

PT 支持兼容的 Ultralytics YOLOv8 / YOLO11 完整检查点、随附 YOLOv5 v7.0 框架可加载的旧版完整检查点，以及带必要任务、类别和通道元数据的标准 TorchScript。裸 `state_dict` 可显示静态信息，但还需要网络架构才能推理。具体预处理、灰度、左右目和输出约定请阅读 [模型说明](docs/模型说明.md) 和 [使用指南](docs/使用指南.md)。

## 从源码构建

依赖 CMake ≥ 3.21、C++17 编译器、Ninja、Qt 6.8.3 SDK，以及 OpenCV Core / Imgproc / Imgcodecs / Videoio / DNN。Qt 模块为 Core、Gui、Widgets、Svg、Network、WebChannel、WebEngineWidgets，测试还需要 Test；WebEngine 需要配套的 Quick / Qml / Positioning 模块及资源。

```bash
git clone https://github.com/misaka-ning/vision-studio.git
cd vision-studio

# Python 3.10 或 3.11，需要 venv；首次准备会下载依赖
./scripts/setup_pt.sh

# QT_PREFIX 可按本机 SDK 位置修改
QT_PREFIX="$HOME/Qt/6.8.3/gcc_64" ./scripts/build.sh
./build/bin/vision-studio
```

示例模型和测试资源保留在仓库中。`scripts/build.sh` 完成 CMake 配置、构建与 CTest。也可直接使用 CMake：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64" -DBUILD_TESTING=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

源码开发的 PT / Netron 依赖由 `requirements-pt.txt` 和 `requirements-pt.lock.txt` 约束；正式 DEB 使用的 headless OpenCV runtime 另有 [发行依赖锁](docs/release/requirements-release.lock.txt)。`VISION_STUDIO_BASE_PYTHON` 可指定建立 runtime 的 Python，`VISION_STUDIO_PYTHON` 可指定运行时解释器的绝对路径。

GPU 依赖使用独立的 `requirements-gpu.txt`／`requirements-gpu.lock.txt`；完整锁包含版本与官方 wheel 的 SHA256，按 `--no-deps --require-hashes` 安装，避免 Ultralytics 额外装入 GUI OpenCV。源码开发也可执行：

```bash
./scripts/setup_gpu.sh
/usr/bin/python3.10 scripts/gpu_probe.py
```

默认解释器明确使用 `/usr/bin/python3.10`，不跟随 PATH 中的 Conda。可用 `--base-python /绝对路径/python3.11`、`--runtime-dir /用户专用目录/gpu-runtime` 指定配置位置，并用 `VISION_STUDIO_GPU_RUNTIME_DIR` 在应用中选择同一目录。Netron 继续使用独立 CPU 环境。

## 验证与发行

2.0.0 的最终 Qt 6.8.3 Release 构建通过全部 **10 组 CTest**（200.968 秒，零失败、零跳过），覆盖四项操作改进、CPU／CUDA、ONNX／PT、灰度 C1／C3、左右目、录像、批量回看和模型三视图缓存。发布／清理安全 48 项、版本元数据 11 项及 GPU 环境保护 21 项检查通过；独立 ONNX profile 记录 364 个实际 CUDA 节点事件。

最终 DEB 通过普通用户 CPU／现代与旧版 PT、ONNX／PT 三视图缓存、APT 依赖模拟、隔离离线安装／卸载及包内 ONNX／PT CUDA／CPU 四轮检查。包报告绑定 SHA256 `2e5a467b07d82cf600eedeec2bd40624a724efccb4715366997ef04f3a4a042e`，无错误或警告。本机 1080p／4K 自动保存测试中停止观察值为 40／30 毫秒，界面心跳最大间隔为 20／15 毫秒；这不是所有后端和模型的统一时限，CPU 当前 OpenCV forward 仍需自然结束。模型拖动检查采用 Qt 内部移动与重启恢复，物理摄像头未实测。

最终对应源码的文件集合与 SHA256 核验由独立 `source-bundle-qa.json` 记录，发布工具要求其通过后才公开 Release；报告与冻结清单在归档生成后加入仓库，避免递归自校验。范围与真实原始日志见 [2.0.0 发布说明](docs/releases/2.0.0.md) 和 [同版验收证据](docs/releases/evidence/2.0.0)。

历史预发布与正式版记录保留在 [Beta.2](docs/releases/2.0.0-beta.2.md)、[Beta.1](docs/releases/2.0.0-beta.1.md) 和 [1.6.0](docs/releases/1.6.0.md) 及各自证据目录，不改写其原始报告或发行文件。

```bash
./build/bin/vision-studio --smoke /tmp/vision-studio-onnx-smoke
./build/bin/vision-studio --smoke-pt /tmp/vision-studio-pt-smoke
./build/bin/vision-studio --smoke-model /tmp/vision-studio-model-smoke
./build/bin/vision-studio --smoke-model /tmp/vision-studio-pt-model-smoke \
  --display-model "$PWD/models/yolov8n.pt"
```

模型显示自检生成结构图、层级树、参数表截图及含内容数量、模式和缓存复用状态的报告。无显示器时可以使用 `QT_QPA_PLATFORM=offscreen`；正常桌面启动保留 WebEngine sandbox。

打包、对应源码及上游重建说明见 [打包说明](packaging/README.md) 和 [对应源码与重建](docs/release/对应源码与重建.md)。完整源码包内已有应用源码拆分归档、`SOURCE-SHA256SUMS`、`SOURCE-INVENTORY.json` 及上游源码、许可和原始通知，可解压后获取。GitHub 自动生成的 Source code ZIP／tar.gz 来自 Git 标签，不能替代 Release 中的完整对应源码包。

源码主线为 `main`，正式标签为 `vX.Y.Z`，预发布标签为 `vX.Y.Z-beta.N`。每个新版必须按 [版本管理](docs/版本管理.md) 完成测试、源码与标签推送，并使用 `scripts/github_release.py` 发布和校验 Release；远端核验后本地保留最新两个正式版的发行与独立构建副本；2.0.0 发布后为 2.0.0 与 1.6.0。旧 Beta 副本逐版精确核验后单独清理，历史版本可通过标签和 Release 获取，原归档保持不变。

## 文档、贡献与许可

- [使用指南](docs/使用指南.md)、[模型说明](docs/模型说明.md)、[UI 设计](docs/UI设计.md)
- [更新日志](CHANGELOG.md)、[参与贡献](CONTRIBUTING.md)、[版本管理](docs/版本管理.md)
- [第三方许可清单](docs/release/第三方许可清单.md)、[对应源码与重建](docs/release/对应源码与重建.md)

欢迎提交 [Issue](https://github.com/misaka-ning/vision-studio/issues) 或 Pull Request。维护者：**misaka_ning <1468549029@qq.com>**。

新编写的应用代码按 **AGPL-3.0-only** 发布，完整正文见 [LICENSE](LICENSE)。Qt、Ultralytics、YOLOv5、Netron、示例模型和其他组件保留各自许可；通知与正文位于 `packaging/licenses/`。转发发行版时请继续提供对应源码、原始通知与校验文件。
