# Vision Studio Debian 发布包

此包面向 **Ubuntu 22.04 LTS、amd64（x86_64）**。Qt 6.8.3 / WebEngine、CPU PyTorch、Ultralytics、Netron 9.3.1、ONNX 1.17.0／protobuf 6.33.0 转换依赖与示例模型随包提供；系统负责 Python 3.10、OpenCV 4.5 和桌面图形基础库。CPU 首次启动、推理、模型结构显示和转换不需要下载页面、模型或 Python 依赖；可选 NVIDIA GPU 需用户首次明确准备独立依赖。

默认程序安装在 `/opt/VisionStudio`，菜单与命令为 `vision-studio`。模型、运行库和程序属于系统文件；设置、会话和导出文件使用当前用户的应用数据目录。卸载程序不会删除用户创建的结果。

2.1.0 本轮新增模型库右键管理与 PT → ONNX／TorchScript 转换，保持既有 Fluent 界面、推理、灰度 C1／C3、左右目、录制与结果约定。名称和备注持久保存，重命名只改显示名称，移除列表项保留文件。转换页位于「录制视频」之后、「更多」之前，默认使用全局模型，CPU FP32、batch=1、固定正方形、无 NMS，ONNX 可选 Opset 12／17；后台可取消，已有输出拒绝覆盖，完成后可打开目录或加入模型库。入库后根据元数据应用任务、类别、尺寸和通道，手工标签优先。正式 Release 采用 `prerelease=false`／Latest；最终包与对应源码须在发行前以同版文件完成验收，未通过不发布。

本版内部 Debian Control Version 为 `2.1.0-1`，公开文件名为 `vision-studio_2.1.0-1_amd64.deb`。该版本高于 `2.0.0-1` 及此前 Beta，可直接升级；安装前结束检测、录像与模型转换，导出临时结果并退出旧程序。版本排序由 Control 决定，文件名采用 GitHub 已验证的安全字符集。

## 安装与启动

```sh
sudo apt install ./vision-studio_2.1.0-1_amd64.deb
vision-studio
```

也可从应用菜单打开“Vision Studio 视觉模型工作台”。`apt` 会安装 control 声明的系统依赖。此包需要 Python 3.10，因此不适用于默认 Python 3.12 的 Ubuntu 24.04；其他发行版需要单独构建和验证。

图片结果按后台写入完成节奏交付，临时缓存、标注绘制、自动导出与运行记录写入不在 UI 线程执行。停止不等待后台写入；已交付结果可回看，已接受的写入完成后恢复下一任务入口。CPU OpenCV 的当前 forward 不能强制中断，界面保持响应；CUDA／PT 保留各自取消机制。关闭时异步等待收尾。

## CPU 模型转换

`scripts/model_convert.py` 与 Qt 转换页随包部署，运行使用 `/opt/VisionStudio/runtime/bin/python`，不随工作台 GPU 选择改变。支持兼容的现代 Ultralytics 检测／分类、旧版 YOLOv5 检测完整 PT 及有必要元数据的标准 TorchScript；导出 ONNX 或带 `config.txt` 的 `.torchscript`，读取模型实际 C1／C3，输入尺寸为 32 的倍数。本次来源在开始时固定，完成后须明确入库才改变全局模型；C1 自动选择灰度，C3 仍需按训练预处理选择颜色模式。不提供 ONNX → PT 图转换或训练检查点还原，未知自定义网络需原始定义。

输出默认位于个人数据目录 `converted-models/`，也可选其他可写目录，原始模型与已有输出受保护。取消通过后台进程停止完成；若完整结果已在取消请求前写出，界面提示完成或保留文件，不承诺删除已完成结果。ONNX 针对旧版 OpenCV 的部分静态形状与广播限制做等价规约，保持标准算子，保存前进行图检查和转换环境 Python OpenCV CPU 输出比较；系统 C++ OpenCV、真实图片和 GPU 后端仍须分别验证。

CPU runtime 为实际 59 项发行版，采用 `docs/release/requirements-release.lock.txt` 和 `--no-deps` 重建，避免装入开发 GUI OpenCV；TorchScript 无新增依赖，ONNX 转换使用 ONNX 1.17.0／Python protobuf 6.33.0，无需 ONNX Script、ONNX Slim 或独立 ONNX Runtime。转换必须在最终安装树内普通用户、无网络情况下实际核验，不能仅以源码运行成功代替。

## 可选 NVIDIA GPU

工作台提供自动、CPU、NVIDIA GPU 与编号选择。CPU ONNX 使用 C++ OpenCV DNN；GPU ONNX 使用 C++ ONNX Runtime 1.23.2 CUDA；PT 三类兼容模型使用独立 PyTorch CUDA 环境。自动模式初始化无法使用 GPU 时报告原因并回退 CPU，显式 GPU 不会静默回退。

在「更多」检查并点击「准备 GPU 支持」，也可执行：

```sh
/opt/VisionStudio/scripts/setup_gpu.sh
/usr/bin/python3.10 /opt/VisionStudio/scripts/gpu_probe.py
```

首配从官方源下载约 4–5 GB，建议预留至少 20 GB（包含解包暂存与缓存）。GPU 环境位于个人数据目录的 `gpu-runtime/`，准备成功并通过实际 CUDA 探针后原子发布；可以取消，取消／失败保留原 CPU／GPU 环境。默认系统 Python 3.10，不跟随 Conda PATH；DEB 声明 `python3.10-venv`。不修改驱动，不要求全局 CUDA Toolkit，不在 DPKG maintainer scripts 下载 GPU 依赖。

固定 PyTorch `2.9.1+cu128`、Torchvision `0.24.1+cu128`、CUDA 12.8、cuDNN 9.10.2.21 与 ORT 1.23.2。使用完整版本／SHA256 锁与官方 CloudFront Torch 链接，PyPI 提供其他依赖。可用 `--wheelhouse /本地官方wheel目录` 复用按锁校验的 wheel；完整且兼容的缓存会禁用网络索引，仅从本地安装。准备完成后可以离线推理。GPU 安装环境保留实际依赖／许可清单和原通知，NVIDIA 二进制不写入 DEB，也不做 strip 或 GNU 库修改。

推荐 Linux NVIDIA 驱动 ≥ 570.26；TensorRT、AMD、Intel GPU 未提供。设置、灰度 C1／C3、左右目与模型结构查看不随设备选择改变。`VISION_STUDIO_GPU_RUNTIME_DIR` 可选择另一专用用户目录；CPU 与 Netron 环境独立保留。

## 无管理员权限构建

构建需要已有的 Release 二进制、包含 WebEngine / WebChannel / Positioning 的 Qt 6.8.3 SDK、包含 Netron 9.3.1、ONNX 1.17.0／protobuf 6.33.0 的 Python 3.10 CPU 环境，以及 `dpkg-dev`、`patchelf`、`binutils`。Qt SDK 的间接依赖包括 Qml / Quick / QuickWidgets。打包只修改新 staging 目录中的副本，不修改输入目录。

```sh
python3 packaging/build_deb.py \
  --binary build/release/bin/vision-studio \
  --qt-prefix /path/to/Qt/6.8.3/gcc_64 \
  --extra-qt-plugins build/qt-imageformats-prefix/plugins \
  --runtime build/release-runtime \
  --source-root /干净源码绝对路径
```

默认产物在 `output/releases/2.1.0/`，staging 在 `output/deb-stage/2.1.0-1/`。默认版本从 CMake 的 `VISION_STUDIO_APP_VERSION` 读取；本版应用版本为 `2.1.0`，Debian 包版本为 `2.1.0-1`。已存在的 staging 和 `.deb` 会被拒绝覆盖；需要重打包时指定新的 `--stage` 和 `--output`。`--install-root` 可改变应用安装目录，默认 `/opt/VisionStudio`；编译时的生产资源根也应使用相同路径。`--compression zstd` 提供另一种 Ubuntu 支持的压缩方式，默认使用 xz。

构建脚本保留 ELF 依赖闭包需要的 Qt 运行库与图片、X11、Wayland、输入法及桌面文件对话框插件，重写为 `$ORIGIN` 相对 RUNPATH，去除 staging 内应用和 Qt 的调试符号。WebP/TIFF 等补充插件通过 `--extra-qt-plugins` 加入，必须匹配 Qt 6.8.3。

WebEngine 部署另包含 `libexec/QtWebEngineProcess`、六个 `resources/` 文件和 `translations/qtwebengine_locales/`。辅助进程作为 ELF 依赖分析入口，使用自身相对 RUNPATH；应用与辅助进程的 `qt.conf` 及启动器提供正确资源路径。默认启动器保留 Chromium sandbox，不要求修改系统权限或安装浏览器。Netron 的 Python / JS 资产由 runtime 直接携带，`scripts/netron_server.py` 只在本机回环随机端口服务当前文件。

发行环境使用 `opencv-python-headless`，避免 Python OpenCV 再带入另一套 Qt5。Python wheels 的原生库不重新 strip，以免损坏 auditwheel 调整过的 ELF 段布局。构建检查符号与版本 ABI，将 GNU 数学/线程运行库依赖重定向到 Jammy 系统库；OpenCV 的旧 LP64 OpenBLAS 使用系统 BLAS/LAPACK 替代，NumPy 的独立 ILP64 OpenBLAS仍保留。实际版本与全部包锁由所选发行环境元数据生成。

Python 包包含运行所需模块与授权资料，删除 pip、旧环境的 console 入口、激活脚本和字节码缓存；解释器通过 `/usr/bin/python3.10` 运行，不依赖原虚拟环境位置。重建 headless 环境时应按完整发行锁使用 `pip --no-deps`，避免 Ultralytics 的默认依赖重新安装 GUI OpenCV。`dpkg-shlibdeps` 根据 Jammy 的库元数据生成系统依赖；`dpkg-deb --root-owner-group` 写入系统所有权，无需 sudo。

## 验证与发布

2.1.0 开发验收有效 13 套通过、0 失败／跳过，由完整基线和受影响复验汇总，不是一次整套通过。最终 C++ 转换推理 14 项／62.674 秒、Python 20 项／56.295 秒通过，含 18 次实际导出。ONNX 以标准等价乘加兼容旧 OpenCV 的常量左侧减法；TorchScript 检测缓存随设备移动，跟踪和 CUDA 推理保持严格 FP32。完整 YOLOv8n／YOLOv5n 固定 640 输入的 CPU／CUDA 对照与原始输出核验已通过，范围和限制见 `docs/releases/evidence/2.1.0/ctest-qa.json` 与 `conversion-portability-proof/`。早期候选及原始失败记录保留，最终安装包还须单独通过下面的发行验收。

V1.6 的配置安全与许可测试执行 `/usr/bin/python3.10 -m unittest discover -s tests -p gpu_runtime_tests.py -v`。正式发布还要求绑定最终 DEB SHA256 的 `gpu-qa.json`，包含实际 ONNX CUDA 节点、PT CUDA、安装包 ONNX／PT CUDA、CPU 回归、灰度 C1／C3、左右目与录像全部通过的真实记录；安装包实际 CUDA 验收不能由下载成功、设备名称或 CPU 测试代替。结果记录请求与实际设备、GPU 名称／编号和自动回退原因。

包旁生成 SHA256 和 JSON manifest。发布前应运行 `packaging/verify_deb.py`，检查 ownership、RUNPATH、依赖与路径，再在解压树内以普通用户运行真实 ONNX、现代 `.pt`、旧版 YOLOv5 `.pt` 推理和模型显示。模型显示分别验证 ONNX 与 PT 的结构图、层级树、参数表，保存 `model-display.png`、`model-hierarchy.png`、`model-parameters.png`；报告核对 `nodes`、`hierarchy_items`、`parameter_rows`、`display_modes` 与 `cache_preserved`，确认同服务、同浏览器缓存复用，不能仅以 HTTP 打开作为成功。安装到 `/opt` 后应用目录应保持不可写，导出结果必须写入用户目录。转换验收另需真实新模型输出、ONNX checker、TorchScript 重载和配置元数据、与原 PT 的输出比较、入库后推理、取消与已有文件保护，保存 `conversion-qa.json` 和原始日志并绑定最终 DEB；最终实际结果以同版证据报告为准，未通过不发布。

2.1.0 保持原 15 份上游材料，新增 ONNX 源码、Python protobuf v33.0 完整源码与 ONNX wheel 静态 C++ protobuf 3.21.12 源码，总计 18 份。四份原通知与成员哈希位于 `packaging/licenses/model-conversion/`，Python 实际清单重收为 59 项，Qt 原通知／SDK 记录不变。名称／备注／移除、后台转换、C1／C3、类别与配置、取消／文件保护和原功能回归应在冻结前实际核验；真实开发记录与最终包验收分别保存在 `docs/releases/evidence/2.1.0/`，开发通过不代替最终安装包验收。

`build_deb.py` 和源码工具递归复制不会自动只选 Git 文件。使用从 Git 跟踪与明确授权输入整理的干净源码目录；不包含其他任务、个人训练数据、runtime 或输出。新增转换 helper 必须进入打包明确复制列表；所有依赖、通知、锁与包内说明冻结后再构建，改变这些输入须重打包重验。旧版本归档不覆盖，正式 DEB 生成后执行：

```sh
python3 docs/release/build_source_archive.py --source-root /干净源码绝对路径 \
  --release-directory output/releases/2.1.0
```

该命令生成应用源码归档、完整源码包和两层校验清单；若同版本的归档已存在，会拒绝覆盖。只在确定需要重建当前版本时使用 `--overwrite`，不要修改旧版本文件。

```sh
dpkg-deb --info output/releases/2.1.0/vision-studio_2.1.0-1_amd64.deb
python3 packaging/verify_deb.py output/releases/2.1.0/vision-studio_2.1.0-1_amd64.deb \
  --smoke --apt-simulate --source-dir output/releases/2.1.0
python3 packaging/verify_install_namespace.py \
  output/releases/2.1.0/vision-studio_2.1.0-1_amd64.deb
python3 packaging/verify_gpu_deb.py \
  output/releases/2.1.0/vision-studio_2.1.0-1_amd64.deb \
  --runtime-dir "$HOME/.local/share/vision-studio/gpu-runtime" \
  --report output/releases/2.1.0/packaged-gpu-qa.json
```

`verify_gpu_deb.py` 使用已通过准备自检的独立 GPU 环境，以普通用户运行解压包的真实启动器，分别检查 ONNX CUDA、PT CUDA、ONNX CPU 和 PT CPU，并核验实际后端／设备导出。它不安装 GPU 依赖或修改驱动，报告绑定被测 DEB SHA256；该报告仅证明包内四轮检查，最终 `gpu-qa.json` 还须合并 CUDA 节点、灰度与左右目／录像等实际验收，不能直接复制改名。

隔离安装验收在私有命名空间内执行安装、断网普通用户运行和卸载，不修改宿主的软件包数据库。无显示器检查可使用 `--disable-gpu` 软件渲染，默认仍保留 sandbox；若特定测试容器无法提供 Chromium 所需的内核隔离，应在该次验收记录中明确说明测试限制，不改正常启动器。

运行环境只读取选定的本地 `.pt`，不会根据文件名下载或替换权重。可用 `VISION_STUDIO_PYTHON` 指定另一个兼容的本地环境，用 `VISION_STUDIO_DATA_DIR` 指定测试或导出数据目录。`VISION_STUDIO_QT_HOME` 可选择自行构建的兼容 Qt 6.8.3 prefix，须包含 `lib/`、`plugins/`、`libexec/QtWebEngineProcess`、`resources/` 和 `translations/qtwebengine_locales/`；替换库、插件和 WebEngine 资产共同加载。发布授权、模型来源及第三方许可见包内 `/usr/share/doc/vision-studio/copyright` 和 `licenses/`，以及 `docs/release/`。每版 Release 只上传三个手动附件：DEB、`vision-studio-2.1.0-complete-source.tar.xz` 与校验两个主文件的 `SHA256SUMS`，源码内的 `SOURCE-SHA256SUMS` 校验各原始归档。GitHub 另外自动显示 Source code zip／tar.gz，它们不能替代完整对应源码包；QA 和原始日志保存到仓库 `docs/releases/evidence/2.1.0`，不零散上传 Release。

对应源码包含 qtbase、qtsvg、qtwayland、qtimageformats、qtdeclarative、qtwebchannel、qtpositioning、qtwebengine 八个模块；QtWebEngine 归档包含 Chromium 源码及原通知。Netron 9.3.1 原始 tag 源码和 MIT、Dagre / Graphlib 原通知也随源码交付。C++ ONNX Runtime 1.23.2 API 头文件、原始 MIT 许可和官方来源 SHA256 保留在 `vendor/onnxruntime/`。新增 ONNX 1.17.0、Python protobuf v33.0 与 ONNX 静态 C++ protobuf 3.21.12 的原始源码和通知一同保存，总计 18 份上游材料。CPU 许可快照与实际 59 项发行 runtime 元数据一致，完整重建步骤见 `docs/release/对应源码与重建.md`。

参考：[dpkg-deb 官方手册](https://manpages.debian.org/bookworm/dpkg/dpkg-deb.1.en.html)、[dpkg-shlibdeps 官方手册](https://manpages.debian.org/bookworm/dpkg-dev/dpkg-shlibdeps.1.en.html)。
