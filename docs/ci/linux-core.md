# Linux PR 构建与核心回归

工作流：`.github/workflows/qt-core-checks.yml`，检查名 `qt-core`。在 PR、`main` push 和手动运行时执行，权限仅 `contents: read`。既有 `source-checks.yml` 保留原样。本检查不制作安装包、不创建标签、不执行发布工具，也不替代完整发行验收。

## 固定环境与执行规则

- Ubuntu 22.04 x86_64、GCC、CMake、Ninja、发行版 `libopencv-dev`（Core / Imgproc / Imgcodecs / DNN / Videoio，包括 MJPEG AVI 编解码器）。
- Qt **6.8.3** `linux_gcc_64` SDK，安装目录 `6.8.3/gcc_64`。基础归档包含 Core / Gui / Widgets / Network / Test / Svg / Qml / Quick；额外安装 `qtwebengine qtwebchannel qtpositioning qtshadertools`，满足 WebEngine 的传递依赖。
- 独立 Python 3.10 venv，仅安装 `scripts/ci/requirements.txt`：固定 Qt 下载器和 Netron 9.3.1；不安装 PyTorch、Ultralytics 或 CUDA。窗口关闭测试最后重新打开真实 ONNX 结构，需要 Netron 的本地 HTTP 服务和 Qt WebEngine。
- `Debug` 构建应用和测试，保留 C++ / Qt 断言，不设置 `NDEBUG` / `QT_NO_DEBUG`。`VISION_STUDIO_PR_CORE_CHECKS=ON` 要求 Linux、Qt 6.8.3、Debug 和 `BUILD_TESTING=ON`；默认关闭，不改变发行构建。
- `vision-pr-core` 目标依赖主程序 `vision-studio` 和完整的 21 套测试；`tests/pr_core.cmake` 是唯一选择名单。每套全部运行，不筛掉单个数据行。
- `scripts/ci/run_core_tests.py` 对比名单、CTest JSON 发现结果和 JUnit 实际结果。缺失、禁用、未构建、零测试、非零退出、失败、跳过、blacklist 或缺失 QtTest 总计均失败；CTest 也将 `SKIP` / `XFAIL` 输出判失败。保留详细输出，JUnit 输出上限提升到 1 MiB，避免默认截断隐藏套件总计。
- offscreen、软件图形；移除继承的 `DISPLAY` / `WAYLAND_DISPLAY`。临时数据、设置、缓存和不存在的 GPU runtime 路径置于构建目录；不使用宿主机安装的应用 runtime。QtTest 内部 helper 仍可访问本机回环端口。
- 串行 CTest，保留套件自身 60–120 秒超时；默认超时 180 秒，测试步骤 10 分钟，整个作业 45 分钟。编译并行度 2，避免共享 runner 内存压力影响测试。
- 缓存 Qt SDK、pip 下载和 ccache；不缓存构建目录或测试结果。首次没有缓存也必须成功。缓存 key 包含 OS / SDK / 配置或依赖清单；SDK 缓存同时包含工作流哈希，模块变化会失效。
- 无论成功失败均上传配置、编译、CTest 详细日志、JUnit、测试发现结果和 CMake 诊断，保留 14 天。步骤失败不会被 `tee` 吞掉（GitHub 的显式 bash shell 使用 `-eo pipefail`），没有 `continue-on-error` 或失败重试绿灯。

## CI 实际执行的 CTest 套件

以下是整个套件的依赖核查，不将名称中的 GPU、PT 或 camera 字段误认为真实硬件调用。

| CTest 名称 | 覆盖与必需依赖 |
| --- | --- |
| `vision-core-tests` | C++ 生成的真实 ONNX、仓库 YOLOv5n / bus.jpg，CPU OpenCV DNN、预处理、批量与合成视频；无 Python 推理 |
| `vision-inference-delivery-tests` | 仓库 ONNX、生成图片，CPU 推理交付、背压和取消 |
| `vision-model-yaml-tests` | 内存元数据、YAML 文本与临时文件；无模型框架 |
| `vision-dataset-session-tests` | 临时 JSON 持久化与目录权限失败注入；必须以普通用户运行，root 导致跳过会失败 |
| `vision-app-services-tests` | 临时设置、历史、像素渲染和异步结果写入 |
| `vision-gpu-environment-service-tests` | Python 标准库 helper 模拟进度、超时、取消、进程回收；不准备 CUDA，不访问 GPU |
| `vision-model-session-controller-tests` | 导入、选择、文件身份、过期解析、线程 TLS 清理，以及仓库真实 ONNX 元数据 |
| `vision-detection-controller-tests` | 检测取消、原生线程尾部、结果排空；真实 ONNX CPU 和临时 MJPEG 视频 |
| `vision-evaluation-controller-tests` | 合成报告快照、审核状态、拒绝启动与线程退出；不执行完整 Python 评估 |
| `vision-operation-coordinator-tests` | 配置冻结、权限矩阵、重入关闭、所有参与者就绪屏障 |
| `vision-recording-controller-tests` | 临时 MJPEG AVI 编解码、导出、线程与关闭；无摄像头 |
| `vision-workbench-configuration-tests` | 纯配置组合、模型文件身份、类别筛选 |
| `vision-image-result-store-tests` | 合成图像、缓存、异步快照、失败写入与关闭 |
| `vision-result-delivery-tests` | 序列交付、历史/导出失败、逐票确认、异步排空 |
| `vision-workbench-page-tests` | 原生 Qt 控件投影与操作信号；offscreen |
| `vision-workspace-pages-tests` | 模型库、模型显示和录像页投影；不解析真实 PT |
| `vision-simple-pages-tests` | 更多、历史和使用指南页投影；资源来自 qrc |
| `vision-component-shutdown-tests` | Python 标准库 helper、POSIX FIFO 阻塞读、转换/查看器子进程与预览线程池关闭；helper 产物不代表真实转换 |
| `vision-runtime-setup-shutdown-tests` | 本机 HTTP 假归档、测试可执行文件充当解压子进程、锁和暂存目录回收；不下载真实 runtime，也不需要 PowerShell |
| `vision-workspace-drop-filter-tests` | 临时文件、拖入格式验证和 Qt 控件事件 |
| `vision-window-shutdown-tests` | C++ 生成真实 ONNX 的 CPU 推理、接受结果写完后退出、队列与孤儿线程屏障、helper 回收、同进程重建窗口并用 Netron 打开 ONNX |

## 本检查排除的 CTest 套件

当前 Linux 共注册 44 套，选择 21 套、排除下列 23 套。排除是 CI 范围选择，不表示它们通过；仍保留 CMake 注册，完整回归必须另备依赖后运行。

| CTest 名称 | 排除原因 |
| --- | --- |
| `vision-recording-tests` | 同套含真实 CUDA 录像行；无 GPU 会 QSKIP，整套排除，CPU 录像由 core / detection-controller / recording-controller 覆盖 |
| `vision-gpu-tests` | 真实 NVIDIA / CUDA runtime 必需，无硬件会 QSKIP |
| `vision-conversion-inference-tests` | 同套混合真实 CPU / CUDA 转换与推理，另需 Torch / ONNX 导出依赖；无 GPU 会 QSKIP |
| `vision-pt-tests`, `vision-pt-batch-python-tests` | 真实 PT 推理与 batch，需 PyTorch / Ultralytics / Python OpenCV 和对应模型 |
| `vision-model-conversion-tests` | 真实 PT → ONNX / TorchScript 导出，需完整转换 runtime |
| `vision-modelviewer-tests` | 生成并解析多种 PT / ONNX fixtures，需要 PyTorch / ONNX 等额外依赖；本 CI 的窗口测试只验证 ONNX 重建生命周期 |
| `vision-ui-tests` | 大范围 MainWindow 集成，包含真实 PT 与模型显示，需完整 runtime |
| `vision-scope-filter-ui-tests` | Python 生成 ONNX fixtures，需 ONNX / NumPy 等完整测试依赖 |
| `vision-input-session-controller-tests`, `vision-input-drop-ui-tests`, `vision-dataset-session-ui-tests` | 真实数据集导入执行 Python 扫描/评估 helper，需要额外数据集依赖；单纯图片/窗口关闭已覆盖 |
| `vision-evaluation-worker-tests`, `vision-evaluation-integration-tests` | 真实评估、报告及 PT 集成，需准备评估 runtime |
| `vision-evaluation-python-tests`, `vision-evaluation-reporting-tests`, `vision-evaluation-snapshot-tests`, `vision-evaluation-import-tests` | Python 评估/绘图/快照/导入，需 NumPy / Pillow / PyYAML / Matplotlib 等，不属于本次精简 C++ 门禁 |
| `vision-batch-responsiveness-tests` | 专项大队列/数据集响应与界面集成，本轮聚焦控制器及关闭路径，未纳入完整 UI 验收 |
| `vision-evaluation-panel-tests` | 大范围评估界面布局/可见性/交互专项，属于后续 UI 检查；不是硬件不兼容 |
| `vision-conversion-ui-tests` | 转换页面完整交互专项；基础依赖可满足，但本轮仅纳入转换关闭生命周期，不宣称转换页面完整验收 |
| `vision-netron-server-tests` | 独立 Python HTTP/协议全量专项，依赖可满足，本轮 Qt 核心集合未重复纳入 |
| `vision-gpu-runtime-tests` | 已由保留的 `source-checks` 执行同一 Python 文件，不在 Qt job 重复；这是模拟安全检查，不是 GPU 验收 |

原 `source-checks` 中 `windows_packaging_tests.py` 的 7 项原生 PowerShell 用例在 Linux 上已有显式 skip；保留原行为并如实报告，不将其计入新 `qt-core` 的零跳过结果。

`vision-cpu-runtime-tests` 仅在 Windows 注册，不在上述 Linux 44 套内。物理摄像头、真实 CUDA、完整 PT/评估/转换、Windows、安装/卸载和完整对应源码重建均需单独验收。配置阶段仍会明确警告完整评估 Python 依赖缺失；不屏蔽警告，不禁用这些发行测试。

## 本地等价命令（Ubuntu 22.04，普通用户）

在干净 checkout 根目录执行。首次准备需要网络；已有 Qt 6.8.3 可将 `QT_PREFIX` 指向现有 SDK 并略过 SDK 下载，记录这个差异。系统包准备需要 sudo，测试不要以 root 运行。

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build ccache python3-venv python3-pip \
  libopencv-dev libgl1-mesa-dev libegl1-mesa libopengl0 libxkbcommon-dev \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-shape0 \
  libxcb-randr0 libxcb-xfixes0 libxcb-sync1 libxcb-xkb1 libxcb-glx0 \
  libnss3 libnspr4 libasound2 libdbus-1-3 libxcomposite1 libxdamage1 \
  libxrandr2 libxtst6 libxi6 libxshmfence1 libgbm1 fonts-noto-cjk

export PIP_CACHE_DIR="$PWD/.cache/pip"
export CCACHE_DIR="$PWD/.cache/ccache" CCACHE_MAXSIZE=400M
/usr/bin/python3 -m venv build-ci-tools
build-ci-tools/bin/python -m pip install -r scripts/ci/requirements.txt
build-ci-tools/bin/python -m aqt install-qt linux desktop 6.8.3 linux_gcc_64 \
  --outputdir .cache/Qt --modules qtwebengine qtwebchannel qtpositioning qtshadertools
export QT_PREFIX="$PWD/.cache/Qt/6.8.3/gcc_64"
test "$("$QT_PREFIX/bin/qmake" -query QT_VERSION)" = 6.8.3

set -o pipefail
mkdir -p build-ci/logs
cmake -S . -B build-ci -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  -DVISION_STUDIO_PR_CORE_CHECKS=ON \
  -DCMAKE_PREFIX_PATH="$QT_PREFIX" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DPython3_EXECUTABLE="$PWD/build-ci-tools/bin/python" \
  -DVISION_STUDIO_TEST_PYTHON="$PWD/build-ci-tools/bin/python" \
  2>&1 | tee build-ci/logs/configure.log
cmake --build build-ci --target vision-pr-core --parallel 2 \
  2>&1 | tee build-ci/logs/build.log
test -x build-ci/bin/vision-studio
build-ci-tools/bin/python scripts/ci/run_core_tests.py build-ci
```

CTest 实际命令见入口脚本（`-L '^pr-core$' --no-tests=error --parallel 1 --timeout 180 --verbose --output-on-failure --output-junit …`）；请使用脚本以同时检查发现数量和零跳过结果，单独 `ctest -N` 只是发现，不能算执行。新增/删除名单项时同步核查依赖并更新上表；不靠改断言、吞掉失败或选择部分测试函数获得通过。

## 本次实现验证

实际结果与运行链接记录在 PR 的「验证」部分。CI 日志 artifact 是单次作业的证据；是否通过以该次作业实际状态、`summary.txt`、`ctest.xml` 和完整日志为准，不能沿用历史发行报告。
