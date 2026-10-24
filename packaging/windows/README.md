# Windows 双安装包

2.9.1 使用 Qt 6.8.3、MSVC x64、OpenCV 4.12.0 构建，提供同一应用的离线和在线安装器。实际系统、检查和限制以[同版证据](../../docs/releases/evidence/2.9.1/windows/README.md)为准。

## 安装与分发

| 文件后缀（前缀 `vision-studio-X.Y.Z-`） | 用途 |
| --- | --- |
| `windows-x64-offline-setup.exe` | 完整 CPU 环境，一个 EXE 即可分发，导入自己的模型与输入后离线使用 |
| `windows-x64-online-setup.exe` | 先安装 Qt/OpenCV 与运行脚本；需要 Python 的功能首次使用时准备共用环境 |
| `windows-x64-cpu-runtime.zip` | 在线包按固定 URL、大小、SHA256 下载的同一 CPU 组件 |
| `windows-sources.zip` | 独立应用源码、重建材料、原始通知与 Windows FFmpeg 对应源码 |
| `SHA256SUMS-windows-x64` | 上述四个文件的校验值 |

普通用户选择一个 EXE 即可。两个安装器均不安装源码归档、历史文档、QA、示例模型或测试图片，保留界面资源、使用指南、运行脚本和许可证。在线包的 ONNX 本地推理不需要 Python；PT、转换、Netron 视图、数据集导入／评估和 GPU 准备共用按需下载环境，启动时不自动下载。

默认 `%USERPROFILE%\VisionStudio`，通常为 `C:\Users\用户名\VisionStudio`，可自选专用目录，无需管理员。固定 AppId 保持旧版升级兼容；拒绝盘根、用户根、Users 根及含未知文件的现有目录。

个人配置、记录、结果、录像、评估报告和运行环境默认放在 `%LOCALAPPDATA%\vision-studio`，`VISION_STUDIO_DATA_DIR` 可覆盖。CPU 缓存为 `runtimes/cpu/<运行包SHA256>`；下载有进度、取消、重试和跨进程锁。系统 PowerShell/.NET 解压检查路径、重复成员、链接、大小和每文件哈希，实际依赖自检通过后才写入完成标记并发布环境。取消会结束解压与其 Python 子进程。

在线包减少首次下载量，补齐全部功能后的占用接近离线包。兼容的内置环境或已有缓存可复用。显式 `VISION_STUDIO_PYTHON` 继续优先，无效配置会报错，不静默使用系统 Python。

升级仅按旧清单的路径、大小、SHA256 清理过期且未修改的文件；未知或修改过的文件保留。新卸载记录只跟踪新交付文件，个人数据与按需下载环境保留。

NVIDIA 环境继续独立按需准备，使用 Windows CUDA 12.8 / Torch 2.9.1 / ONNX Runtime 1.23.2 哈希锁，不安装或修改驱动。在线包先准备 CPU 环境。源码获取说明在安装目录 `SOURCE-DOWNLOAD.txt`，安装器不会下载源码；原始许可保留。安装器没有商业代码签名。

## 重建

使用 Qt 6.8.3 MSVC SDK（WebEngineWidgets、WebChannel、Svg 及配套模块）、兼容的动态 OpenCV、Inno Setup 7.1 和可搬移 CPython 3.11 x64。原始构建输入见 `BUILD-INPUTS.json`，CPU 精确依赖见 `requirements-cpu.lock.txt`，GPU 版本与 wheel 哈希另有锁。

运行组件仅移除明确的头文件、已知测试目录、缓存、调试符号和带构建机路径的 Scripts 启动器。不得删掉实际参与推理、转换、评估和 GPU 准备的 Python 模块、模型架构 YAML、pip 或依赖元数据。

```powershell
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=D:/Qt/6.8.3/msvc2022_64 `
  -DOpenCV_DIR=D:/Opencv-C++/opencv/build/x64/vc16/lib `
  -DVISION_RESOURCE_FALLBACK=.
cmake --build build-windows --parallel 4
python packaging/windows/build_windows.py `
  --binary build-windows/bin/vision-studio.exe --runtime build-windows/python `
  --qt-prefix D:/Qt/6.8.3/msvc2022_64 `
  --opencv-bin D:/Opencv-C++/opencv/build/x64/vc16/bin `
  --msvc-redist "C:/path/to/VC/Redist/MSVC/version/x64/Microsoft.VC145.CRT" `
  --iscc build-windows/tools/InnoSetup7/ISCC.exe `
  --ffmpeg-source-dir third-party/ffmpeg `
  --profile offline --stage output/windows-2.9.1/stage-offline `
  --output output/windows-2.9.1
```

再以 `--profile online` 和新目录 `--stage .../stage-online` 执行相同命令。两种配置复用 `components/cpu` 和 CPU ZIP，锁不一致时拒绝复用。暂存目录必须为新目录，不自动删除已有目录。`--stage-only` 仅准备文件；`--skip-source` 可用于先验收 EXE，再独立冻结源码。

独立源码 ZIP 包含应用、依赖锁、重建材料、通知、`WINDOWS-SOURCE-SNAPSHOT.json` 及 `third-party/ffmpeg/` 两份原归档。FFmpeg 4.4.5 / 4.4.6 按 `windows-native-sources.json` 校验；共享上游对应源码使用同版原完整源码包，精确引用见 `upstream-source-reference.json`。源码 ZIP 不自包含当前 Windows QA，避免 QA 绑定其哈希导致递归；这些报告从独立源码标签取得。无 `.git` 时仍可依据快照重建。

## 验收与发布

分别验收两个最终安装器：默认与自选目录、中文和空格路径、清单哈希、真实 ONNX/PT、四种模型视图、转换后推理、数据集评估／阈值／报告／恢复、临时图片拖入、录像、升级和卸载保留个人数据。CUDA 在实际 NVIDIA 硬件验收。在线包另测未准备环境的 ONNX、真实下载／解压／自检及缓存离线复用。

```powershell
python -m unittest discover -s tests -p windows_packaging_tests.py -v
python -m unittest discover -s tests -p windows_release_tests.py -v
python scripts/github_release.py append-windows --version 2.9.1 `
  --release-dir output/windows-2.9.1 --notes-file output/windows-2.9.1/release-notes.md --dry-run
```

`windows-package-qa.json` schema 2 绑定两个 EXE、CPU ZIP 与源码 ZIP；逐包报告是 `installer-offline-qa.json`、`installer-online-qa.json`，组件报告为 `cpu-runtime-qa.json`。实际门禁通过后才正式 `append-windows`，再 `verify-windows`，同名文件绝不覆盖。证据随仓库提交，通过 `windows/X.Y.Z-1` 追溯；原 Linux 资产、校验清单及版本标签保持冻结。规则见[版本管理](../../docs/版本管理.md)。
