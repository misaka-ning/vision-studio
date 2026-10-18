# Windows 安装包

Windows 包以应用 2.1.0 为功能基线，生成独立的 `vision-studio-2.1.0-windows-x64-setup.exe`，不覆盖已公开的 Linux DEB、源码包、校验清单或标签。目标为 Windows 10 / 11 x64；本机验收以报告中的实际系统为准。

## 安装与分发

把一个安装 EXE 发给接收者即可安装；CPU 推理、PT、Netron 与模型转换不要求接收者另装 Qt、Python、OpenCV 或开发工具。默认位置为 `%USERPROFILE%\VisionStudio`，通常是 `C:\Users\用户名\VisionStudio`。安装向导保留目录选择页，创建开始菜单入口及可选桌面快捷方式，以当前用户权限运行。

个人配置、历史、结果、录像、转换文件与可选 GPU 环境默认放在 `%LOCALAPPDATA%\vision-studio`。卸载仅删除安装器记录的程序文件，保留个人数据。`VISION_STUDIO_DATA_DIR` 仍可指定其他数据位置。安装目录必须是专用目录；安装器拒绝将程序直接混入驱动器根目录、用户根目录或已有其他文件的目录。

GPU 和 Linux 发行版一样按需准备。首次点击「更多 → 准备 GPU 支持」需联网下载 Windows CUDA 12.8 / PyTorch 2.9.1 / ONNX Runtime 1.23.2 的固定 wheel，按 `requirements-gpu-windows.lock.txt` 核验 SHA256。程序不安装显卡驱动，CPU 环境不受影响。下载与真实 CUDA 自检通过后才发布用户 GPU 环境；此前的已知环境保留为明确的恢复副本。Windows Job Object 在准备进程退出时结束 pip 子进程，避免取消后继续安装。

安装器没有商业代码签名。Windows 应用源码快照随包安装到 `sources/`，交付目录另附相同 ZIP，分发一个 EXE 即携带本次 Windows 应用源码。Qt 6.8.3、WebEngine、Netron 等已有第三方对应源码沿用 2.1.0 Release 的原始完整源码归档，地址与原始 SHA256 记录在 `upstream-source-reference.json`。原始许可随安装包保存。`SHA256SUMS` 记录最终交付文件，验收日志保存于 `docs/releases/evidence/2.1.0/windows/`。

## 重建

先用 MSVC x64 构建应用，Qt 6.8.3 SDK 必须包含 WebEngineWidgets、WebChannel、Svg 和配套模块；OpenCV Windows 库的 ABI 须兼容所用 MSVC。Qt 与 OpenCV C++ DLL 使用动态链接。

准备可搬移的 CPython 3.11 Windows x64 环境；本次使用 Python Build Standalone 20261013 的 CPython 3.11.17，官方归档的 SHA256 保存在构建输入清单中。按照交付包的 `requirements-windows.lock.txt` 使用 `--no-deps` 安装，PyTorch CPU wheel 使用官方 CPU 源。使用 headless Python OpenCV，避免将其他 Qt 运行时带入应用。

在仓库根目录执行以下命令，将路径替换为实际的构建输入。`--stage` 必须是新目录，脚本不会自动删除已有暂存目录。

```powershell
python packaging/windows/build_windows.py `
  --binary build-windows/bin/vision-studio.exe `
  --runtime build-windows/python `
  --qt-prefix D:/Qt/6.8.3/msvc2022_64 `
  --opencv-bin D:/Opencv-C++/opencv/build/x64/vc16/bin `
  --msvc-redist "C:/path/to/VC/Redist/MSVC/version/x64/Microsoft.VC145.CRT" `
  --iscc build-windows/tools/InnoSetup7/ISCC.exe `
  --ffmpeg-source-dir build-windows/downloads/ffmpeg-sources `
  --stage output/windows-2.1.0/stage `
  --output output/windows-2.1.0
```

`build_windows.py` 部署 Qt / WebEngine 与 OpenCV、应用本地 VC runtime、可搬移 Python、模型、资产、脚本、文档和原始许可，生成文件 SHA256 清单，再调用 Inno Setup。安装后 `bin/vision-studio.exe` 从相邻安装标记定位自身运行环境，不依赖构建机器路径和系统 PATH。

Windows C++ OpenCV 4.12.0 的视频插件标识 FFmpeg 4.4.6，Python OpenCV 4.11.0.86 的视频插件标识 FFmpeg 4.4.5；两份原始源码一并放在安装目录的 `sources/ffmpeg/`。构建前按 `windows-native-sources.json` 的官方 URL 下载到 `--ffmpeg-source-dir`，脚本核验已冻结的 SHA256，并保存 Windows SDK 自带的完整 OpenCV 第三方通知。应用源码 ZIP 无需 `.git` 即可重新构建。

使用支持长路径的 Inno Setup 7.1 或更新兼容版本，以保留 WebEngine 原始许可目录。简体中文消息来自官方 `jrsoftware/issrc` 仓库的 `Files/Languages/ChineseSimplified.isl`，保留原维护者说明。编译器本身不进入安装包。可选 GPU 的固定 wheel 来源与哈希保存在 `gpu-wheel-sources.json`，更新时通过 `lock_gpu.py` 从官方索引重新解析并人工核对版本。

## 验收

对最终安装包至少核验：安装到中文和含空格的目录、默认目录展开、开始菜单／桌面入口、隔离开发环境后的 ONNX／PT 推理、模型三种视图、ONNX／TorchScript 导出后实际推理、录像解码、升级与卸载保留用户数据、文件清单与安装包 SHA256。GPU 单独在 NVIDIA 硬件上实测，不用 CPU 结果代替。

当前 Windows 打包流程独立于 Linux `github_release.py` 的三个附件规则；若后续要将 Windows 包纳入正式公开发行，需要同步维护多平台发布规则，不能修改历史核心附件或复用同名资产覆盖内容。
