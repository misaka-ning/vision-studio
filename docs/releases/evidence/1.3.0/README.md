# Vision Studio 1.3.0 发布文件

适用于 **Ubuntu 22.04 amd64**，默认安装到 `/opt/VisionStudio`。包含 Qt 6.8.3、独立 CPU PyTorch 环境、本地 ONNX / PT 示例模型与中文使用说明。

本版保留四个原页面，随后依次增加「录制视频」「更多」。运行示例与导出结果移至「更多」，工作台移除这两个入口。选择灰度时，预览及导出底图使用与模型相同的灰度信息；1 通道模型直接输入灰度，3 通道模型输入三份相同灰度。单设备左右拼接画面可选择完整画面、左目或右目。

## 安装

```bash
sudo apt install ./vision-studio_1.3.0-1_amd64.deb
vision-studio
```

APT 可能安装系统共享库；模型与 Python 依赖已随包准备。设置、运行记录与默认导出位于 `${XDG_DATA_HOME:-$HOME/.local/share}/vision-studio`，录像在其中的 `recordings/`。正常使用无需写入安装目录。当前后端为 CPU。

## 录制

选择视频或摄像头并开始检测，再点击「开始录制」，再次点击「结束录制」。录像保存当前所选画面和颜色模式，并叠加检测标注。停止检测、视频结束或正常关闭窗口时会完成保存。进入「录制视频」查看、播放或导出录像，也可打开存储文件夹。

格式为 MJPEG AVI 与同名 JSON；视频沿用来源帧率，摄像头按实际录制时长设置帧率。奇数尺寸在编码时仅补齐边缘，内置播放裁回原内容。录像包含已推理的图像帧，不含音频。

## 发布与源码

对外提供以下两个主文件，并附 `SHA256SUMS`：

- `vision-studio_1.3.0-1_amd64.deb`：完整安装包。
- `vision-studio-1.3.0-complete-source.tar.xz`：完整配套源码，包含应用源码、Qt 四模块、Ultralytics、THOP、OpenCV / wheel 配方及 FFmpeg 对应源码归档。

应用代码按 AGPL-3.0-only 发布。维护者：**misaka_ning <1468549029@qq.com>**。模型和第三方组件保留各自许可。对应源码、原始许可证、来源与哈希随完整源码包提供；许可也安装在 `/usr/share/doc/vision-studio/licenses`。

在本目录执行 `sha256sum -c SHA256SUMS` 可核验两个主文件。解压完整源码包后，在其根目录执行 `sha256sum -c SOURCE-SHA256SUMS` 可核验内部源码归档。`sources/` 与独立应用源码归档为方便逐项验收而保留的拆分文件，完整源码包已经包含。

## 实测资料

- `ctest-qa.json` 与 `CTest-2026-10-06.log`：四套测试全部通过，包括实际灰度输入、16 位 PT 图像、六页导航、录像启停、保存、播放与导出。
- `qa-report.json`：包权限、ELF 依赖、版本、模型、许可、来源、源码完整性、APT 模拟与只读树实际推理审计。
- `install-qa.json`：私有离线命名空间实际 dpkg 安装、普通用户 ONNX / PT 推理与卸载，用户数据及宿主状态保留。
- `desktop-qa.json`：正式包 X11 灰度 PT 推理、PNG / JSON / CSV 导出与图像插件检查。
- `home-qa.json`：家目录便携版源码、二进制同步及彩色 / 灰度 PT 推理，原偏好与历史保留。
- `ui-preview.png`、`ui-camera.png`、`ui-recordings.png`、`ui-more.png`：本次界面截图。

详细结论见 `验收概要.md`。双目与录像使用本地合成视频验证，未进行摄像头硬件实机测试。操作与重建方法在安装文档和配套源码中。
