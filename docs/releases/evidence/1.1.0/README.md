# Vision Studio 1.1.0 发布文件

适用于 **Ubuntu 22.04 amd64**，安装到 `/opt/VisionStudio`。包含 Qt 6.8.3、独立 CPU PyTorch 环境、本地 ONNX / PT 示例模型和中文使用说明。

## 安装

```bash
sudo apt install ./vision-studio_1.1.0-1_amd64.deb
vision-studio
```

APT 可能安装系统共享库；模型和 Python 依赖已经随包准备。设置、运行记录和默认导出写入 `${XDG_DATA_HOME:-$HOME/.local/share}/vision-studio`，正常使用无需向安装目录写入文件。当前后端均为 CPU。

## 发布与源码

对外提供以下两个主文件，并附 `SHA256SUMS`：

- `vision-studio_1.1.0-1_amd64.deb`：完整安装包。
- `vision-studio-1.1.0-complete-source.tar.xz`：完整配套源码，包含应用源码、Qt 四模块、Ultralytics、THOP、OpenCV / wheel 配方和 FFmpeg 对应源码归档。

新编写的代码按 AGPL-3.0-only 发布。维护者：**misaka_ning <1468549029@qq.com>**。模型和第三方组件保留各自许可。对应源码、原始许可证、来源和哈希均随上述源码包提供；许可目录也安装在 `/usr/share/doc/vision-studio/licenses`。

在此目录执行 `sha256sum -c SHA256SUMS` 可核验两个主文件。解压完整源码包后，在其根目录执行 `sha256sum -c SOURCE-SHA256SUMS` 可核验各份内部源码归档。`sources/` 和独立 `vision-studio-1.1.0-sources.tar.xz` 为同一配套源码的拆分文件，方便逐项验收；完整源码包已包含这些材料。

## 实测资料

- `install-qa.json`：私有命名空间实际 dpkg 安装、普通用户离线 ONNX / PT 推理与卸载；用户数据和宿主系统状态保留。
- `desktop-qa.json`：X11 真实界面推理、PNG / JSON / CSV 导出与 PNG / JPEG / WebP / TIFF 读写检查。
- `qa-report.json`：发行包权限、ELF 依赖、许可、版本、模型、来源校验及解包后实际推理审计。
- `ui-preview.png`：本次正式包运行界面截图。

最终完整包审计、对应源码检查和实际安装/离线推理/卸载均已通过，详细结论见 `验收概要.md`。验收执行边界以原始报告为准；摄像头硬件仍需在实际设备上验证。完整操作和重建说明在安装目录文档与配套源码中。
