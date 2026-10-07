# Vision Studio 1.1.0

这是当前保留下来的最早完整发行包；不据此推断不存在更早的开发版本。

### 已提供功能

- C++17／Qt 6.8.3 桌面工作台，支持图片、批量图片、视频和摄像头，目标检测与单标签分类。
- C++／OpenCV DNN CPU 执行 ONNX；常驻本地 Python／PyTorch CPU 后端执行兼容的 Ultralytics、YOLOv5 和 TorchScript PT 模型。
- 模型库、运行记录、检测框与类别显示、缩放和平移、PNG／JSON／CSV 导出，以及可取消的后台推理。
- Ubuntu 22.04 amd64 系统级 DEB，安装到 `/opt/VisionStudio`，随包提供独立 CPU runtime、示例模型、中文文档、许可及完整对应源码。

### 验证

- 三组 CTest、真实 YOLO 推理与导出、普通用户离线安装／运行／卸载验收通过，用户数据和宿主系统状态保留。

依据：本版源码包中的 `docs/release/发行说明.md`、`CTest-2026-10-05.log`，以及发行目录的 README、`验收概要.md`、`qa-report.json` 和 `install-qa.json`。

## 安装与源码

适用于 Ubuntu 22.04 amd64，默认安装至 `/opt/VisionStudio`。本版采用 CPU 推理。

请下载 DEB、同版本 `complete-source.tar.xz` 和 `SHA256SUMS`，运行 `sha256sum -c SHA256SUMS` 后安装。完整源码包包含应用及对应上游源码；GitHub 自动生成的源码 ZIP 不能替代它。应用代码采用 AGPL-3.0-only，第三方组件保留各自许可。

本版由原始本地发行文件于 2026-10-07 迁移到 GitHub，原始安装包、源码包和校验值保持不变。条目日期为原本的交付／验收日期，GitHub 公开发布时间以本页为准。
