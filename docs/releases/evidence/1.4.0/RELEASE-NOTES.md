# Vision Studio 1.4.0

### 新增

- 增加内嵌 Netron 9.3.1 的「模型显示」页，通过 Qt WebEngine 查看本地模型结构。
- 模型库选择成为全局模型，同时决定检测工作台和模型显示使用的文件；选择后在后台异步预加载，进入页面直接使用缓存。
- 对不支持、损坏或没有可显示图的文件提供中文说明与重试；非 ONNX 文件显示导出 ONNX 的提示。
- 模型页面由本机回环地址提供，模型文件留在本机，结构查看不执行 `torch.load`。

### 交付

- 发行环境加入 Netron、Qt WebEngine 及其辅助进程与资源，完整源码包补充对应的 Qt 模块、Netron 和第三方通知。
- 保留灰度、左右目、录像与结果导出；完成模型图、HTTP 服务、安装及离线推理验收。

依据：本版源码包中的 `docs/release/发行说明.md`、`CTest-1.4.0-2026-10-07.log` 和 details 日志，以及 Release 中的 `qa-report.json`、`desktop-qa.json`、`install-qa.json` 和 `source-qa.json`。原本独立发行目录缺少 README，本次历史日志根据归档内说明补齐。

## 安装与源码

适用于 Ubuntu 22.04 amd64，默认安装至 `/opt/VisionStudio`。本版采用 CPU 推理。

请下载 DEB、同版本 `complete-source.tar.xz` 和 `SHA256SUMS`，运行 `sha256sum -c SHA256SUMS` 后安装。完整源码包包含应用及对应上游源码；GitHub 自动生成的源码 ZIP 不能替代它。应用代码采用 AGPL-3.0-only，第三方组件保留各自许可。

本版由原始本地发行文件于 2026-10-07 迁移到 GitHub，原始安装包、源码包和校验值保持不变。条目日期为原本的交付／验收日期，GitHub 公开发布时间以本页为准。
