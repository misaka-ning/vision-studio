# Vision Studio 2.0.0 完整源码包

维护者：misaka_ning <1468549029@qq.com>。应用新代码按 AGPL-3.0-only 发布；第三方资源保留其原始许可。

此目录内 `vision-studio-2.0.0-sources.tar.xz` 包含完整应用源码、测试、构建及打包脚本、许可证、文档、示例模型和 YOLOv5 兼容源码。`sources/` 保留与离线 CPU 二进制配套的 15 份原始上游源码与构建输入：Qt 6.8.3 八个模块、Netron 9.3.1、Ultralytics、THOP、OpenCV、OpenCV wheel 构建项目及 FFmpeg；同时提供构建配方。八个 Qt 模块为 qtbase、qtsvg、qtwayland、qtimageformats、qtdeclarative、qtwebchannel、qtpositioning、qtwebengine；QtWebEngine 原始归档包含对应 Chromium 源码。具体文件身份见 `SOURCE-INVENTORY.json`。

本版为 2.0 正式发行，新增参数滚轮保护、批量停止响应改进、保存目录入口与模型库排序持久化，并保留 Fluent 界面及批量结果回看，沿用 1.6.0 的推理、GPU 配置、依赖和 15 份上游源码／构建输入，保持原始上游字节不变。应用归档包含受限图片结果交付与取消、后台结果缓存／标注／自动导出／历史写入、模型排序保存、参数滚轮保护、回看与清理处理、正式版本元数据、发布保护与相应回归测试；配套应用归档、源码清单和校验文件按 2.0.0 重新生成，不能用旧 Beta 应用源码包替代。应用版本为 `2.0.0`，Debian 包版本为 `2.0.0-1`。

GPU 配置脚本为 `scripts/gpu_setup.py`、`scripts/gpu_probe.py` 和 `scripts/setup_gpu.sh`；`requirements-gpu.txt` 与 `requirements-gpu.lock.txt` 固定完整版本和官方 wheel SHA256。`vendor/onnxruntime/` 保留 C++ 使用的官方 C API 头文件、来源记录及 MIT 原始许可。可选环境使用 PyTorch `2.9.1+cu128`、Torchvision `0.24.1+cu128` 和 ONNX Runtime 1.23.2。GPU 二进制不随 DEB 或完整源码包分发，由用户通过配置工具从官方源下载至个人数据目录；NVIDIA 组件保留自身许可，实际运行时另保存原始通知和许可清单。

解压应用源码后，可将旁边的原始归档目录关联到应用源码目录供重建 Qt 使用：

```bash
tar -xf vision-studio-2.0.0-sources.tar.xz
cd vision-studio-2.0.0
ln -s ../sources sources
```

在当前完整源码目录执行 `sha256sum -c SOURCE-SHA256SUMS`，可核验内部应用归档和各份上游源码。阅读应用目录中的 `README.md` 与 `docs/release/对应源码与重建.md`；外层 `docs/` 也提供发行说明、许可摘要及重建说明。应用源码没有包含机器缓存、个人设置或虚拟环境；发行用 Python 依赖锁记录在 `docs/release/requirements-release.lock.txt`。

发布这次版本时，仅提供 3 个手动附件：`vision-studio_2.0.0-1_amd64.deb`、`vision-studio-2.0.0-complete-source.tar.xz` 和 `SHA256SUMS`。GitHub 另有自动生成的 Source code (zip) / Source code (tar.gz)，不能替代完整源码包。完整包保留内部应用与上游原始归档，用户不需要逐个访问上游站点才能取得本次配套源码；正式 GPU 与安装验收证据保存于仓库 `docs/releases/evidence/2.0.0/`，不作为额外 Release 附件。
