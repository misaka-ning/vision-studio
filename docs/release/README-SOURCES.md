# Vision Studio 2.1.0 完整源码包

维护者：misaka_ning <1468549029@qq.com>。新代码按 AGPL-3.0-only 发布，第三方保留原始许可。

`vision-studio-2.1.0-sources.tar.xz` 包含应用源码、测试、构建与打包脚本、UI 资源、文档、许可、示例模型和 YOLOv5 兼容源码，不包含用户设置、模型输出、训练资料、虚拟环境或缓存。完整包中的 `sources/` 保存 18 份原始上游源码／构建输入，具体身份见 `SOURCE-INVENTORY.json`，内部校验见 `SOURCE-SHA256SUMS`。

原 15 份材料保持字节，包括 Qt 6.8.3 八模块、Netron 9.3.1、Ultralytics、THOP、OpenCV、wheel 构建源码／配方和 FFmpeg；Qt 模块为 qtbase、qtsvg、qtwayland、qtimageformats、qtdeclarative、qtwebchannel、qtpositioning、qtwebengine，后者包含 Chromium。新增加 ONNX 1.17.0 源发行、Python protobuf 6.33.0 对应完整 v33.0 源码、ONNX wheel 静态 C++ protobuf 3.21.12 源码。原始 ONNX、pybind11 与两份 protobuf LICENSE 保存于应用 `packaging/licenses/model-conversion/`。

本轮为模型库右键管理与后台 PT → ONNX／TorchScript 转换，包含显示名称与备注持久化、保留文件的列表移除、后台取消和输出保护，以及转换元数据入库后自动应用任务、类别、输入尺寸和通道的处理。此前推理、Fluent 界面、批量回看、灰度、左右目、录制与停止机制保持；本版不是只替换旧包的版本字符串，不能由旧应用源码归档替代。应用版本 `2.1.0`，Debian 包版本 `2.1.0-1`。

发行 CPU runtime 的 59 项版本由实际 dist-info 生成，包含 PyTorch 2.9.1+cpu、headless OpenCV、Netron、ONNX 1.17.0 和 Python protobuf 6.33.0；正式重建使用 `docs/release/requirements-release.lock.txt` 配合 `pip --no-deps`。开发依赖锁保留开发 OpenCV 配置，不能替代正式运行时锁。ONNX 内静态 C++ protobuf 3.21.12 与 Python protobuf 6.33.0 是不同负载，不表示 Python 环境同时安装两版。

可选 GPU 脚本、探针、完整版本／SHA256 锁和 C++ ONNX Runtime API 头文件及 MIT 通知保留。CUDA wheel 不随 DEB 或完整源码包分发，由用户配置到个人目录，保留各自上游通知与许可；GPU 环境不与离线 CPU／Netron／转换环境混淆。

在完整包根目录先核验，再解压应用源码并关联旁边的上游目录：

```bash
sha256sum -c SOURCE-SHA256SUMS
tar -xf vision-studio-2.1.0-sources.tar.xz
cd vision-studio-2.1.0
ln -s ../sources sources
```

阅读应用 `README.md` 和 `docs/release/对应源码与重建.md`。外层 `docs/` 也提供发行说明、许可摘要与重建材料。Qt／Chromium 原 SDK 和源码通知保持，Python 清单按本版实际环境重收，文件身份由清单及哈希确定。

仅提供三个手动附件：`vision-studio_2.1.0-1_amd64.deb`、`vision-studio-2.1.0-complete-source.tar.xz`、`SHA256SUMS`。GitHub 自动 Source code ZIP／tar.gz 不含完整上游材料，不能替代对应源码包。QA 与原始日志保存在仓库 `docs/releases/evidence/2.1.0/`，未执行检查不声明通过，归档生成后的独立源码核验报告随标签保存并作为发布门禁，不零散上传附件。
