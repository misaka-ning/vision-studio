# Vision Studio 2.9.1 完整源码包

维护者：misaka_ning <1468549029@qq.com>。新代码按 AGPL-3.0-only 发布，第三方保留原始许可。

`vision-studio-2.9.1-sources.tar.xz` 包含应用源码、测试、构建与打包脚本、UI 资源、文档、许可、示例模型和 YOLOv5 兼容源码，不包含用户设置、模型输出、训练资料、虚拟环境或缓存。完整包中的 `sources/` 保存 18 份原始上游源码／构建输入，具体身份见 `SOURCE-INVENTORY.json`，内部校验见 `SOURCE-SHA256SUMS`。

此前 15 份材料保持字节，包括 Qt 6.8.3 八模块、Netron 9.3.1、Ultralytics、THOP、OpenCV、wheel 构建源码／配方和 FFmpeg；Qt 模块为 qtbase、qtsvg、qtwayland、qtimageformats、qtdeclarative、qtwebchannel、qtpositioning、qtwebengine，后者包含 Chromium。继续包含 ONNX 1.17.0 源发行、Python protobuf 6.33.0 对应完整 v33.0 源码、ONNX wheel 静态 C++ protobuf 3.21.12 源码。原始 ONNX、pybind11 与两份 protobuf LICENSE 保存于应用 `packaging/licenses/model-conversion/`。

本轮调整为侧栏决策布局：推荐策略、当前阈值、曲线与阈值表同页，另有能力诊断、场景分析、报告管理。保留九类评估诊断图表、六类完整场景分析、PNG／SVG／PDF 与可选 DPI、数值表及 LaTeX 表格行；支持只读导入外部脚本完整评估快照，以及无需原模型／数据集的历史快照独立重分析。来源运行时、原始类别和完整／子集范围保持，未知配置禁止采用到工作台；图像叠加仍需原图哈希一致。现代／旧版 PT 检测支持真实批量推理，ONNX／TorchScript 明确按单张执行；可调候选下限、每图框数和 PT CPU 线程。此前模型管理、转换、数据集与临时图片拖入、灰度、左右目、录制、停止与持久报告继续保留，无新增运行依赖。应用版本 `2.9.1`，Debian 包版本 `2.9.1-1`。验收状态、实测结果和限制以[同版证据](https://github.com/misaka-ning/vision-studio/blob/v2.9.1/docs/releases/evidence/2.9.1/README.md)为准；未执行检查不声明通过，旧版归档和报告不能替代本版材料。

发行 CPU runtime 的 59 项版本由实际 dist-info 生成，包含 PyTorch 2.9.1+cpu、headless OpenCV、Netron、ONNX 1.17.0 和 Python protobuf 6.33.0；正式重建使用 `docs/release/requirements-release.lock.txt` 配合 `pip --no-deps`。开发依赖锁保留开发 OpenCV 配置，不能替代正式运行时锁。ONNX 内静态 C++ protobuf 3.21.12 与 Python protobuf 6.33.0 是不同负载，不表示 Python 环境同时安装两版。

可选 GPU 脚本、探针、完整版本／SHA256 锁和 C++ ONNX Runtime API 头文件及 MIT 通知保留。CUDA wheel 不随 DEB 或完整源码包分发，由用户配置到个人目录，保留各自上游通知与许可；GPU 环境不与离线 CPU／Netron／转换环境混淆。

在完整包根目录先核验，再解压应用源码并关联旁边的上游目录：

```bash
sha256sum -c SOURCE-SHA256SUMS
tar -xf vision-studio-2.9.1-sources.tar.xz
cd vision-studio-2.9.1
ln -s ../sources sources
```

阅读应用 `README.md` 和 `docs/release/对应源码与重建.md`。外层 `docs/` 也提供发行说明、许可摘要与重建材料。Qt／Chromium 原 SDK 和源码通知保持，Python 清单按本版实际环境重收，文件身份由清单及哈希确定。

仅提供三个手动附件：`vision-studio_2.9.1-1_amd64.deb`、`vision-studio-2.9.1-complete-source.tar.xz`、`SHA256SUMS`。GitHub 自动 Source code ZIP／tar.gz 不含完整上游材料，不能替代对应源码包。QA 与原始日志保存在仓库 `docs/releases/evidence/2.9.1/`，未执行检查不声明通过，归档生成后的独立源码核验报告随标签保存并作为发布门禁，不零散上传附件。
