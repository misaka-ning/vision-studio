# Vision Studio 2.6.0 验收记录

目标环境为 Ubuntu 22.04 amd64、Qt 6.8.3、OpenCV 4.5.4 和 NVIDIA GeForce RTX 4060。无新增运行依赖。本目录保存本版实际日志、JUnit、截图、公开报告与最终包核验，不复用旧版通过结论。

## 完整回归与新功能

一次完整执行的 **20 套回归全部通过：343 项 Qt、101 项 Python，零失败、跳过或 blacklisted**。计数来自实际日志／JUnit；Qt 计数含初始化和清理，并非 343 个独立业务场景。原件见 [ctest-qa.json](ctest-qa.json)、[qa-full](qa-full/)、[构建日志](diagnostics/build-and-launch/)。生产代码、CMake cache、应用及全部原生测试二进制在运行前后 SHA256 一致。两份实际 ORT profile 各记录 364 个 CUDA 节点事件，不以设备可用探针代替推理。

- [dataset-session-qa.json](dataset-session-qa.json)：版本／字段／大小限制，失败与取消保全，手动输入优先，目录／YAML × Train／Val／Test 恢复及审核排除，错例与阈值等 20 项汇总检查通过。实际原子读写观察为 120 次提交、8818 次完整读取；本机观察不代表断电持久性保证。
- [packaged-dataset-session-qa.json](packaged-dataset-session-qa.json)：最终包分别用两个不同进程验证 YAML 与目录入口。先真实 ONNX CPU 评估四张公开图，自然写入 JSON 并退出；随后调用生产恢复入口。原始来源、完整顺序、GT、首图像素和记录一致，无额外推理、历史或报告写入；没有预写一个假定成功的 JSON。
- 当前阈值错例按 FP／FN 图片并集去重。四图 fixture 在 0.7 下有 2 张、0.9 下有 3 张错误，回看推理请求为 0。覆盖零错、当前单图、已完成中断前缀、前后边界、阈值过期回调和缓存校验失败。
- 额外[功能预检查](diagnostics/feature-preflight/)与[错例截图检查](diagnostics/error-ui-capture/README.md)独立保留，**不重复计入完整 20 套**。后者是两项 CPU 原生 fixture，测试 harness 版本属性为 2.5，不能当作 2.6 DEB 实拍。

## 最终包与素材

以下包级报告均绑定 DEB SHA256：`0189cb2aa7819b3f0e2e98c723de110418d86df38c5a10adba750b04e50b6431`。

| 检查 | 实际结果与证据 |
|---|---|
| CPU、PT、Netron、转换与 APT 模拟 | [qa-report.json](qa-report.json)，零错误／警告 |
| 隔离安装、运行、卸载 | [install-qa.json](install-qa.json)，宿主 dpkg 状态未变；不代表宿主升级 |
| 包内 CPU／CUDA | [packaged-gpu-qa.json](packaged-gpu-qa.json)、[gpu-qa.json](gpu-qa.json)，GPU 七项门禁通过 |
| 公开集评估 | [packaged-evaluation-qa.json](packaged-evaluation-qa.json)，六用例、三组同权重比较、每组 91 阈值 |
| 文件夹入口与路径迁移 | [folder-package-qa.json](folder-package-qa.json)，十项检查通过 |
| 私人目录只读预检 | [packaged-private-precheck.json](packaged-private-precheck.json)，1223 张、813 正／410 负，16667 个文件内容未变，未推理 |
| 既有 UI／队列与转换 | [ui-queue-qa.json](ui-queue-qa.json)、[conversion-qa.json](conversion-qa.json)、[evaluation-qa.json](evaluation-qa.json) |
| 完整对应源码 | [source-bundle-qa.json](source-bundle-qa.json)、[source-freeze.json](source-freeze.json)，当前生产输入与冻结源码一致，18 份上游材料、59 项实际 CPU 分发及许可映射核验 |

小型公开原件归档为 [evaluation-artifacts.zip](evaluation-artifacts.zip)、[folder-artifacts.zip](folder-artifacts.zip)、[namespace-artifacts.zip](namespace-artifacts.zip) 和 [dataset-session-artifacts.zip](dataset-session-artifacts.zip)，各附同名前缀 JSON 成员清单与 SHA256。共 59 张 PNG 完整解码，ZIP CRC 和成员哈希核对。材料随仓库保存，不另外上传 Release 附件。

实际目检：1580×960 的最终包 CPU 预览、CUDA 评估页及两种入口恢复页，无明显溢出，GT／队列／版本可读，恢复后无旧预测。CUDA 评估截图截于异步错图统计整理中，代表该时刻状态；完整计数／导航由原生检查与实际包预览另行证明。未扩大为所有分辨率验收。

## 交付与边界

三个手动资产大小／SHA256 见 [final-assets.json](final-assets.json) 及 [SHA256SUMS](SHA256SUMS)。构建源码为合并的 `8507624689d9f409d81badd7a847b036f8646aa0`；后续证据和文档提交不改生产输入。正式发布、远端 Latest／标签／资产核验及两版保留均已完成，见[发布后凭证](post-publication/README.md)。这些材料另随 `main` 提交，不改写冻结资产或标签。

不公开私人图片、标签、配置、文件清单或实际用户路径 JSON。公开四图及合成／派生标注只验证功能与管线，不代表足球模型能力、通用精度或部署性能。Windows、物理摄像头、断电及部分关闭／磁盘错误 UI 时序未验收，见[源码审查](source-review.md)。

YOLO YAML 展示另已完成[接入设计](../../../YOLO模型YAML展示设计.md)和[公开模型静态调研](diagnostics/yolo-yaml-research.json)，**未作为本版已实现功能**。四个无关用户训练文件、个人资料和共享 SDK／runtime 保留。
