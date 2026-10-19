# Vision Studio 2.2.0 验收证据

正式发布及远端文件复核见 [PUBLICATION.md](PUBLICATION.md)。这些发布后回执补充到 `main`，已冻结的 `v2.2.0` 标签和三个发行附件保持原样。

目标环境为 Ubuntu 22.04.5 amd64、GCC 11、C++17 / Qt 6.8.3、OpenCV DNN 4.5.4。CUDA 实测为 NVIDIA GeForce RTX 4060、驱动 595.99.02，使用已准备的用户 GPU 环境，不修改驱动。CPU runtime 为 Python 3.10 / PyTorch 2.9.1+cpu，沿用 59 项依赖，无新增运行依赖。

## 软件回归

[ctest-qa.json](ctest-qa.json) 是实际原始日志重审后的有效汇总：14 套，Qt 226 项、Python 52 项通过，0 失败、0 跳过、0 blacklist。Qt 数字包含各 suite 的初始化与清理；不将重复复验累加为新用例。另验收生产资源路径 `/opt/VisionStudio` 构建的范围、完整 UI 与停止响应。

- [范围与筛选 38 项](regression-01/vision-scope-filter-ui-tests.log)：15 种范围边界、12 个范围／筛选组合；61 张输入、64 张选第 51 张、无选择、分类、80→1 类切换、动态类别数、原始 ID、零匹配、回看、导出及外部替换身份。
- [CPU／CUDA 筛选矩阵](regression-01/vision-gpu-tests.log)：官方 ONNX、现代 PT、旧版 PT 与真实 33 类 TorchScript；全部、32、多 ID 和零匹配。TorchScript 正匹配保留 ID32；公开 bus 图片的 COCO32 为零匹配，不据此宣称足球准确率。
- 原始 [ORT profile](regression-01/artifacts/) 两份各记录 364 个 CUDA 节点事件，182 个不同 CUDA 节点；由真实执行产生。
- [后台保存与停止](regression-01/vision-batch-responsiveness-tests.log)：本机 1080p／4K 随机图自动保存时，停止后的接受数不再增长，缓存、导出及磁盘失败恢复通过。观察到停止收尾 20／30 ms、最大 UI 心跳间隔 19／16 ms；这些是局部回归观察，不是性能保证。
- [完整转换复验](conversion-final/ctest.log)：14 项通过，包括两代完整 YOLO 的 ONNX／TorchScript、CPU／CUDA、C1／C3、任务／类别元数据与持久化。
- [实际工作台截图](regression-01/artifacts/scope-filter-workbench.png) 与 README 同图，显示 64 张队列中仅第 51 张、原始类别 ID 0／5。

原始失败没有覆盖：`regression-01` 保留旧 UI 断言和失效的旧转换解释器路径；`production-ui` 保留来源测试误读宿主 2.1 helper 的问题；`conversion-replay` 保留失效 preset 后错误假设 COCO80 的旧断言。仅修正这些测试预期／运行环境后完整复验对应 suite，产品仍按新要求使参数变化后的结果失效，按模型实际类别数验证。接受的复验映射和日志 SHA256 在汇总 JSON 中。

## 最终文件与安装

最终 DEB 检查分别保存为 `qa-report.json`（结构、只读 CPU／Netron／转换与依赖）、`install-qa.json`（隔离离线 DPKG 安装／运行／卸载）、`packaged-gpu-qa.json`（最终包 CPU／CUDA）、`gpu-qa.json`（真实节点与最终包聚合）、`source-bundle-qa.json`（源码逐文件与 DEB 对应）。每份报告只在实际执行后生成；最终门禁绑定同一 DEB SHA256。

对应源码在已提交的干净源码上冻结，不带其他任务的足球微调文件或个人输出。最终安装、源码自校验与发布回执在归档冻结后加入 Git，避免报告包含自身 SHA256 的循环引用；它们不改变已冻结的运行代码、DEB 或完整源码包。原始日志和报告随仓库保存，Release 仍只上传三个核心附件。

未执行物理摄像头及 Windows 2.2.0 构建／安装，不列为通过。双目回归使用可解码的受控视频输入。详细计划与原始依赖输入哈希见 [QA-PLAN](QA-PLAN.md) 和 [input-reuse-proof.json](input-reuse-proof.json)。
