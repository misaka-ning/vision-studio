# 2.2.0 最终文件验收

本报告记录最终文件的实际验收，发布状态另见同版 GitHub Release 和远端回执。源码、安装包与对应源码已冻结，不用新内容覆盖同名已发布文件。

| 项目 | 实际结果 | 原始证据 |
| --- | --- | --- |
| 完整软件回归 | 14 套；Qt 226、Python 52 项，零失败／跳过 | [ctest-qa.json](ctest-qa.json) |
| 最终包真实界面 | 10 项，实际保存 83 组 PNG/JSON/CSV | [packaged-scope-filter-qa.json](packaged-scope-filter-qa.json)、[原始日志](packaged-gui.log) |
| 61 张队列、选中第 51 张 | 整队列 61、向后 11、仅当前 1；进度与实际导出相符 | 同上 |
| 无效类别／零匹配／停止 | -1、80 行内禁止；32 成功导出零预测；第 11 张向后 51 张任务完成 1 张后停止，无后续增长并可回看 | 同上 |
| DEB 结构、APT 模拟、只读 CPU、Netron 与转换 | 通过，无警告或错误 | [qa-report.json](qa-report.json) |
| 实际离线安装／推理／卸载 | 隔离命名空间通过，宿主安装、DPKG 数据与用户资料保持 | [install-qa.json](install-qa.json) |
| 最终包 ONNX／PT，CPU／CUDA | 四条实际路径通过 | [packaged-gpu-qa.json](packaged-gpu-qa.json) |
| CUDA 节点、C1／C3、双目视频与录制 | 真实回归及最终包聚合七项通过 | [gpu-qa.json](gpu-qa.json) |
| 转换回归与只读包实际导出推理 | 完整 14 项及 ONNX／TorchScript 两条包内导出通过 | [conversion-qa.json](conversion-qa.json) |
| 对应源码 | 18 份原始上游归档；3690 份应用源码逐字节相等；DEB 中 2730 份源码／文档／许可映射，另核对 59 份 METADATA 和实际 ELF／release-info | [source-bundle-qa.json](source-bundle-qa.json) |

DEB SHA256：`8ef059801bcef7f24189cba16bdaf66551fafb3f48ceb6b042bfb12f522226ac`。三个公开附件的实际大小和 SHA256 见 [final-assets.json](final-assets.json)，全部小于 2 GiB。[source-freeze.json](source-freeze.json) 固定应用输入提交 5098cdd697bc5c392d414295a5c4234651e73825；最终报告与 GUI QA 工具随后加入 Git，不改变已冻结的运行代码与发行文件。

独立 GUI driver 只调用真实 Qt 公共控件，不链接本应用的推理或结果函数。使用最终 DEB 解包的只读树、包内 Qt/runtime 和官方示例模型，实际执行并检查原生 JSON／表格，绑定运行中的 ELF SHA256。工具源码与重建方法见 [gui-driver](gui-driver/)。[QA-ARTIFACTS.zip](QA-ARTIFACTS.zip) 保存 83 份实际 JSON/CSV 及一张实际标注图；全部原生 PNG/JSON/CSV 的 SHA256 见 [packaged-export-manifest.json](packaged-export-manifest.json)。

停止时间数值为本机回归观察，CPU 当前 forward 不能强杀；GUI driver 的计时含保存／250 ms 静默观察，不当作停止槽延迟。原始失败和有效复验映射继续保留。未测试物理摄像头和 Windows 2.2.0，不列为通过。COCO32 是多种球类的 sports ball，不表示足球专用识别。
