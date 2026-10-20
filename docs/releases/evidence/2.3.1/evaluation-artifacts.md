# 2.3.1 最终包公开数据集评估素材

`evaluation-artifacts.zip` 保存 192 个原始小文件，合计 13,196,543 字节，ZIP 为 8,899,082 字节。逐文件来源、大小、SHA256、ZIP 哈希与完整解码结果位于 `evaluation-artifacts.json`。ZIP 成员 CRC 和逐文件 SHA256 已重新读取核验，24 份逐图快照校验文件与 JSON 原字节相符。归档原始总报告为 `packaged-evaluation-qa.json`，绑定最终 DEB SHA256 `1e959057337832f2dac8d950d738426ad426ea124aee7cefc925286792d19758`。

本矩阵实际运行六个最终安装包评估场景：ONNX CPU、PT CPU、TorchScript CPU、原始 ID 32 零结果 CPU、ONNX CUDA 和 PT CUDA。三个同权重后端比较为 PT／TorchScript CPU、ONNX CPU／CUDA、PT CPU／CUDA；具体数值与容差保留在原报告中。ONNX 的实际 ORT profile 一并归档，含 910 个 `CUDAExecutionProvider` 节点事件。归档动作本身没有重新执行推理。

收录内容包括每个任务的报告、低分候选 JSONL、逐图快照及 SHA256、阈值／错例 CSV、Markdown、PNG／SVG 曲线、真实应用 stdout／stderr、原始请求、smoke 报告、截图，以及当前图片 PNG／JSON／CSV 导出。TorchScript 转换只保留原始 stdout JSONL 与 stderr，清单标明从 stdout 结果行解析出的元信息；没有伪称存在独立的原始转换 JSON 文件。

数据仅为公开 COCO8 的 4 张 Val 图片，来源及配置原件保留在 ZIP 的 `public-dataset/`。这是小规模管线与后端一致性验收，不是足球准确率、部署基准或泛化能力测试。原始类别 32 为 sports ball，COCO8 Val 中没有该类 GT，零结果场景只验证无目标处理；它不能证明模型只识别足球。没有收录维护者私人数据集、个人设置或历史。

明确排除解包应用树、runtime、GPU／字体及其他缓存、配置／状态目录、原始模型和转换后的 TorchScript 文件。源码与模型通知继续通过发行包提供，无需在 QA 归档复制大型二进制。

30 张 PNG 已用 Pillow 完整解码。另目检 ONNX CUDA 的分析页与 PT CPU 的标注预览两张关键帧：深色 UI、当前目标映射、指标和 GT／预测叠加可见。分析页下方内容通过滚动查看；窄表格的长类别文本使用省略号。其余 PNG 只执行完整解码，未声称全部逐张视觉审阅。

没有 Windows 2.3.1 或物理摄像头验收；本归档也不代替文件夹入口专项交互测试或宿主安装验证。
