# 2.3.1 隔离安装验收素材

`namespace-artifacts.zip` 保存最终 DEB 隔离安装验收的 49 个原始小文件，共 5,780,525 字节：安装／卸载与各检查日志、CPU 推理与模型显示／转换报告、12 张截图和标注结果，以及 PNG／JSON／CSV 导出。逐文件大小、SHA256、ZIP 哈希和素材来源见 `namespace-artifacts.json`；ZIP 成员 CRC 与逐字节 SHA256 已复核。原始验收报告为同目录的 `install-qa.json`。

验收在私有 bubblewrap 用户、进程和断网命名空间执行，安装时仅私有 dpkg 数据库与暂存安装树可写；推理以 UID 1000 运行，安装目录只读。宿主文件系统只读，宿主 dpkg 数据库和应用集成入口保持不变。本项证明隔离安装、运行及卸载，不代表已经在宿主系统安装新版。卸载后私有用户结果仍保留。

实际通过的检查包含：ONNX／PT CPU 真实推理，ONNX／PT 的三种模型显示和缓存，PT → ONNX／TorchScript 导出后真实推理，以及 GPU 未准备状态的明确检测。CPU 包验收不代替单独的最终 DEB CUDA 验收。

素材只使用随应用交付的公开示例图片 `bus.jpg` 和示例模型，属于专用 QA 输出，未包含维护者的私人数据集、设置或运行历史。归档明确排除 `history.json`、runtime、`input.deb`、转换后的模型文件及缓存／配置目录；完整模型来源与许可仍以发行源码通知为准。

12 张 PNG 已用 Pillow 完整解码。另打开 `user/onnx/workbench.png` 和 `user/model_display/model-hierarchy.png` 查看真实像素：2.3.1 版本、数据集按钮与相邻箭头、公开示例标注和目标表，以及深色 ONNX 层级树均可见且文字可读。其余图片只完成解码，未声称逐张视觉审阅；这些检查也不代替完整交互测试。

没有物理摄像头或 Windows 2.3.1 验收。本归档不包含任何性能保证或对私人数据集模型能力的结论。
