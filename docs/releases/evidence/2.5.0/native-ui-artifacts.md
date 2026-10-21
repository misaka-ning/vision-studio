# 2.5.0 原生 UI 截图归档

[native-ui-artifacts.zip](native-ui-artifacts.zip) 保存 **16 张实际 Qt 原生 PNG** 与 3 份必要诊断原始日志，共 19 个文件；大小 **1,746,830 bytes**，SHA256 `1ed452abad81c0c87ae4f7248357f30a53dd1aca7ba798442f58cf2b19c9d2f6`。[逐项清单](native-ui-artifacts.json)记录每项大小、SHA256、图片完整解码尺寸和诊断来源。仓库副本与 `output/releases/2.5.0/native-ui-artifacts.zip` 一致；ZIP CRC、逐项 SHA256 和 Pillow 完整解码均通过。

图片原件来自 `output/v2.5.0/ui-final`：8 张 CPU/CUDA 选择器上下文窗口、4 张设置 Popup、1 张独立 Panel Popup、完整预览／评估页各 1 张、2000 张队列的灰度预览 1 张。CPU/CUDA 布局状态不能代替真实显卡推理验收；主窗口没有拼接悬浮窗，Popup 原件单独保留。

单色图、标注和已知 Conv → GlobalAveragePool → Reshape 网络由公开测试源码生成。显示的框、耗时和测试指标用于 UI／匹配／数据流验证，**不是 YOLO／足球准确率或性能宣传数据**。队列的 2000 张小图也是程序生成，截图展示灰度预览，该用例没有模型推理。`lazy-queue-grayscale-preview.png` 左栏 `2.0.0` 是沿用测试框架固定的 `applicationVersion` 值；这是未修改的 2.5.0 源码测试截图，没有修图掩盖该文字。

`diagnostics/` 保留沙箱回环权限与 Popup 失败／崩溃预检、随后 3 处异步等待失败的预检，以及等待修复后的 scope 单套诊断复验。**失败或未完成运行未改写成通过，诊断不额外累加通过数量**。正式结果及聚合边界见 [验收说明](README.md)、[ctest-qa.json](ctest-qa.json) 和 [ui-queue-qa.json](ui-queue-qa.json)。

仅含生成素材与公开测试日志；未复制私人图片、训练数据、模型、配置正文、隔离设置、缓存或 runtime。日志里的本仓库与临时测试路径用于执行追溯。归档只保存既有产物，没有重跑推理、安装或修改生产／测试源码。
