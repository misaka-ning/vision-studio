# 错例回看原生 UI 补充截图

本目录保留 2026-10-21 对已冻结 **2.6.0 原生集成测试程序**独立执行的两个 CPU 用例及实际截图，供目检错例工具条。它不代替最终安装包验收，也不增加完整 20 套回归的通过数量。

实际执行 `errorReviewNavigatesUniqueImagesAtCurrentThresholdWithoutInference` 和 `zeroErrorReviewDoesNotInventErrorsOutsideCurrentImageRange`；[原始日志](native-tests.log)与 [JUnit](native-tests.junit.xml)均为 **4 PASS、0 FAIL、0 SKIP**，其中 4 包含 `initTestCase`、两个用例和 `cleanupTestCase`。用例先以确定性的 Conv 模型进行真实 CPU 评估，再验证回看没有新增推理请求；未运行 GPU。

- [有错例的实际窗口](unique-error-review-navigation/native-window.png)：阈值 0.700，错图 2／已评估 4／集合 4，当前 2/2。全宽单行工具条、计数、筛选、导航及返回评估按钮清晰，没有观察到重叠或溢出。
- [零错例的实际窗口](zero-errors-current-image-range/native-window.png)：错图 0／已评估 1／集合 4，当前 0/0；前后按钮禁用，旧预测已清除，范围外图片不冒充错例。画布沿用通用空输入占位图，顶栏计数及右侧无匹配说明标识实际空筛选状态。

两图由测试原生窗口直接保存，实际尺寸为 **1380 × 940**；没有单独截取 1260 或 1580 宽的错例状态。现有测试 harness 在初始化中将 `QCoreApplication` 版本写为 **2.5.0**，因此截图侧栏显示 2.5.0；原图保留，不将它标作最终 **2.6.0 DEB** 的实拍截图。冻结生产程序 SHA256 为 `c355cd5add789edf699ed0a9c623954a141080675b0fdc60b80748f78c5dc312`，原生测试程序 SHA256 为 `1afcb2aebd2f87d18857ac2a8608a5350d04912112146f4b69f1acaaf3fd1267`，均在执行前后匹配本版冻结快照。

[目录适配后的检查报告](error-ui-capture-qa.json)使用本目录相对素材路径。[原报告逐字节副本](original/error-ui-capture-qa.json)的 SHA256 为 `3dba15610cb1d4c5e8a9e79092ce4d35024ca23be3e326a9cbb2b39ead9c08e7`；[原始执行元数据](capture-execution.json)中的绝对命令、程序及临时路径仅记录当时来源，不表示这些程序随诊断材料交付。[逐文件清单](archive-manifest.json)绑定本目录保存的原始文件；归档没有运行新测试或改写原始日志、报告与 PNG。

两个场景子目录包含合成公开 fixture 的原报告、状态、Markdown 与 CSV。它们保留原报告分析工作点 0.3，图中的 0.7 是后续浏览阈值；回看不改写历史指标。这些小图和模型用于交互、匹配及像素验证，不提供足球精度或部署性能结论。评估页错图标签由原生断言验证，本次两张截图只目检图像预览页。
