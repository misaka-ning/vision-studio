# Vision Studio 2.7.1 验收

本次只修复深色下拉弹窗的亮色边条。一次完整22套回归实际通过：396项Qt（含初始化/清理）、101项Python，零失败/跳过/blacklisted；生产输入、应用和原生测试二进制前后SHA256一致。

[弹窗专项](popup-visual.json)使用真实Qt6.8.3主窗口对象、隔离数据目录，实际展开设备、GPU编号、颜色和检测范围四个菜单。四处上下边缘亮色像素由非零降到0，外层调色板变为`#202020`，Escape未改变选项。四张前后截图保持原始Qt渲染字节，不是概念图；该项是视觉观察，不是新增的一套推理测试。

最终DEB已实际通过CPU/CUDA、ONNX/PT/TorchScript、转换、公开四图六用例与三组同权重比较/91阈值、隔离安装/运行/卸载、完整对应源码及原有数据集/拖入/YAML验收。私人Val仅做只读预检，不作为精度测试。Windows和物理摄像头未列为本轮验收；隔离安装不等于宿主已升级。

[ctest结果](reports/ctest-qa.json)、[包验收](reports/qa-report.json)、[GPU](reports/gpu-qa.json)、[评估](reports/packaged-evaluation-qa.json)、[安装](reports/install-qa.json)、[对应源码](reports/source-bundle-qa.json)提供去标识结果。原始报告、日志、命令、完整原件和6个QA ZIP全部保留本地；为遵守先前自动审批对本机路径/进程信息公开的限制，此目录只公开白名单投影、[原件SHA256承诺](original-commitments.json)、[归档承诺](local-archive-commitments.json)与已人工检查的截图。投影不是新的QA执行、原始完整日志或finalizer输入。

开发路径预检失败以及归档工具目录参数/源码构建清单顺序的诊断均保留本地，未计入最终22套通过数量；最终完整运行和最终包验收独立通过。

构建源固定在主线`0c178ff`，后续证据维护不改生产代码或冻结资产。两个确定性源码归档在补齐最终清单前后字节一致。[最终三文件大小/SHA256](final-assets.json)及[SHA256SUMS](SHA256SUMS)绑定正式交付内容。发布后核验及保留状态另行记录。
