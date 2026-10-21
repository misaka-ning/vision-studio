# Vision Studio 2.7.0 公开验收结果

本版已实际完成完整一次 22 套回归：**396 项 Qt、101 项 Python，零失败／跳过／blacklisted**。Qt 数量含初始化和清理。生产输入及应用／原生测试二进制前后 SHA256 一致；最终 DEB 的 CPU／CUDA、转换、公开评估、隔离安装／运行／卸载、完整对应源码，以及两项新功能包级检查均已通过。

## 公开资料的边界

自动审批拒绝提交包含本机绝对路径、临时目录和进程编号的原始验收目录。因此完整原件已逐字节保存在本地；本目录只公开严格选取的结果、布尔检查、数量、原件 SHA256 承诺及三张已人工目检的原样截图。这里不是新的 QA 执行，不把投影当原始完整日志或 finalizer 输入，也不上传验证器代码或原始 ZIP。

- [ctest 结果](reports/ctest-qa.json)及 [outcomes.log](outcomes.log)：后者只提取 PASS／FAIL／Totals／Ran／OK 结果行，明确区分最终完整运行与诊断，不累加预检数量。
- [新 YAML 功能](reports/model-yaml-qa.json)与[最终包三模型比对](reports/packaged-model-yaml-qa.json)：YOLOv8 的 23 层、YOLOv5 的 25 层架构逐字段逐类型与独立静态 Netron oracle 一致；ONNX 明确缺失，不生成猜测配置。四视图缓存、安全另存、源文件保护和 12 张原始截图解码均实际验证。
- [新拖入功能](reports/preview-input-qa.json)与[最终包 12 阶段工作流](reports/packaged-preview-drop-qa.json)：真实 Qt 拖入和三选框，替换／临时追加／取消、忙时拒绝、临时图 CPU 检测及所选导出、原集合评估、视频和 YAML。临时图没有 GT、不当负样本、不写入数据集或路径 JSON；评估仍只包含原集合。原生大队列及停止／三范围组合均已执行。
- [CPU 与转换](reports/qa-report.json)、[隔离安装与卸载](reports/install-qa.json)、[GPU](reports/gpu-qa.json)、[公开评估](reports/packaged-evaluation-qa.json)、[对应源码](reports/source-bundle-qa.json)均保留原成功结果与原件承诺。隔离安装不代表宿主升级，未改动宿主安装。
- 私人数据仅只读预检，未推理、未写入；原图、标签、配置及文件名单不公开。公开四图和合成 fixture 验证功能与管线，不作为足球精度或性能成绩。Windows／物理摄像头未列为本轮验收。

## 校验与源码

[原件承诺](original-commitments.json)覆盖 257 份本地保全原件。[六份本地归档承诺](local-archive-commitments.json)记录原 ZIP 的字节数／SHA256／CRC／成员及 PNG 解码计数；合计 411 成员、82 PNG，归档本身不上传。诊断中的失败与主动中断仍如实记录为 false，不当作最终通过。

投影中的 `original_file_commitment` 与 `original_success` 指本地原件；原始 SHA 不冒充投影后字节。[投影成员清单](public-projection-manifest.json)单独记录公开文件的新 SHA；[截图记录](screenshots.json)只包含经过目检并保持原字节的三张 PNG，不编辑或伪造画面。

构建源码来自已合并的 `80d4d872e83a455699f056e1c8f52aaf6f6663ef`，完整源码包与最终 DEB 的生产输入一致。后续证据及文档不改生产代码或发行资产。三个核心文件的原始大小与 SHA256 见 [final-assets.json](final-assets.json)及 [SHA256SUMS](SHA256SUMS)，不受本次公开投影影响。发布后核验与两版保留另按同版发布凭证记录。
