# Vision Studio 2.10.0 验收

模型库支持从文件管理器拖入单个或多个本地 ONNX／PT／TorchScript；导入按钮支持多选并共用校验。整个批次先验证，再按输入顺序与规范路径去重、按序追加。已有条目保留名称、备注、顺序及关联配置，仅将**去重后最后一项**全局选用；检测工作台、模型显示和转换来源同步，原文件保持引用。

生产源码冻结于 `cecd6ea12552e247b686dff090e2dbdffd4b4c93`。沿用 C++17／Qt 6.8.3、Fluent 深色界面和既有 CPU／NVIDIA GPU 后端，无新增运行依赖。

| 实际执行范围 | 结果 |
| --- | --- |
| 单轮完整回归 | 26 套；436 项 Qt（含初始化／清理）、159 项 Python；零失败／跳过／blacklisted；289 项生产输入与冻结对应源码一致。 |
| 新增与既有模型库原生回归 | 五个实际方法原始日志已重开，覆盖批量保序、单次全局选用、去重及元数据、非法整批拒绝、忙时不更换任务、导入取消、内部排序与菜单。 |
| 原生窗口重建持久化 | 实际关闭并重新构造 MainWindow，从设置恢复顺序、当前模型、名称及备注；该范围不是跨进程重启。 |
| 最终只读 DEB 模型库专项 | 21 项断言、14 组实际 Qt 拖放事件通过；公开 ONNX／PT／本版真实导出的 TorchScript 实际解析，重复项复用结构与类别检查缓存。当前全局 ONNX 实际 CPU 推理并保存 JSON／CSV／PNG。 |
| 最终包通用验收 | 本版聚合中的 CPU／CUDA、数据集及评估、报告、转换、模型显示、输入与恢复等适用门禁通过；详见链接中的实际范围。 |
| 对应源码与隔离安装 | 完整对应源码、最终 DEB／应用 ELF／辅助脚本身份绑定，隔离安装及运行验收通过。 |
| 宿主实际安装 | 等待系统管理员认证，未计为通过。 |

包专项 `restart_executed=false`：它验证实际设置写入，没有关闭并重启应用进程；原生窗口重建测试另行记录。转换／评估元数据标记仅验证记录保留，不冒充实际转换或评估。专项的 PT／TorchScript 完成实际解析，末尾实际推理是 ONNX CPU；不据此声明全部导入模型均已执行推理。

专项使用公开模型、公开测试图及本版实际转换出的 TorchScript，原文件与隔离副本摘要不变；无效混合输入整批拒绝，忙时当前任务与模型不变。记录验证工作流与导出一致性，不是精度、泛化或速度基准。原始日志、报告、设置与截图保存在本地 Git 外；公开目录保留审阅摘要与原件 SHA256 承诺，不公开私人资料。开发定向复验不重复计入上述完整回归数量。

- [完整回归](reports/ctest-qa.json)、[最终聚合](reports/finalize-reports.json)
- [最终包模型库专项](reports/packaged-model-library-drop-qa.json)、[原生与包联合结果](reports/model-library-drop-qa.json)
- [对应源码](reports/source-bundle-qa.json)、[隔离安装](reports/install-qa.json)
- [最终资产](final-assets.json)、[校验清单](SHA256SUMS)、[原件承诺](original-commitments.json)、[门禁承诺](raw-gate-commitments.json)

| 冻结产物 | SHA256 |
| --- | --- |
| `vision-studio_2.10.0-1_amd64.deb` | `5161f54ddbb40ee177426627149582c357285d525ac9854c5fe5223a8a07759c` |
| 应用 ELF | `b2ea7886cf8f930a1f9bb586eae6362610d8a5fc83e0c99283306e506dabcb40` |
| `scripts/pt_worker.py` | `e246524b28a98e3ad2db328dc5b687c7b2cc62748957b63b7d7644e89ac95162` |
| `vision-studio-2.10.0-complete-source.tar.xz` | `b345fd81a3173fd4650f3d492db9f6eb2fc68d6e2907104a9c8096437c931a73` |

Ubuntu 22.04 amd64；系统安装位置 `/opt/VisionStudio`。Windows 已发布补充版仍为 2.9.1，本次不声明 Windows 2.10.0 已验收或发布。

远端发布与旧发行保留使用独立实际凭据。本文件的生成工具没有执行或验证网络发布、Latest、远端资产或清理，因此不以本地验收与安装推定这些步骤完成；历史标签与资产保持。

验收工具首次启动的日志重名保护在测试开始前中止；原件保留，之后单轮完整回归通过。模型库专项首轮因验收器只读 stdout 而漏读 Qt 写入 stderr 的结果，旧失败报告保留；只修正 Git 外日志解析后，在新隔离目录对同一最终 DEB 实际重跑通过，未修改产品或放宽断言。见[工具观察记录](qa-harness-observations.json)。

[开发定向检查](dev-model-library-drop.json)、[开发 smoke](dev-drop-smoke.json)与[48项发布工具检查](release-tools-tests.json)单独保存，不重复计入正式完整回归。

## 发布与保留

2.10.0 已正式发布为 [v2.10.0](https://github.com/misaka-ning/vision-studio/releases/tag/v2.10.0)，发行提交为 `72802d410dc7f76dc8a242f514b172e86a3ff8b7`。官方 Latest 端点确认它为最新正式版；[发布凭证](publication.json)、[二次远端核验](remote-verification.json)和 [Latest 快照](latest.json)记录三个附件的大小与 GitHub REST SHA256，与冻结文件一致。历史标签与附件未覆盖或移动。

在发行提交同步主要检出与当前工作树，并逐字节核对两处完整发行目录的 64 个文件，八个用户源码文件保持，见[源码及发行副本核验](source-sync.json)。这些是发行时刻的记录，本次后续维护只补充凭据，不改变安装包或冻结生产源码。

远端复核后完成五处精确旧副本清理：2.9.1 的独立构建、源码导出、DEB 暂存，以及两处 2.9.1 发行目录；两处发行根均只保留 2.10.0 与 2.9.2。完整旧构建备份及 86 对原始 QA／元数据保全，其他 97 个旧输出条目、历史 QA、Windows 资料、共享环境和用户文件保持，见[实际清理记录](retention.json)。进程检查限于同 UID 的可见范围，明确保留受限身份复核与已证明退出的进程记录，不宣称全系统描述符可见。

宿主升级仍等待 Ubuntu 管理员认证，观测到的已安装版本为 `2.9.2-1`，不计为已安装 2.10.0，见[宿主待安装状态](host-install-pending.json)。隔离安装验收与真实宿主升级分别记录。

[原始交付凭据承诺](original-receipts.json)只公开整文件 SHA256；原件继续保存在本地。包内文档与完整对应源码仍为原冻结字节，不将事后记录混入已发布归档。
