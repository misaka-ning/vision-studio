# Vision Studio 2.10.1 本地最终验收证据

**状态：本地最终验收完成，尚待发布／远端核验。** GitHub 标签、三附件、Latest 与本地旧版清理均不得由本页推断为已完成。当前已发布 Linux 为 2.10.0，Windows 为 2.9.1。

A3 冻结输入：`0f874c9b4028c29f11d76dda581e6532db77ec10`。本轮正式一次完整回归为 **44 套、1235 项 Qt、163 项 Python，失败／跳过／blacklist 均为 0**；[回归投影](final/reports/ctest-qa.json)及[聚合投影](final/reports/finalize-reports.json)记录实际来源承诺。

## 最终包 17 份必需报告

以下均为实际成功原件的白名单公开投影，**不是 finalizer 输入，也不替代本地原件**。原件 bytes／SHA256 见[原件承诺](final/original-commitments.json)。

| 原始报告角色 | 仓库记录 |
| --- | --- |
| `qa-report.json` | [公开投影](final/reports/qa-report.json) |
| `install-qa.json` | [公开投影](final/reports/install-qa.json) |
| `source-bundle-qa.json` | [公开投影](final/reports/source-bundle-qa.json) |
| `packaged-gpu-qa.json` | [公开投影](final/reports/packaged-gpu-qa.json) |
| `packaged-evaluation-qa.json` | [公开投影](final/reports/packaged-evaluation-qa.json) |
| `folder-package-qa.json` | [公开投影](final/reports/folder-package-qa.json) |
| `packaged-private-precheck.json` | [公开投影](final/reports/packaged-private-precheck.json) |
| `packaged-dataset-session-qa.json` | [公开投影](final/reports/packaged-dataset-session-qa.json) |
| `packaged-model-yaml-qa.json` | [公开投影](final/reports/packaged-model-yaml-qa.json) |
| `packaged-preview-drop-qa.json` | [公开投影](final/reports/packaged-preview-drop-qa.json) |
| `packaged-evaluation-features-qa.json` | [公开投影](final/reports/packaged-evaluation-features-qa.json) |
| `packaged-external-import-qa.json` | [公开投影](final/reports/packaged-external-import-qa.json) |
| `packaged-startup-preference-qa.json` | [公开投影](final/reports/packaged-startup-preference-qa.json) |
| `packaged-model-library-drop-qa.json` | [公开投影](final/reports/packaged-model-library-drop-qa.json) |
| `packaged-decision-layout-qa.json` | [公开投影](final/reports/packaged-decision-layout-qa.json) |
| `refactor-architecture-qa.json` | [公开投影](final/reports/refactor-architecture-qa.json) |
| `source-rebuild-qa.json` | [公开投影](final/reports/source-rebuild-qa.json) |

独立重建实际执行 configure、build、version、cpu_smoke 四条命令，核对真实新二进制版本、ONNX CPU 推理和 PNG／JSON／CSV；本项不宣称重建全部第三方依赖或二进制逐位可重现。

## 额外直接执行的原始日志

[补充说明](final-supplemental/README.md)与[逐套报告](final-supplemental/supplemental-direct-qa.json)保存四套相同最终二进制的直接 QTest 原件：AppServices 18、DetectionController 12、RecordingController 13、WindowShutdown 11，共 **54 项 Qt**，零失败／跳过。**这是额外重复执行，不累加正式 44 套／1235 Qt／163 Python。** 八份 stdout／stderr 从进程启动起捕获，经严格扫描及逐份审阅后原字节复制，没有脱敏改写；正式 CTest 原件另有整件承诺。

## 三个待发布本地附件

数值直接取本次实际 [final-assets.json](final/final-assets.json)，是本地冻结文件身份；不表示已上传或远端核验。

| 附件 | bytes | SHA256 |
| --- | ---: | --- |
| `vision-studio_2.10.1-1_amd64.deb` | 387178888 | `ad6bbcf79ae4a059c64cd4d89fea0d54f663781029d2e8199691f6ca7c8d894c` |
| `vision-studio-2.10.1-complete-source.tar.xz` | 929398024 | `9c2d10fd1563bb7029922c52a6e89365870cae5de8f10c0ee893f076ad8df491` |
| `SHA256SUMS` | 209 | `19524f4d4efcab8f9a183be09fbb74d7523ebba29db037b6fa82266abf241618` |

GitHub 自动生成的 Source code ZIP／tar.gz 不替代完整对应源码包。QA 不作为零散 Release 附件上传；三个手动附件始终是 DEB、完整对应源码、SHA256SUMS。

## 来源、原件与隐私

冻结源码／归档输入为 `0f874c9b4028c29f11d76dda581e6532db77ec10`（A3）。最终标签将指向只新增真实验收材料和限定状态文字的证据提交；完整源码包保留 A3 冻结整树输入，不声称与最终标签新增的文档／证据逐字节相同。A3→证据提交的最终 diff、blob SHA256 和完整生产输入一致性，在该提交创建后另行核验保全，不写回收据自身。

- [公开投影清单](final/public-projection-manifest.json)、[公开文件 SHA256SUMS](final/SHA256SUMS)、[工具身份](final/tools/dependency-hashes.json)记录投影范围及来源。
- 包含私有路径、数据集／模型／样本信息、命令环境或私有指标的完整原始 QA 留在本地，只公开允许的布尔结果、计数、固定角色及整文件 SHA256／bytes；不公开原始图片、私人模型或私有精度指标。
- 正式 CTest 日志含绝对构建路径，不能删改后称为完整原件；其全部日志／JUnit、四条重建命令的双流和执行 receipt 均有整件承诺。可选本地 `local-originals.tar.xz` 被 Git 忽略，不入库。
- A1／A2 候选被替换的理由和原件保留在阶段证据中，不覆盖失败／误判记录。48,056 条 A2 relocation 明细只保留本地及整文件承诺，不把其完整列表加入 Git。
- Windows 2.10.1 原生构建／安装／运行和真实物理摄像头未验收；公开 COCO8／合成标注核对的是流程与一致性，不是足球精度基准；本地时序观察不作性能保证。

发布与远端核验后，事后记录应在后续纯文档／新增证据提交中补充；既有标签及资产不移动、不覆盖。清理须另有实际预览、执行、远端核验与受保护文件保全凭据。

[实际界面检查](final-visual-review/README.md)保留独立重建程序与最终包的真实截图；[验收重试记录](final-attempts/README.md)说明启动偏好检查因旧原件目录保护而中止、改用新目录后通过的过程。
