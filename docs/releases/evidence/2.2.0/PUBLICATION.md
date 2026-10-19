# 2.2.0 发布回执

2026-10-19 已公开发布 [Vision Studio 2.2.0](https://github.com/misaka-ning/vision-studio/releases/tag/v2.2.0)，为正式版且已设为 Latest。带说明的标签 `v2.2.0` 指向 `962710c2a568cd586b81a8fdac03f38ca6795074`，应用构建输入仍为 [source-freeze.json](source-freeze.json) 固定的 `5098cdd697bc5c392d414295a5c4234651e73825`；二者之间只增加最终验收证据。

发布工具先以草稿上传并核验文件，再公开正式版。[发布回执](release-publish.json)、[独立复核](release-verify.json) 和 [Latest 快照](latest-release.json) 分别记录发布结果、远端标签及三个资产的大小与 SHA256。全部与 [final-assets.json](final-assets.json) 中的最终文件相符。公开附件只有 DEB、完整对应源码包、SHA256SUMS 三项；GitHub 自动源码入口另外显示。

发布后的回执和本地保留记录作为仓库维护提交补充到 `main`，不移动标签、不覆盖公开附件，也不重新构建同名安装包。应用、模型与安装验收见 [FINAL-QA.md](FINAL-QA.md)；Windows 2.2.0 未构建或发布。

本机统一发行目录为 `/home/misaka/文档/ChatGPT/yolo/output/releases`。已将新发行文件逐个核对 SHA256 后保存到该目录的 `2.2.0` 子目录，见 [保存回执](local-release-normalization.json)。大归档和 DEB 使用同文件系统的硬链接保留冻结字节，运行时、SDK、工作源码不参与链接。

## 本地保留与清理

[官方工具预览](prune-preview.json) 及 [执行回执](prune-executed.json) 均重新核验 2.2.0、2.1.0 和旧 2.0.0 的远端标签、大小与 SHA256，执行仅移除旧 `output/releases/2.0.0`。统一发行目录最终只保留 `2.2.0`、`2.1.0`。

独立旧源码／构建先进行 [只读审计](old-2.0-code-audit.json)、[逐文件准备](old-build-cleanup-prepare.json)，再运行 [精确清理工具](cleanup_tools/)。工具的六项测试仅在 `/tmp` 临时树中执行，见 [原始测试日志](cleanup-selftest.log)，不计入应用回归数量。冻结的完整文件身份与 SHA256 清单为 [old-build-cleanup-plan.json.gz](old-build-cleanup-plan.json.gz)。

原始 DEB 的全部 payload/control、3229 份原应用源码、3179 份共同源码及纯生成构建均核验后，删除以下四个精确目录，见 [实际清理回执](old-build-cleanup.json)：

- `output/deb-stage/2.0.0-1`
- `output/v2.0-stable/source-inputs`
- `output/v2.0-stable/review/delivery-build`
- `output/v2.0-stable/package-source-inputs`

`package-source-inputs` 中 48 份唯一历史文档、截图和 QA 共 626211 字节，先复制并逐 SHA256 核验到 `/home/misaka/文档/ChatGPT/yolo/output/legacy-qa-preserved/2.0.0-prepublication/files`，才允许删除原副本，见 [保全清单](legacy-qa-preservation.json)。原审计、完整冻结计划和清理回执也保存在该保全目录。其余旧 QA 父目录、用户模型、设置、运行记录、录像、导出、当前工作树、Git 元数据和共享 Qt／CPU／GPU 环境保留。

四个独立目录按 inode 和外部硬链接计算的释放估计为 1977729024 字节；文件系统可用空间实际变化为 1977712640 字节，约 1.98 GB。观测值可能含并发磁盘活动；旧发行归档有其他硬链接，因此不把旧目录逻辑大小另加为已释放空间。
