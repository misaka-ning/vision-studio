# 2.3.0 公开发行与本地保留

正式发行：[Vision Studio 2.3.0](https://github.com/misaka-ning/vision-studio/releases/tag/v2.3.0)，GitHub 公开时间 `2026-10-20T09:50:12Z`，已独立核验为 Latest、非草稿、非预发布，只有三个手动附件。

- 发行标签 `v2.3.0` → `05ea7df054ad9afbbe431e094e6f52fba0c51d99`，包含最终 QA；应用对应源码输入冻结于 `9dba95c0081cf1d1616c6537f6e41ad58d07722a`。
- [发布回执](release-publish.json)、[独立远端核验](release-verify.json)、[Latest 快照](latest-release.json) 分别保存实际资产 ID、大小与 SHA256。源码、验收通过 PR #13／#14 合入，后续仅追加发布／清理凭据，不移动标签或重写资产。
- 首次上传在两个大附件成功后遇到 GitHub TLS 握手超时；保留同一草稿，用 `--resume-draft` 校验已有内容、补齐校验文件后公开。错误与续传回执均保留，没有覆盖已发布内容。

主文件夹发行目录：`/home/misaka/文档/ChatGPT/yolo/output/releases/2.3.0`。大归档通过硬链接规范保存，并逐文件核验，见 [local-release-normalization.json](local-release-normalization.json)。

[prune-preview.json](prune-preview.json) 与 [prune-executed.json](prune-executed.json) 证明先核对 2.3.0、2.2.0 及旧版 2.1.0 远端备份后，才移除旧 2.1.0 发行目录。保留的正式发行版本为 2.3.0、2.2.0；所有历史远端标签与附件继续可用。

独立旧源码副本经过完整审计，仅精确删除 `v2.1.0/clean-source-verified` 及其 `.tar`，合计 193,599,559 字节。另 113 项、197,936,642 字节的唯一 QA 资料保留，当前工作树、Git 元数据、共享 Qt／CPU／GPU 环境、用户模型、设置、记录与录像均未清理。详见 [old-2.1-audit.json](old-2.1-audit.json)、[old-source-cleanup-preview.json](old-source-cleanup-preview.json)、[执行回执](old-source-cleanup.json)。安全工具的七项单测是工具测试，不当作远端验证。

[temporary-payload-cleanup.json](temporary-payload-cleanup.json) 记录远端确认后移除的三份一次性只读 DEB 解包树及一份安装测试包副本；原生 QA 结果、预测缓存、图表、日志与隔离状态留在原外层目录。

发布后又用最终只读包验证了非同 ID 映射 `模型32 → 数据集0`，原候选保持32、GT保持0/person、坐标一致。该刻意语义错配只验证 ID／坐标管线，不作为足球准确率。命令与原始文件见 [mapping-extra](mapping-extra/README.md)，原正式标签和验收报告不改写。

宿主的旧版应用安装未在本轮升级；本轮实际安装、启动、卸载验收在私有命名空间完成。新 DEB 已交付，可按使用指南升级。
