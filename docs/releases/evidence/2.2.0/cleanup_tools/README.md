# 2.0.0 旧构建与源码副本的精确清理

这是 2.2.0 发布后的单次维护工具，只接受代码中列出的四个绝对目录；不删除 `v2.0-stable` 父目录、其他历史 QA、当前工作树、共享 SDK 或用户模型。正式 Release 的清理由 `scripts/github_release.py prune` 独立完成。

## 验证与准备

`cleanup_old_2_0.py` 默认只输出预览。`--prepare` 在原 2.0.0 发行目录仍存在时进行完整校验，并将每个文件的 SHA256、大小、软链接目标及目录／文件设备号和 inode 冻结到计划：

```bash
python3 docs/releases/evidence/2.2.0/cleanup_tools/cleanup_old_2_0.py \
  --prepare \
  --plan /tmp/vision220-old-code-cleanup-plan.json \
  --release-verify output/releases/2.2.0/release-verify.json \
  --prune-receipt output/releases/2.2.0/prune-preview.json
```

本次准备重新验证了应用源码归档的 3,229 项、`source-inputs` 对应的全部内容、`package-source-inputs` 的 3,179 项相同内容及 48 项唯一内容，以及原 DEB 的完整 payload 和 Control。额外 Python 字节码已在只读审计中通过内存重构证明可逐字再生成；本工具核对该证明绑定的源码及字节码 SHA256。`review/delivery-build` 则与原审计清单逐文件比较，并核实 2.0.0 构建版本。

准备计划约 19 MB，先保存在 `/tmp`，执行后以 [压缩清单](../old-build-cleanup-plan.json.gz) 随证据归档（约 2.3 MB），不将旧运行库文件加入 Git。准备阶段不复制或删除用户目录。

本次清理已完成，四个目标目录和原 2.0.0 本地发行目录已删除。上面的 `--prepare` 命令是当次执行记录；若要复现准备，必须先从可信备份恢复四个目标快照、原应用源码归档及原 DEB，不能修改目标路径或跳过校验。已执行计划的完整副本保存在下述保全目录的 `prepared-cleanup-plan.json.gz` 中。

## 执行与保全

`--execute` 必须提供已存在的准备计划、有效的正式 2.2.0 三附件验证 JSON，以及包含 2.0.0 `verified_candidates` 的 prune 回执。回执中的保留版本必须是 2.2.0 与 2.1.0；旧资产 SHA256 必须与准备计划一致。复核后的实际执行命令：

```bash
python3 docs/releases/evidence/2.2.0/cleanup_tools/cleanup_old_2_0.py \
  --execute \
  --plan /tmp/vision220-old-code-cleanup-plan.json \
  --release-verify output/releases/2.2.0/release-verify.json \
  --prune-receipt output/releases/2.2.0/prune-executed.json
```

执行时重新计算四个目录的完整 SHA256 和身份；外部硬链接数量变化可以接受，内容、路径、inode、模式或修改时间变化均阻止清理。执行阶段不再依赖已经由正式 prune 删除的旧 DEB 与源码归档。

先将 48 个唯一文件逐个复制并验证到：

```
/home/misaka/文档/ChatGPT/yolo/output/legacy-qa-preserved/2.0.0-prepublication/
```

其中保存 `files/`、逐项 SHA256 的 `preservation-manifest.json`、原始审计、压缩的冻结计划和 `cleanup-result.json`。原审计按原 SHA256 核验；所有保全均成功后才进入删除。若该保全目录已经存在，工具拒绝覆盖，须人工检查后再决定如何继续。

删除使用固定目录描述符、`O_NOFOLLOW`、逐项 inode 验证及显式子项集合，没有通配符或跟随软链接。每删完一个目录即记录结果；报告包含精确路径、条目数量、逻辑大小、保全信息、考虑外部硬链接的 inode blocks 释放估计及文件系统空闲空间变化。空间变化可能受到并行活动和共享物理区块影响，不能解释为精确独占磁盘收益。

## 自检

```bash
python3 docs/releases/evidence/2.2.0/cleanup_tools/test_cleanup_old_2_0.py
```

六项自检全部通过，删除行为只发生在新建的 `/tmp/vision220-cleanup-selftest-*` 测试树中。覆盖 `python3.10` 与 `.qt` 路径、文件／目录软链接不跟随、意外额外文件阻断、根 inode 替换阻断、子目录 inode 替换阻断、目录替换为软链接阻断，以及硬链接计数与内容变化的区分。实际日志见 [cleanup-selftest.log](../cleanup-selftest.log)。

## 本次执行结果

完成复核后执行，四个精确目标目录全部删除，48 个唯一文件及原审计、冻结计划完整保全。inode blocks 释放估计为 **1,977,729,024 字节**，文件系统空闲空间观测增加 **1,977,712,640 字节**；完整回执随仓库证据归档，保全目录的 `cleanup-result.json` 保存逐目录结果。
