# 2.6.0 发布后的本地保留计划（尚未执行）

本文件是清理边界与工具说明，不是通过证据。编写时未执行删除，也未替代远端核验。当前 2.6.0 仍在构建，目录状态必须在实际清理时重新检查。

## 实际盘点

| 位置 | 2.4.0 | 2.5.0 | 2.6.0 |
|---|---|---|---|
| 当前 worktree `output/releases/` | 存在 | 存在 | 存在，发布尚未完成 |
| 当前 worktree `output/build/` | `2.4.0-release` | `2.5.0-release` | `2.6.0-release` |
| 当前 worktree `output/deb-stage/` | `2.4.0-1` | `2.5.0-1` | 构建中，实时复核 |
| 主 checkout `output/releases/` | 存在 | 存在 | 初次盘点时不存在，发布后须补齐 |
| 主 checkout 的 2.4 构建、stage、`output/v2.4.0` | 均不存在 | 不作旧版删除候选 | 不作旧版删除候选 |

当前 worktree：`/home/misaka/.codex/worktrees/ef0b/yolo`。主 checkout：`/home/misaka/文档/ChatGPT/yolo`。两个 2.4 DEB 当时各有两个硬链接；不能把两次 `du` 的值简单相加宣称可释放空间。

## 次序与门禁

1. 完成真实 2.6.0 QA、提交、不可移动标签和三个正式 Release 核心附件。分别用官方 `scripts/github_release.py verify` 重新核验 2.6.0、2.5.0、2.4.0 的标签提交、正式发布状态、资产名称、大小与 SHA256，保存新凭证。
2. 将 2.6.0 三个已核验资产复制到主 checkout 的 `output/releases/2.6.0`，逐个比对哈希。主 checkout 尚无 2.6 时，默认 prune 仍保护 2.5 与 2.4，不能声称已经完成“两版保留”。
3. 在旧 2.4 Release 文件仍在时，先预览并执行本目录的 `cleanup_verified_240_copies.py`。它需要旧 DEB 和来源快照作逐文件证明。每次执行前重新核验三个版本远端，执行所用凭证必须晚于预览，且不超过 24 小时。
4. 独立生成副本清理结束后，分别在两个 checkout 运行官方 `github_release.py prune` 预览，再 `--execute`；最高两个正式版必须恰为 2.6.0、2.5.0。不得手动 `rm -rf output/releases/2.4.0` 绕过门禁。先保全发行目录中尚未入库或另存的唯一小日志、清单与元数据；公开证据仍留在 `docs/releases/evidence/2.4.0`。

## 专用工具边界

`cleanup_verified_240_copies.py` 只允许三个固定路径：

- `output/v2.4.0/clean-source`
- `output/build/2.4.0-release`
- `output/deb-stage/2.4.0-1`

不允许参数扩展删除路径，不包含任何 Release 根目录、整个 `output/v2.4.0`、用户数据或其他版本。保留 `output/v2.4.0/local-install/settings-backup`、原始 QA、诊断和私人配置。源码按 2.4 最终 `source-freeze.json` 对 Git 归档逐文件比较（仅额外允许已知 `runtime-info.json`）；构建核对历史应用、测试二进制与 CMake cache；stage 与已核验 2.4 DEB 的 payload/control 文件、模式及链接逐项比较。

工具复用 2.5 清理脚本中的纯哈希、清单和路径函数，不调用旧的 PID 豁免、旧版 proof 或 main。工具和依赖脚本的哈希均绑定预览。预览将每个候选的完整哈希清单保存为 gzip；执行前及删除前再次比较目录身份和全部成员。不同文件系统、嵌入 Git、特殊文件、候选内挂载点、危险新文件和变化后的清单均拒绝。临时链接只记录和解除自身，不遍历外部目标；必须使用提供描述符安全 `rmtree` 的平台。

预览会先将候选中的小型日志、CMake/Testing 文本及 runtime 来源记录保存在 `output/v2.6.0/cleanup-240-unique-local-only/`。这些保全文件没有自动通过隐私审查，不得直接批量上传公开仓库。

## 用户与共享环境保护

主 checkout 的四个未跟踪文件必须仍恰为：

- `docs/足球模型微调.md`
- `scripts/evaluate_ball.py`
- `scripts/finetune_ball.py`
- `scripts/run_finetune_ball.sh`

工具在预览与执行前后比较它们的字节数与 SHA256；不读取展示正文，不 reset、不 clean、不纳入历史源码清理。出现其他未跟踪文件时停止审计。

共享 Qt SDK `/home/misaka/Qt`、其 plugins、仓库 runtime、`/opt/VisionStudio`、GPU runtime、用户模型、设置、录制、结果及 Git 元数据均不在删除白名单内。stage 内属于旧 DEB 的 Qt/plugins 仅删除该 stage 下的副本/链接，不操作 SDK 或链接目标。

每次预览和执行都重新扫描当前 `/proc` 中同用户进程的 cwd、exe、文件描述符和 maps；发现候选仍被引用或有未经审计的不可读进程即停止。不硬编码旧 PID、不按进程名放行、不读取环境变量或 agent socket。确有不可读的系统认证进程时，须先人工只读核验并提供新生成的 `--opaque-process-audit` JSON，格式为 `review_kind=manual-readonly-identity-review`、当前 `created_at_epoch` 与 `identities` 数组；每项严格含 pid、comm、starttime_ticks、完整 uid/gid 数组及 cmdline_sha256。工具逐项复核相同身份，任何 PID 复用或变化均拒绝；旧 2.5 记录不可直接沿用。

## 命令模板

以下命令只应在上述门禁完成后运行；此说明不表示命令已经执行。

```bash
python3 docs/releases/evidence/2.6.0/cleanup_verified_240_copies.py \
  --plan /home/misaka/.codex/worktrees/ef0b/yolo/output/v2.6.0/cleanup-240-preview.json \
  --report /home/misaka/.codex/worktrees/ef0b/yolo/output/v2.6.0/cleanup-240-preview.json \
  --verify-260 /absolute/path/fresh-2.6.0-verify.json \
  --verify-250 /absolute/path/fresh-2.5.0-verify.json \
  --verify-240 /absolute/path/fresh-2.4.0-verify.json
```

审阅成员清单、保全记录与进程审计后，重新获取三个版本核验凭证，用相同 `--plan`、不同且不存在的 `--report .../cleanup-240-executed.json` 追加 `--execute`。如需明确进程身份审计，两次运行都传入同一份仍有效的 `--opaque-process-audit`。所有报告拒绝覆盖；失败时保留文件和诊断，不修改门禁伪造通过。
