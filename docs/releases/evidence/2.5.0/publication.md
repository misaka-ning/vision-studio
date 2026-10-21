## 正式发布与本地同步

2026-10-21，正式标签 `v2.5.0` 已推送并固定到提交 `f25961a125d23dd2d8568a9b9a3cd144b3a16e86`。[GitHub Release](https://github.com/misaka-ning/vision-studio/releases/tag/v2.5.0)已经公开，非预发布，Latest 为 2.5.0。官方工具的[发布结果](publish-result.json)、[独立远端核验](release-verify.json)与 [Latest 原始响应](release-latest.json)确认只有以下三个手动附件，大小与 GitHub REST SHA256 digest 均匹配：

| 手动附件 | 字节 | SHA256 |
| --- | ---: | --- |
| `vision-studio_2.5.0-1_amd64.deb` | 387201252 | `9abc153c5699852ec40b16cb7b261f12796a103642601be4dda47f464d07fe03` |
| `vision-studio-2.5.0-complete-source.tar.xz` | 883232864 | `6cda2f9be5dbda8e06e07868cf0225d30a1fe2d7c8f87a5fd83913da812b990f` |
| `SHA256SUMS` | 207 | `2066875a7fd67e577f330a3af478ce72230da59c3aa3f02eee9a5879d2052024` |

标签触发的 [Source and release tooling checks](https://github.com/misaka-ning/vision-studio/actions/runs/39533775673) 已完成且结论为 success，对应提交与发行标签一致；[原始 CI 响应](tag-ci-run.json)保留该记录。发行标签、三个附件及既有历史版本不覆盖、不移动。

[主检出同步凭证](primary-source-sync.json)记录本地主 `main` 从 `dfe0ac4af45b5811a8bbc3cf46527f97279e543e` 无损快进到上述发行提交；未使用 stash、reset 或 clean，已跟踪工作树保持干净，四份用户训练文件继续保留为未跟踪文件，前后 SHA256 一致。[发行文件镜像](primary-release-copy.json)记录主目录精确新增 2.5.0 发行副本的逐文件大小／哈希验证；完成后的发布凭证由后续同步单独补齐。

[本机安装状态](host-install-pending.json)仍为等待系统 Polkit 管理员验证，观测到的宿主包版本为 `2.4.0-1`，不能把已通过的隔离安装写成宿主升级完成。个人数据快照与设置备份仅保存在本地。

新旧版本远端核验通过后，官方工具[工作树清理预览](prune-worktree-preview.json)、[主检出清理预览](prune-primary-preview.json)及各自[执行](prune-worktree-executed.json)、[执行](prune-primary-executed.json)均成功，两处发行根现仅保留 **2.5.0、2.4.0**。另按[完整预览](cleanup-old-231-v2-preview.json)和[执行凭证](cleanup-old-231-v2-executed.json)逐成员重验后，精确删除 2.3.1 的干净源码导出、Release 构建和 DEB 暂存；8 份唯一日志／metadata 和三份完整成员清单已保全。首轮因两个系统认证进程不可见而安全退出、没有删除的[诊断](cleanup-old-231-executed.json)保留；第二版仅对实际核实的完整进程身份作窄化例外，其余保护不变。

[重复归档只读审计](redundant-unpublished-archive-audit.json)确认旧未发布归档的全部内容可从最终资产、已保全原测试／清单与小型重建材料恢复，之后仅移除三个精确的未发布中间文件，见[执行记录](redundant-unpublished-archive-cleanup.json)。原始失败、诊断小文件、上游材料和最终对应源码均保留；逻辑字节数不当作实测物理磁盘释放量。

[最终本地检查](final-local-state.json)确认新版及 2.4.0 源码／构建／暂存保留，旧三目录不存在，共享 Qt SDK、图像插件、CPU／GPU runtime、私人备份及 QA 资料不在清理范围。公开标签、发行核心文件与历史证据未覆盖。
