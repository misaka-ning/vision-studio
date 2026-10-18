# Windows 2.1.0 补充发行核验

2026-10-18 已公开发布到 [v2.1.0 Release](https://github.com/misaka-ning/vision-studio/releases/tag/v2.1.0)，保持正式版与 Latest。Windows 包修订为 `windows.1`。

| 新增附件 | 字节 | SHA256 |
| --- | ---: | --- |
| `vision-studio-2.1.0-windows-x64-setup.exe` | 411996451 | `8bb8eb7da78235f29312aa43aa5b58c2451fae883c9761a8631ac3da2ed68858` |
| `vision-studio-2.1.0-windows-sources.zip` | 52221333 | `1ed27bc1a186094e95e56e4e4a4868c15bceaa046b981028b548b55f74f4f73a` |
| `SHA256SUMS-windows-x64` | 214 | `7cadd6002be32bcbb7149d9fcc1d85f2444937ea8488d86181e11e0ea62c1eff` |

三项远端资产均为 `uploaded`，GitHub REST SHA256 与本地一致。另以未登录 HTTP 请求从公开下载地址取得 Windows 校验清单并计算 SHA256，一致。原 Linux DEB、完整对应源码包与 `SHA256SUMS` 的 ID、大小和 SHA256 全部保持原样，原 `v2.1.0` 仍指向 `48604cdcf0316bda282854af45be7a34abcd12e1`。

Windows 实际源码以独立 ZIP 交付，追溯标签为 `windows/2.1.0-1`，提交 `14943c42a5ced385fc190304b6c4a1f0472be2c8`；源码和 13 项安装验收材料已通过 PR #8 合并到 `main`。Windows ZIP 的 3,580 个文件按原始 SHA256 核验，并与追溯标签核对；允许 Git 换行规范化，仅 `.gitattributes` 是封包后增加原始文件字节保留规则的明确例外。

Ubuntu Actions 对最终发布工具执行 92 项测试全部通过：Linux 发布与清理 48、版本元数据 11、Windows 发布安全 12、GPU 环境安全 21。Windows 本机的安装、真实 CPU/CUDA 推理和其他适用验收见 `windows-package-qa.json`、`installer-qa.json` 与原始日志；CI 不代替硬件验收。

首次上传完成后的说明精确比对发现 Windows 临时文件 LF→CRLF 转换。修复为明确写入 LF 后，仅修正相同内容的换行，未重新上传或覆盖资产；保留失败记录 `publication-first-attempt.json`。最终发布与独立复核分别记录在 `publish-result.json`、`remote-verification.json`，CI 原始日志和摘要见 `publication-ci-final.log`、`publication-ci.json`。

本次没有旧版发行目录需要清理。构建暂存、安装包、源码归档及用户资料保留；没有修改驱动或历史 Linux 发行文件。
