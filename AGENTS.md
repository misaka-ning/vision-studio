# Vision Studio 协作约定

## 仓库与版本

- 官方仓库：`https://github.com/misaka-ning/vision-studio`，源码主线为 `main`。
- 功能分支默认使用 `codex/` 前缀。正式版标签为 `vX.Y.Z`；预发布使用 `vX.Y.Z-beta.N`，Debian Control Version 为 `X.Y.Z~beta.N-1`，公开文件名保留安全的 `-beta.N`。CMake 的数字 `project VERSION` 保留 `X.Y.Z`，完整应用版本由 `VISION_STUDIO_APP_VERSION` 给出。
- 开始工作前读取 `README.md`、`CHANGELOG.md` 和 `docs/版本管理.md`，保留用户设置、运行记录、模型和录像。
- 当前桌面程序是 C++17 / Qt 6.8；ONNX 使用 C++ OpenCV DNN，PT 使用常驻 Python / PyTorch 后端。新增后端时保持颜色通道和单目裁剪约定一致。

## 每次版本更新的交付要求

维护者已明确授权：本项目后续每次版本更新都上传 GitHub，发布逐版安装包和源码压缩包，远端核验成功后本地只保留最新两个发行版本。将以下步骤作为版本更新的必要交付流程：

1. 更新真实的用户可见变更日志，统一源码、界面、打包与文档版本号。
2. 完成适用的测试、构建、真实模型及安装验收，不把未执行的检查写成通过。将报告和原始日志保存在 `docs/releases/evidence/X.Y.Z` 并随仓库提交，发布说明链接同版证据。
3. 提交并推送完成的源码，创建并推送带说明的发行标签；已发布标签和资产不可覆盖或移动。
4. 使用 `scripts/github_release.py` 发布同版本的 3 个手动附件：DEB、`vision-studio-X.Y.Z-complete-source.tar.xz`、`SHA256SUMS`。QA 必须通过并保留在仓库证据目录，不零散上传为 Release 附件。
5. 检查远端标签、发布状态、每个资产的大小和 SHA256；上传失败或认证失效时保留本地资料并明确报告。
6. 通过核验后使用发布工具预览并执行旧发行目录清理，保留最新两个版本。构建暂存目录按明确版本分别检查，禁止按宽泛通配符删除。

预发布必须标记 GitHub `prerelease=true`、`latest=false`，保留当前正式版的 Latest。默认 `prune` 只识别正式版本，beta 不挤掉正式版备份，也不自动删除；若按最新两版要求保留 beta 与当前正式版，其他旧副本须逐版精确核验后单独清理。

此授权针对本仓库的版本发布；不涉及其他仓库、显卡驱动安装、历史重写或账号权限变更。需要身份验证时请维护者完成登录，不索取密码或访问令牌。

## 发布和清理边界

- 源码与公开示例模型进入 Git；安装包、大型对应源码归档、runtime、构建产物和用户输出放在 Git 之外。
- GitHub 自动显示 Source code (zip) 与 Source code (tar.gz) 两个入口，本项目无法移除；它们不能替代完整对应源码包。应用拆分归档、SOURCE 清单、上游源码与原通知从完整源码包内取得，不另外上传附件。
- 发布工具的清理默认为预览；执行前必须确认远端数据完整。当前工作树、Git 元数据、共享 Qt SDK/runtime、用户文件和未备份资料不能删除。
- 只修改文档或仓库工具时可提交维护变更，不改写已冻结的应用版本安装包。
- 按用户要求整理历史 Release 附件时，先把旧 QA 和说明保存到同版仓库证据目录，再移除额外的手动附件；保留三个核心文件的原始内容和 SHA256，不重新打包或重写历史 DEB／完整源码。

具体命令和规范见 `docs/版本管理.md`。发布工具测试：`python3 -m unittest discover -s tests -p github_release_tests.py -v`。
