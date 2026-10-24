# 2.9.1 Windows 双安装包验收

2026-10-24 完成本机实际构建、模型、组件准备和安装验收。交付报告为 [windows-package-qa.json](windows-package-qa.json)，发布后的远端状态见主线保存的[远端核验记录](https://github.com/misaka-ning/vision-studio/blob/main/docs/releases/evidence/2.9.1/windows/publication.json)。源码使用独立带说明标签 `windows/2.9.1-1`；原 `v2.9.1` 及 Linux 三个资产保持冻结。

## 文件与占用

| 文件 | 下载字节 | MiB | 安装目录字节 |
| --- | ---: | ---: | ---: |
| 离线 EXE | 292959557 | 279.4 | 1301510982 |
| 在线 EXE | 114598006 | 109.3 | 379682685 |
| 共用 CPU ZIP | 275278287 | 262.5 | 缓存解压量 910750855 |
| 独立源码 ZIP | 173561791 | 165.5 | 不随安装器安装 |

旧 2.1.0 安装器为 411996451 字节（约 393 MiB）；离线包下载减少 28.9%，在线包首次下载减少 72.2%。安装占用为新装后逐文件求和，包含卸载器与清单，不含后来产生的个人数据或 GPU 环境。在线包补齐 CPU 环境后，运行文件占用接近离线包；在线 EXE 加 CPU ZIP 的下载总量可能高于离线 EXE。

## 实际检查与证据

- 两包默认用户目录、自选中文和空格目录、全部安装文件大小与 SHA256；目录不含源码、历史 docs、models 或测试 assets。
- 外部提供真实 YOLOv5 ONNX、YOLOv8 PT、legacy PT、bus 图片及公开 COCO8 Val 4 图。真实 CPU/CUDA 推理、ONNX/PT 四种视图、转换后推理、评估决策布局、阈值分析、SVG 报告、会话恢复、临时图片拖入、视频与录制均有原始报告。
- 在线未准备 Python 时 ONNX 可用。使用本机 HTTP 镜像执行最终 CPU ZIP 的真实下载、校验、解压、依赖自检；URL 替换仅用于 QA，大小与 SHA256 保持发布清单的固定值。停止镜像后验证离线复用。两个实际进程只下载一次。取消、锁释放、错误校验及重试由 Qt 原生测试覆盖；恶意归档和越界由真实 Windows PowerShell 测试覆盖。
- 精简 CPU 环境通过锁定本地 wheels 准备全新独立 GPU 环境，真实 PyTorch CUDA 和 ONNX CUDA MatMul 自检成功。最终安装包再用该新环境完成 CUDA 推理，未改驱动。
- [升级报告](upgrade-qa.json) 验证真实 2.1.0 升级：未修改的旧源码、文档和示例被清理；改过的旧源码/模型及用户文件保留。离线转在线保留兼容 CPU 环境，在线卸载删除其受管理文件并保留用户新增文件；在线转离线及最后卸载也保持用户资料。
- [原生汇总](native-tests-summary.json)：27 个注册测试组，26 个适用组在针对性修复及复验后通过；POSIX GPU 安装组不适用于 Windows，以真实 Windows GPU 准备验收补充。Windows 发布工具 17 项、Windows 打包保护 10 项通过。GitHub Ubuntu CI 通过发布与清理安全测试、版本元数据、Windows 发布规则及无硬件 GPU 安全测试。
- [离线最终绑定](installer-offline-qa.json)、[在线最终绑定](installer-online-qa.json) 使用完整功能验收加最终差异复验。最后运行载荷改动为 `scripts/gpu_probe.py`、`scripts/cleanup_install.ps1` 与构建来源标记；安装器还将迁移材料前移以免重复解压。清理优化保留所有原安全校验，并实际重跑旧版升级和两种包切换；两个完整原始报告和清单保留。逐文件证明其余载荷一致后，对最终 EXE 重新安装、校验全部 SHA256、使用新 GPU 环境复验 CUDA 并卸载。应用二进制 SHA256 为 `34c25860e54a2362d2dae7f5d42d57f5e7829130c492e66641ed4ac7acde8311`。

旧版迁移的旧控制器在更新前已加载 600 秒安装时限，因而记录超时；实际 Inno 子进程继续安装并成功结束。恢复原注册后，独立验收对全部 21,044 个最终载荷及原始修改文件重新校验，再完成包切换与卸载，控制窗口调整为 1800 秒。原超时报告保持在 `upgrade-final-optimized/upgrade-qa.json` 原始日志中，最终完成结果来自 `upgrade-final-complete`，并关联 `upgrade-final-resumed` 的已通过阶段。原用例在卸载后有用户文件的目录重装被正确拒绝，在线转离线改在新专用目录完成，没有把超时写成通过。

完整原始文本日志与 JSON 在 [raw-logs.tar.gz](raw-logs.tar.gz)，文件逐项哈希见 [raw-logs-index.json](raw-logs-index.json)。初轮失败、后续修复和最终结果均保留，没有把早期失败改写成通过。验收工具位于仓库 `packaging/windows/accept_*.py`，额外编排工具在 [tools](tools)。

## 范围与限制

- Windows 11 x64 本机，Qt 6.8.3、MSVC Release、OpenCV C++ 4.12.0，RTX 4060 Laptop。PATH 移除开发 SDK/Python，并审查进程加载模块未取用开发依赖；没有干净虚拟机、Windows Sandbox 或 Windows 10 实测。
- CPU 功能验收阻断外部 Python 下载代理，Netron 使用回环地址；物理网络仍连接，未将其描述为物理断网测试。
- 未测试实体摄像头。视频与录像使用真实生成的输入流、真实推理和已安装 DLL。
- 当前测试账号不提供部分符号链接与 OLE 剪贴板能力：对应 Python/Qt 个别用例明确跳过；模型图像导出和结构视图实际检查不依赖剪贴板。未修改账号权限。POSIX 信号/文件权限注入检查按平台跳过，Linux 发布工具测试由 Ubuntu CI 执行。
- 安装 QA 关闭桌面快捷方式任务，开始菜单使用隔离组，并恢复原应用注册和入口；没有冒充实际覆盖用户入口的交互验收。首次试验对默认开始菜单的影响已恢复，用户原安装、模型、数据和桌面快捷方式未删除。

原 Linux 远端基线在 [release-before-windows.json](release-before-windows.json)。QA 不单独上传 Release；普通用户只需下载一个 EXE，不需要源码 ZIP 或测试资源。

## 发布后核验与公网下载限制

五个 Windows 附件已上传，逐文件大小及 GitHub REST SHA256 均核验成功，见 [发布结果](publication.json)和[独立远端核验](remote-verification.json)。原 Linux 三个资产的 ID、大小与 digest 及原标签均未变化；[Latest 仍为正式版 v2.9.1](latest-after-windows.json)。本地[保留检查](retention-check.json)只有 2.9.1、2.1.0 两个 Windows 发行目录，无过期发行目录可删，未触及共享 SDK、原 Python/GPU 环境或用户资料。

[正式 GitHub 地址补测](public-runtime-download.json)没有完成：原系统路由在约 90.5 秒发生传输超时；不改系统代理的直连重试也未在 600 秒冒烟测试窗口内下完 262.5 MiB CPU ZIP，未进入解压和依赖自检。600 秒是此 CLI 验收的总时限，普通交互准备没有该总时限，但仍受网络可用性影响。失败过程没有发布不完整环境或修改已有环境；不得把此次公网完整准备记为通过。

发布前通过的是固定大小/SHA256 的受控镜像下载、真实解压与自检、取消重试、并发和离线复用，方法保持上文所述。访问 GitHub 不稳定时推荐全功能离线 EXE；在线包需要网络能够完整下载其运行组件。本次补测工具保留在 [tools/public_runtime_download.py](tools/public_runtime_download.py)。
