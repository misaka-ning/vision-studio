# Vision Studio 2.8.0 验收

一次完整22套回归实际通过：400项Qt（含初始化／清理）、101项Python，零失败／跳过／blacklisted；生产输入、应用和原生测试二进制前后SHA256一致。另行执行的21项发行元数据／视图契约与48项发布工具测试未重复计入。

[原生界面专项](reports/native-ui-visual.json)使用最终生产UI对象连接的真实Qt6.8.3主窗口与隔离数据，9项实际检查通过：One Dark语法、等宽字体、行号／当前行、PT可用入口、ONNX隐藏且不可进入、恢复开关默认与持久、指南位置／锚点和真实Tab顺序。[六张截图](screenshots.json)字节保持，已目检无本机私人路径；该harness不冒充最终DEB或宿主安装。

最终DEB的CPU／CUDA、ONNX／PT／TorchScript、转换、公开四图六用例与三组同权重比较／91阈值、隔离安装／运行／卸载、对应源码及数据集／拖入／模型YAML均实际通过。[额外启动检查](reports/packaged-startup-preference-qa.json)在复制的真实成功会话中启动三个独立包内进程：关闭时原smoke保留success=false和exit1，空队列且未推理是预期拒绝；开启与缺少设置键时实际恢复，12项检查通过。它不把负向smoke改写为普通通过。原session、历史和公开数据保持，原件另有[SHA256承诺](startup-original-commitments.json)。

私人Val仅做只读导入预检，不是模型准确率测试。Windows、物理摄像头、断电持久性和跨机器性能未列为本轮验收；隔离安装不等于宿主已升级。更多截图使用隔离的未准备GPU状态，所显示的探针原因不代表宿主CUDA能力。

[完整回归投影](reports/ctest-qa.json)、[包](reports/qa-report.json)、[GPU](reports/gpu-qa.json)、[评估](reports/packaged-evaluation-qa.json)、[安装](reports/install-qa.json)、[对应源码](reports/source-bundle-qa.json)只公开严格允许字段。原始日志、报告、命令、身份资料及6个QA ZIP逐字节保留本地；仓库公开[原件承诺](original-commitments.json)、[归档承诺](local-archive-commitments.json)及结果投影，投影不能作为原始finalizer输入。

初次视觉检查发现全局QSS覆盖等宽字体，已修正并从最终构建重新运行完整回归；此前中断轮、旧字体画面和harness编译／交互及启动验证器前置状态判断的诊断原件保留，不计入最终通过数量。

构建来源为主线 `11bd462ef5e32903e73a1d495ab1966ccc029a34`，后续证据维护不改生产代码或冻结资产。[最终三个文件](final-assets.json)与[SHA256SUMS](SHA256SUMS)绑定交付内容；远端发布及保留清理结果在实际完成后追加。

## 宿主安装

[实际宿主升级记录](host-install.json)：旧窗口退出后通过系统权限认证升级到 `2.8.0-1`，命令行返回 `Vision Studio 2.8.0`，安装后的应用 SHA256 与最终 DEB 一致。三个个人记录文件前后 SHA256 一致；未在安装验收中自动执行用户检测／评估。原始安装日志与私人文件路径仅保留本地。
