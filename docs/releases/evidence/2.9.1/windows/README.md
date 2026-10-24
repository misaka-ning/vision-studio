# 2.9.1 Windows 双安装包验收

当前处于构建与验收阶段，尚未认定双安装包发布完成。最终文件哈希、逐包结果和限制将在实际检查完成后写入同目录报告。

- 原 Linux Release 的附件、标签和说明已保存为 `release-before-windows.json`，补充发布必须保持其原值。
- 两个安装器使用独立 Windows 输入；源码、示例和历史证据不进入安装内容。
- 原生测试保留初轮失败与修复后的原始日志，不把 Linux 专用检查或缺少平台能力的检查写成通过。
- 当前主机 Windows 11 x64、Qt 6.8.3、MSVC x64、OpenCV 4.12.0。本机没有 Windows Sandbox，尚无干净虚拟机或 Windows 10 实测结论。

发布门禁要求 schema 2 的 `windows-package-qa.json`、逐包 `installer-offline-qa.json`／`installer-online-qa.json` 和 `cpu-runtime-qa.json`。报告须绑定最终文件，记录实际行为及未执行项目；QA 文件不单独上传为 Release 附件。
