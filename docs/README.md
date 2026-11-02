# 文档中心

从[项目首页](../README.md)了解用途与下载版本；按下面的任务进入详细说明。Linux 与 Windows 的发行节奏、后端和模型支持范围以首页及同版发行记录为准。

## 使用与模型

| 我想做什么 | 阅读 |
| --- | --- |
| 安装并完成第一次检测 | [下载与快速开始](../README.md#下载与快速开始) · [首次检测步骤](使用指南.md#1-选模型完成第一次检测) |
| 校验 Windows 安装包、了解个人数据与升级 | [安装与个人数据](使用指南.md#安装与个人数据) · [Windows 双包说明](../packaging/windows/README.md#安装与分发) |
| 导入 YOLO 数据集、临时图片或视频 | [输入方式与启动恢复](使用指南.md#2-导入数据集或临时输入) |
| 选择检测范围、类别与回看结果 | [范围和类别筛选](使用指南.md#3-检测范围类别筛选与结果回看) |
| 评估目标类、比较阈值、查看错例 | [评估工作流与统计口径](使用指南.md#4-评估模型并选择阈值) |
| 查看结构图、层级树与 YOLO YAML | [模型显示](使用指南.md#5-查看结构与-yolo-yaml) |
| 转换为 ONNX／TorchScript | [操作步骤](使用指南.md#转换模型) · [输入与导出约定](模型说明.md#pt--onnx--torchscript-转换) |
| 设置 CPU／GPU、灰度和左右目 | [输入与设备](使用指南.md#7-颜色设备与流式输入) · [GPU 模型边界](模型说明.md#v16-计算设备与-gpu-模型) |
| 导出、保存录像、恢复历史报告 | [导出与个人数据](使用指南.md#8-导出个人数据与升级) · [完整使用指南](使用指南.md) |
| 确认模型能否运行、核对示例来源 | [模型兼容范围](模型说明.md#推理兼容范围)与[模型说明](模型说明.md) |
| 排查问题 | [快捷键与常见问题](使用指南.md#快捷键与常见问题) · [提交 Issue](https://github.com/misaka-ning/vision-studio/issues) |

## 开发与设计

| 主题 | 文档 |
| --- | --- |
| Linux／Windows 源码运行、Python 环境、测试与 Smoke | [开发指南](开发指南.md) |
| 改进代码、提交问题与 Pull Request | [参与贡献](../CONTRIBUTING.md) · [协作约定](../AGENTS.md) |
| 模块职责、状态归属、任务流与异步关闭 | [应用架构](architecture/应用架构.md) |
| 重构的阶段记录与行为基线 | [迁移进度](architecture/refactor-progress.md) · [行为基线](architecture/mainwindow-refactor-baseline.md) |
| 界面设计与 YAML 静态提取设计 | [UI 设计](UI设计.md) · [YOLO YAML 设计](YOLO模型YAML展示设计.md) |

设计与阶段文档保留当时的背景、提案和状态，不等同于当前功能清单；现有操作与兼容约定请先查使用指南及模型说明。

## 打包、发行与许可

| 主题 | 文档 |
| --- | --- |
| DEB 构建、部署与最终包验收 | [Linux 打包说明](../packaging/README.md) |
| Windows 在线／离线双包、运行组件、重建和验收 | [Windows 打包说明](../packaging/windows/README.md) |
| 版本、标签、发布、对应源码和旧版留存规则 | [版本管理](版本管理.md) |
| 应用及上游完整对应源码的获取与校验 | [完整源码包说明](release/README-SOURCES.md) · [对应源码与重建](release/对应源码与重建.md) |
| 应用许可证与第三方来源 | [AGPL-3.0-only](../LICENSE) · [第三方许可清单](release/第三方许可清单.md) · [原始通知](../packaging/licenses/) |

## 发行历史与验收

从[更新日志](../CHANGELOG.md)查看逐版变化，从[GitHub Releases](https://github.com/misaka-ning/vision-studio/releases)下载。历史[发行说明](releases/)与[验收证据](releases/evidence/)按版本保留：

- Linux 2.10.1：[发行说明](releases/2.10.1.md) · [实际验收与发布记录](releases/evidence/2.10.1/README.md)。
- Windows 2.9.1：[双包发行说明](releases/2.9.1-windows.md) · [实际验收与下载限制](releases/evidence/2.9.1/windows/README.md)。

测试结论只对应记录中的版本、平台、后端和输入。Windows 2.10.1、Windows 10 和物理摄像头未验收；开发、CI、示例图像和小样本流程检查不能作为通用性能或准确率保证。历史发行材料保持原样，不随本索引整理改写。
