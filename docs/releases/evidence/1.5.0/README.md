# Vision Studio V1.5.0

本版增加模型显示的「结构图」「层级树」「参数表」，共用模型库中的全局选择与一次解析缓存。树与表支持搜索、只读详情和张量形状；元素数按整数大小排序。修复 Netron 嵌套模块的白底浅字，保留不同层类型的色块，选中描边使用青绿色。

普通 PT 的树表示模块包含关系；只有权重的文件按参数名称分组。完整计算连接以文件实际提供的结构为准，ONNX 更适合查看运算图。

## 安装

目标平台为 Ubuntu 22.04 LTS amd64，系统级安装目录为 `/opt/VisionStudio`。设置和结果保存在各自的用户目录。

```sh
sudo apt install ./vision-studio_1.5.0-1_amd64.deb
vision-studio
```

主文件夹中的便携应用也已更新，可使用 `/home/misaka/VisionStudio/run.sh` 启动。已有设置与运行记录保留。

## 发布文件

发布时一起提供以下三个文件：

- `vision-studio_1.5.0-1_amd64.deb`
- `vision-studio-1.5.0-complete-source.tar.xz`
- `SHA256SUMS`

应用代码采用 AGPL-3.0-only；第三方许可与对应源码随发行资料提供。维护者为 misaka_ning <1468549029@qq.com>。

```sh
sha256sum -c SHA256SUMS
```

## 验收与预览

六套 CTest 全部通过。`ctest-qa.json`、`metadata-boundary-qa.json`、`desktop-qa.json`、`home-qa.json`、`qa-report.json`、`install-qa.json` 记录各项独立验收；正式包的检查以对应报告内的 `success` 字段为准。UI 截图在 `ui/` 中。源码内部文件使用 `SOURCE-SHA256SUMS` 复核。
