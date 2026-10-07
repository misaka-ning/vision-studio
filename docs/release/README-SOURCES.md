# Vision Studio 1.2.0 完整源码包

维护者：misaka_ning <1468549029@qq.com>。应用新代码按 AGPL-3.0-only 发布；第三方资源保留其原始许可。

此目录内 `vision-studio-1.2.0-sources.tar.xz` 包含完整应用源码、测试、构建及打包脚本、许可证、文档、示例模型和 YOLOv5 兼容源码。`sources/` 保存与本次二进制配套的 Qt 6.8.3 四个模块、Ultralytics、THOP、OpenCV 及 FFmpeg 原始源码归档和构建配方。具体文件身份见 `SOURCE-INVENTORY.json`。

解压应用源码后，可将旁边的原始归档目录关联到应用源码目录供重建 Qt 使用：

```bash
tar -xf vision-studio-1.2.0-sources.tar.xz
cd vision-studio-1.2.0
ln -s ../sources sources
```

在当前完整源码目录执行 `sha256sum -c SOURCE-SHA256SUMS`，可核验内部应用归档和各份上游源码。阅读应用目录中的 `README.md` 与 `docs/release/对应源码与重建.md`；外层 `docs/` 也提供发行说明、许可摘要及重建说明。应用源码没有包含机器缓存、个人设置或虚拟环境；发行用 Python 依赖锁记录在 `docs/release/requirements-release.lock.txt`。

发布这次版本时，将 `vision-studio_1.2.0-1_amd64.deb` 与 `vision-studio-1.2.0-complete-source.tar.xz` 一起提供，并附 `SHA256SUMS`。外层源码包保留了内部原始归档，用户不需要逐个访问上游站点才能取得本次配套源码。
