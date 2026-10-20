# 2.4.0 原始验收素材：folder

最终只读 DEB 的公开目录迁移专项。包含原始请求、stdout／stderr、候选与报告、阈值曲线、导出、失效绝对根目录与名单 fixture 配置；原 JPEG 与模型不重复归档。

- 文件：34 项，解压总大小 2192502 字节。
- 所有 5 张 PNG 已用 Pillow 完整解码；逐项尺寸在 JSON 清单。
- ZIP CRC、成员数量、每个成员字节数与 SHA256 已从生成后的 ZIP 再次读取核验。
- ZIP SHA256：`2167acb19654927ebd336631647ec3b44cad2abeb3ab802cea43fdda07ea48c0`。
- 逐项清单：[folder-artifacts.json](folder-artifacts.json)；归档：[folder-artifacts.zip](folder-artifacts.zip)。

只保留小型公开验收文件；排除解包安装树、runtime、模型、大型源码、个人数据及缓存。这是已经执行的验收结果保存，不作为新增测试套数、足球准确率、吞吐量或性能承诺。原始运行路径在本机临时目录，归档后的相对路径以逐项清单为准。
