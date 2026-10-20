# 2.4.0 原始验收素材：namespace

私有 bubblewrap 命名空间的真实 DPKG 安装、普通用户运行与 purge 验收。包含原始日志、推理／结构显示／转换 smoke 报告和 PNG；仅代表隔离安装，不表示宿主已升级。

- 文件：49 项，解压总大小 5863428 字节。
- 所有 12 张 PNG 已用 Pillow 完整解码；逐项尺寸在 JSON 清单。
- ZIP CRC、成员数量、每个成员字节数与 SHA256 已从生成后的 ZIP 再次读取核验。
- ZIP SHA256：`9458c504a5ab17537eadb2a867a4fe5656ca9e93e9f3bfb503b8e4166086551b`。
- 逐项清单：[namespace-artifacts.json](namespace-artifacts.json)；归档：[namespace-artifacts.zip](namespace-artifacts.zip)。

只保留小型公开验收文件；排除解包安装树、runtime、模型、大型源码、个人数据及缓存。这是已经执行的验收结果保存，不作为新增测试套数、足球准确率、吞吐量或性能承诺。原始运行路径在本机临时目录，归档后的相对路径以逐项清单为准。
