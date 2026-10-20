# 2.4.0 原始验收素材：evaluation

公开 COCO8 Val 4 张真实图片的六个最终包 CPU／CUDA 评估场景与三组同权重容差比较。包含原始请求、stdout／stderr、候选与报告、91 点阈值表、错例、三格式导出、实际 CUDA profile、provenance 和配置。

- 文件：191 项，解压总大小 13388028 字节。
- 所有 30 张 PNG 已用 Pillow 完整解码；逐项尺寸在 JSON 清单。
- ZIP CRC、成员数量、每个成员字节数与 SHA256 已从生成后的 ZIP 再次读取核验。
- ZIP SHA256：`b1ac187c85f4756f00a190c1c27dd10311d6cb974855eb20287fae22cd408eaa`。
- 逐项清单：[evaluation-artifacts.json](evaluation-artifacts.json)；归档：[evaluation-artifacts.zip](evaluation-artifacts.zip)。

只保留小型公开验收文件；排除解包安装树、runtime、模型、大型源码、个人数据及缓存。这是已经执行的验收结果保存，不作为新增测试套数、足球准确率、吞吐量或性能承诺。原始运行路径在本机临时目录，归档后的相对路径以逐项清单为准。
