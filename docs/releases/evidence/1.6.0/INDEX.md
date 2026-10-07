# Vision Studio 1.6.0 验收证据

实际硬件为 NVIDIA GeForce RTX 4060，驱动 595.91.07，Ubuntu 22.04 amd64。CPU 和 CUDA 测试日志、CUDA 节点原始 trace、独立环境探针与最终安装包报告保存在本目录。最终包报告绑定 DEB SHA256。物理摄像头未做验收，左右拼接录像使用真实 CUDA 处理 8 帧右目灰度视频。

测试与截图来自实际程序；CI 只做不需要 GPU 的源码检查。完整大型依赖和 NVIDIA 二进制不放入 Git，依赖锁和许可由源码及用户环境保留。
