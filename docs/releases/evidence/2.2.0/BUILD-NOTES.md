# 2.2.0 构建记录

目标环境：Ubuntu 22.04.5 amd64，GCC 11，Qt 6.8.3，C++17，Release/Ninja。

```bash
cmake -S . -B /tmp/vision-studio-2.2.0-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/misaka/Qt/6.8.3/gcc_64 -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/opt/VisionStudio/runtime/bin/python
cmake --build /tmp/vision-studio-2.2.0-build --parallel 4
```

初次编译新测试时缺少 `core/gpuruntime.h` 声明，已补 include 后重新构建；不是运行时故障。新增 PT 检查使测试工作量增加，完整 PT suite 超时上限调整为 300 秒。可选 Vulkan 头文件和 CUPS 查找未安装不阻断本应用构建；完整配置输出随本目录保存。

实际最终构建和测试结果以原始日志及后续验收报告为准。预检 core/PT 日志只覆盖当时执行的用例，不替代最终完整构建验收。
