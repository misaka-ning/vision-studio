# Vision Studio Debian 发布包

此包面向 **Ubuntu 22.04 LTS、amd64（x86_64）**。Qt 6.8.3、CPU PyTorch、Ultralytics 与示例模型随包提供；系统负责 Python 3.10、OpenCV 4.5 和桌面图形基础库。首次启动与模型加载不需要下载依赖。

默认程序安装在 `/opt/VisionStudio`，菜单与命令为 `vision-studio`。模型、运行库和程序属于系统文件；设置、会话和导出文件使用当前用户的应用数据目录。卸载程序不会删除用户创建的结果。

## 安装与启动

```sh
sudo apt install ./vision-studio_1.1.0-1_amd64.deb
vision-studio
```

也可从应用菜单打开“Vision Studio 视觉模型工作台”。`apt` 会安装 control 声明的系统依赖。此包需要 Python 3.10，因此不适用于默认 Python 3.12 的 Ubuntu 24.04；其他发行版需要单独构建和验证。

## 无管理员权限构建

构建需要已有的 Release 二进制、Qt 6.8.3 SDK、经过测试的 Python 3.10 CPU 环境，以及 `dpkg-dev`、`patchelf`、`binutils`。打包只修改新 staging 目录中的副本，不修改输入目录。

```sh
python3 packaging/build_deb.py \
  --binary build/release/bin/vision-studio \
  --qt-prefix /path/to/Qt/6.8.3/gcc_64 \
  --extra-qt-plugins build/qt-imageformats-prefix/plugins \
  --runtime build/release-runtime
```

默认产物在 `output/releases/`，staging 在 `output/deb-stage/1.1.0-1/`。已存在的 staging 和 `.deb` 会被拒绝覆盖；需要重打包时指定新的 `--stage` 和 `--output`。`--install-root` 可改变应用安装目录，默认 `/opt/VisionStudio`；编译时的生产资源根也应使用相同路径。`--compression zstd` 提供另一种 Ubuntu 支持的压缩方式，默认使用 xz。

构建脚本保留 ELF 依赖闭包需要的 Qt 运行库与图片、X11、Wayland、输入法及桌面文件对话框插件，重写为 `$ORIGIN` 相对 RUNPATH，去除 staging 内应用和 Qt 的调试符号。WebP/TIFF 等补充插件通过 `--extra-qt-plugins` 加入，必须匹配 Qt 6.8.3。

发行环境使用 `opencv-python-headless`，避免 Python OpenCV 再带入另一套 Qt5。Python wheels 的原生库不重新 strip，以免损坏 auditwheel 调整过的 ELF 段布局。构建检查符号与版本 ABI，将 GNU 数学/线程运行库依赖重定向到 Jammy 系统库；OpenCV 的旧 LP64 OpenBLAS 使用系统 BLAS/LAPACK 替代，NumPy 的独立 ILP64 OpenBLAS仍保留。实际版本与全部包锁由所选发行环境元数据生成。

Python 包包含运行所需模块与授权资料，删除 pip、旧环境的 console 入口、激活脚本和字节码缓存；解释器通过 `/usr/bin/python3.10` 运行，不依赖原虚拟环境位置。重建 headless 环境时应按完整发行锁使用 `pip --no-deps`，避免 Ultralytics 的默认依赖重新安装 GUI OpenCV。`dpkg-shlibdeps` 根据 Jammy 的库元数据生成系统依赖；`dpkg-deb --root-owner-group` 写入系统所有权，无需 sudo。

## 验证与发布

包旁生成 SHA256 和 JSON manifest。发布前应运行 `packaging/verify_deb.py`，检查 ownership、RUNPATH、依赖与路径，再在解压树内以普通用户运行真实 ONNX、现代 `.pt`、旧版 YOLOv5 `.pt` 推理。安装到 `/opt` 后应用目录应保持不可写，导出结果必须写入用户目录。

```sh
dpkg-deb --info output/releases/vision-studio_1.1.0-1_amd64.deb
sha256sum -c output/releases/vision-studio_1.1.0-1_amd64.deb.sha256
```

运行环境只读取选定的本地 `.pt`，不会根据文件名下载或替换权重。可用 `VISION_STUDIO_PYTHON` 指定另一个兼容的本地环境，用 `VISION_STUDIO_DATA_DIR` 指定测试或导出数据目录。`VISION_STUDIO_QT_HOME` 可选择自行构建的兼容 Qt 6.8.3 prefix（须包含 `lib/` 与 `plugins/`）；该环境的运行库优先于包内 Qt 加载，便于修改和替换动态链接库。发布授权、模型来源及第三方许可见包内 `/usr/share/doc/vision-studio/copyright` 和 `licenses/`，以及 `docs/release/`。发布时应同时提供相应的应用和 Qt 源码资料。

参考：[dpkg-deb 官方手册](https://manpages.debian.org/bookworm/dpkg/dpkg-deb.1.en.html)、[dpkg-shlibdeps 官方手册](https://manpages.debian.org/bookworm/dpkg-dev/dpkg-shlibdeps.1.en.html)。
