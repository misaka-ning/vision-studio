# Vision Studio Debian 发布包

此包面向 **Ubuntu 22.04 LTS、amd64（x86_64）**。Qt 6.8.3 / WebEngine、CPU PyTorch、Ultralytics、Netron 9.3.1 与示例模型随包提供；系统负责 Python 3.10、OpenCV 4.5 和桌面图形基础库。首次启动、推理和模型结构显示不需要下载页面、模型或 Python 依赖。

默认程序安装在 `/opt/VisionStudio`，菜单与命令为 `vision-studio`。模型、运行库和程序属于系统文件；设置、会话和导出文件使用当前用户的应用数据目录。卸载程序不会删除用户创建的结果。

## 安装与启动

```sh
sudo apt install ./vision-studio_1.4.0-1_amd64.deb
vision-studio
```

也可从应用菜单打开“Vision Studio 视觉模型工作台”。`apt` 会安装 control 声明的系统依赖。此包需要 Python 3.10，因此不适用于默认 Python 3.12 的 Ubuntu 24.04；其他发行版需要单独构建和验证。

## 无管理员权限构建

构建需要已有的 Release 二进制、包含 WebEngine / WebChannel / Positioning 的 Qt 6.8.3 SDK、包含 Netron 9.3.1 的 Python 3.10 CPU 环境，以及 `dpkg-dev`、`patchelf`、`binutils`。Qt SDK 的间接依赖包括 Qml / Quick / QuickWidgets。打包只修改新 staging 目录中的副本，不修改输入目录。

```sh
python3 packaging/build_deb.py \
  --binary build/release/bin/vision-studio \
  --qt-prefix /path/to/Qt/6.8.3/gcc_64 \
  --extra-qt-plugins build/qt-imageformats-prefix/plugins \
  --runtime build/release-runtime
```

默认产物在 `output/releases/1.4.0/`，staging 在 `output/deb-stage/1.4.0-1/`。已存在的 staging 和 `.deb` 会被拒绝覆盖；需要重打包时指定新的 `--stage` 和 `--output`。`--install-root` 可改变应用安装目录，默认 `/opt/VisionStudio`；编译时的生产资源根也应使用相同路径。`--compression zstd` 提供另一种 Ubuntu 支持的压缩方式，默认使用 xz。

构建脚本保留 ELF 依赖闭包需要的 Qt 运行库与图片、X11、Wayland、输入法及桌面文件对话框插件，重写为 `$ORIGIN` 相对 RUNPATH，去除 staging 内应用和 Qt 的调试符号。WebP/TIFF 等补充插件通过 `--extra-qt-plugins` 加入，必须匹配 Qt 6.8.3。

WebEngine 部署另包含 `libexec/QtWebEngineProcess`、六个 `resources/` 文件和 `translations/qtwebengine_locales/`。辅助进程作为 ELF 依赖分析入口，使用自身相对 RUNPATH；应用与辅助进程的 `qt.conf` 及启动器提供正确资源路径。默认启动器保留 Chromium sandbox，不要求修改系统权限或安装浏览器。Netron 的 Python / JS 资产由 runtime 直接携带，`scripts/netron_server.py` 只在本机回环随机端口服务当前文件。

发行环境使用 `opencv-python-headless`，避免 Python OpenCV 再带入另一套 Qt5。Python wheels 的原生库不重新 strip，以免损坏 auditwheel 调整过的 ELF 段布局。构建检查符号与版本 ABI，将 GNU 数学/线程运行库依赖重定向到 Jammy 系统库；OpenCV 的旧 LP64 OpenBLAS 使用系统 BLAS/LAPACK 替代，NumPy 的独立 ILP64 OpenBLAS仍保留。实际版本与全部包锁由所选发行环境元数据生成。

Python 包包含运行所需模块与授权资料，删除 pip、旧环境的 console 入口、激活脚本和字节码缓存；解释器通过 `/usr/bin/python3.10` 运行，不依赖原虚拟环境位置。重建 headless 环境时应按完整发行锁使用 `pip --no-deps`，避免 Ultralytics 的默认依赖重新安装 GUI OpenCV。`dpkg-shlibdeps` 根据 Jammy 的库元数据生成系统依赖；`dpkg-deb --root-owner-group` 写入系统所有权，无需 sudo。

## 验证与发布

包旁生成 SHA256 和 JSON manifest。发布前应运行 `packaging/verify_deb.py`，检查 ownership、RUNPATH、依赖与路径，再在解压树内以普通用户运行真实 ONNX、现代 `.pt`、旧版 YOLOv5 `.pt` 推理和 Netron 模型图显示。模型显示检查须得到正整数图节点数和有效 PNG，不能仅以 HTTP 打开作为成功。安装到 `/opt` 后应用目录应保持不可写，导出结果必须写入用户目录。

```sh
dpkg-deb --info output/releases/1.4.0/vision-studio_1.4.0-1_amd64.deb
python3 packaging/verify_deb.py output/releases/1.4.0/vision-studio_1.4.0-1_amd64.deb \
  --smoke --apt-simulate --source-dir output/releases/1.4.0
python3 packaging/verify_install_namespace.py \
  output/releases/1.4.0/vision-studio_1.4.0-1_amd64.deb
```

隔离安装验收在私有命名空间内执行安装、断网普通用户运行和卸载，不修改宿主的软件包数据库。无显示器检查可使用 `--disable-gpu` 软件渲染，默认仍保留 sandbox；若特定测试容器无法提供 Chromium 所需的内核隔离，应在该次验收记录中明确说明测试限制，不改正常启动器。

运行环境只读取选定的本地 `.pt`，不会根据文件名下载或替换权重。可用 `VISION_STUDIO_PYTHON` 指定另一个兼容的本地环境，用 `VISION_STUDIO_DATA_DIR` 指定测试或导出数据目录。`VISION_STUDIO_QT_HOME` 可选择自行构建的兼容 Qt 6.8.3 prefix，须包含 `lib/`、`plugins/`、`libexec/QtWebEngineProcess`、`resources/` 和 `translations/qtwebengine_locales/`；替换库、插件和 WebEngine 资产共同加载。发布授权、模型来源及第三方许可见包内 `/usr/share/doc/vision-studio/copyright` 和 `licenses/`，以及 `docs/release/`。发布时同时提供 DEB、`vision-studio-1.4.0-complete-source.tar.xz` 与校验两个主文件的 `SHA256SUMS`，源码内的 `SOURCE-SHA256SUMS` 校验各原始归档。

对应源码包含 qtbase、qtsvg、qtwayland、qtimageformats、qtdeclarative、qtwebchannel、qtpositioning、qtwebengine 八个模块；QtWebEngine 归档包含 Chromium 源码及原通知。Netron 9.3.1 原始 tag 源码和 MIT、Dagre / Graphlib 原通知也随源码交付。许可快照与实际 57 项发行 runtime 元数据一致，完整重建步骤见 `docs/release/对应源码与重建.md`。

参考：[dpkg-deb 官方手册](https://manpages.debian.org/bookworm/dpkg/dpkg-deb.1.en.html)、[dpkg-shlibdeps 官方手册](https://manpages.debian.org/bookworm/dpkg-dev/dpkg-shlibdeps.1.en.html)。
