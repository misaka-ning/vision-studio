#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
delivery_dir="${1:-$project_dir/output/VisionStudio}"
if [[ -e "$delivery_dir" ]]; then
    printf '目标目录已存在，为保留原有文件请使用一个新目录：%s\n' "$delivery_dir" >&2
    exit 1
fi
if [[ ! -x "$project_dir/build/bin/vision-studio" ]]; then
    printf '请先运行 scripts/build.sh 构建程序。\n' >&2
    exit 1
fi
mkdir -p "$delivery_dir"
delivery_dir="$(cd -- "$delivery_dir" && pwd)"
for entry in CMakeLists.txt resources.qrc README.md LICENSE .gitignore .clang-format run.sh src assets docs models tests scripts vendor packaging requirements-pt.txt requirements-pt.lock.txt requirements-gpu.txt requirements-gpu.lock.txt runtime-info.json; do
    cp -a "$project_dir/$entry" "$delivery_dir/"
done
mkdir -p "$delivery_dir/build/bin" "$delivery_dir/output"
cp -a "$project_dir/build/bin/vision-studio" "$delivery_dir/build/bin/"
ln -s build/bin/vision-studio "$delivery_dir/VisionStudio"
touch "$delivery_dir/output/.gitkeep"
cat > "$delivery_dir/VisionStudio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Vision Studio
Name[zh_CN]=Vision Studio 视觉模型工作台
Comment=Local YOLO model inference workspace
Exec="$delivery_dir/run.sh"
Icon=$delivery_dir/assets/app-icon.svg
Terminal=false
Categories=Graphics;Science;
StartupNotify=true
DESKTOP
chmod +x "$delivery_dir/run.sh" "$delivery_dir/scripts/build.sh" "$delivery_dir/scripts/package.sh" "$delivery_dir/VisionStudio.desktop"
chmod +x "$delivery_dir/scripts/setup_pt.sh"
chmod +x "$delivery_dir/scripts/setup_gpu.sh"
printf '已生成可启动的完整产品：%s\n' "$delivery_dir"
