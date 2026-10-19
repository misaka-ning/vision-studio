#!/usr/bin/env bash
# Usage: run_gui_qa.sh /absolute/path/to/vision-studio /absolute/fresh/qa-root /absolute/bus.jpg
# The first argument may be the final package's read-only launcher or the built executable.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf '%s\n' 'Usage: run_gui_qa.sh APP_EXECUTABLE FRESH_QA_ROOT REAL_SAMPLE_IMAGE' >&2
  exit 2
fi
app_path=$(realpath "$1")
qa_root=$(realpath -m "$2")
sample_path=$(realpath "$3")
helper_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ ! -x "$app_path" ] || [ ! -f "$sample_path" ] || [ -e "$qa_root" ]; then
  printf '%s\n' 'An executable, a sample image and a new (nonexistent) isolated QA directory are required.' >&2
  exit 2
fi
app_directory=$(dirname -- "$app_path")
if [ -d "$app_directory/lib/qt" ]; then
  qt_libraries="$app_directory/lib/qt"
else
  qt_libraries="${VISION_STUDIO_GUI_QA_QT_LIBS:-/home/misaka/Qt/6.8.3/gcc_64/lib}"
fi
if [ -x "$app_directory/runtime/bin/python" ]; then
  fixture_python="$app_directory/runtime/bin/python"
else
  fixture_python="${VISION_STUDIO_PYTHON:-/opt/VisionStudio/runtime/bin/python}"
fi
mkdir -p "$qa_root/data" "$qa_root/inputs"
PYTHONDONTWRITEBYTECODE=1 "$fixture_python" - "$qa_root" "$sample_path" <<'PY'
import sys
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1])
image=Image.open(sys.argv[2]).convert('RGB')
image.thumbnail((320,430))
for index in range(61):
    image.save(root/'inputs'/f'{index+1:03d}.png')
(root/'data'/'preferences.ini').write_text('[General]\ncomputeDevice=cpu\nautoExport=true\nconfidence=0.25\n',encoding='utf-8')
PY
# Transient offscreen acceptance only. The installed application tree is never written.
set +e
env -u VISION_STUDIO_HOME -u VISION_STUDIO_PT_WORKER \
  LD_PRELOAD="$helper_root/libgui_qa.so" \
  LD_LIBRARY_PATH="$qt_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
  VISION_STUDIO_PYTHON="$fixture_python" \
  HOME="$qa_root/home" XDG_CONFIG_HOME="$qa_root/config" XDG_DATA_HOME="$qa_root/xdg-data" XDG_CACHE_HOME="$qa_root/cache" XDG_STATE_HOME="$qa_root/state" \
  YOLO_CONFIG_DIR="$qa_root/yolo" YOLOV5_CONFIG_DIR="$qa_root/yolov5" MPLCONFIGDIR="$qa_root/matplotlib" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  VISION_STUDIO_DATA_DIR="$qa_root/data" \
  VISION_STUDIO_GUI_QA_DIR="$qa_root/report" \
  VISION_STUDIO_GUI_QA_INPUTS="$qa_root/inputs" \
  "$app_path" > "$qa_root/raw.log" 2>&1
status=$?
set -e
printf 'Application exit: %s\nReport: %s\n' "$status" "$qa_root/report/installed-gui-qa.json"
exit "$status"
