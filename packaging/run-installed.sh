#!/bin/sh
# Resolve the application tree from this file, including an extracted test tree.
set -eu
app_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export VISION_STUDIO_HOME="$app_root"
export VISION_STUDIO_SYSTEM_INSTALL=1
export VISION_STUDIO_PYTHON="${VISION_STUDIO_PYTHON:-$app_root/runtime/bin/python}"
export VISION_STUDIO_PT_WORKER="${VISION_STUDIO_PT_WORKER:-$app_root/scripts/pt_worker.py}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [ -n "${VISION_STUDIO_QT_HOME:-}" ]; then
    qt_libraries="$VISION_STUDIO_QT_HOME/lib"
    qt_plugins="$VISION_STUDIO_QT_HOME/plugins"
    if [ ! -f "$qt_libraries/libQt6Core.so.6" ] || [ ! -d "$qt_plugins/platforms" ]; then
        printf '%s\n' 'VISION_STUDIO_QT_HOME 必须包含兼容 Qt 6.8.3 的 lib 和 plugins 目录。' >&2
        exit 1
    fi
else
    qt_libraries="$app_root/lib/qt"
    qt_plugins="$app_root/plugins"
fi
export QT_PLUGIN_PATH="$qt_plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$qt_plugins/platforms"
export LD_LIBRARY_PATH="$qt_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The application chooses a writable per-user QStandardPaths data directory.
exec "$app_root/build/bin/vision-studio" "$@"
