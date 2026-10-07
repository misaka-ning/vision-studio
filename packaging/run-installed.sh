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
    qt_process="$VISION_STUDIO_QT_HOME/libexec/QtWebEngineProcess"
    qt_resources="$VISION_STUDIO_QT_HOME/resources"
    qt_locales="$VISION_STUDIO_QT_HOME/translations/qtwebengine_locales"
    if [ ! -f "$qt_libraries/libQt6Core.so.6" ] || [ ! -d "$qt_plugins/platforms" ]; then
        printf '%s\n' 'VISION_STUDIO_QT_HOME 必须包含兼容 Qt 6.8.3 的 lib 和 plugins 目录。' >&2
        exit 1
    fi
else
    qt_libraries="$app_root/lib/qt"
    qt_plugins="$app_root/plugins"
    qt_process="$app_root/libexec/QtWebEngineProcess"
    qt_resources="$app_root/resources"
    qt_locales="$app_root/translations/qtwebengine_locales"
fi
if [ ! -x "$qt_process" ] || [ ! -f "$qt_resources/qtwebengine_resources.pak" ] || [ ! -f "$qt_locales/en-US.pak" ]; then
    printf '%s\n' 'Qt WebEngine 6.8.3 的辅助进程、资源或语言文件缺失。' >&2
    exit 1
fi
export QT_PLUGIN_PATH="$qt_plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$qt_plugins/platforms"
export QTWEBENGINEPROCESS_PATH="$qt_process"
export QTWEBENGINE_RESOURCES_PATH="$qt_resources"
export QTWEBENGINE_LOCALES_PATH="$qt_locales"
export LD_LIBRARY_PATH="$qt_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The application chooses a writable per-user QStandardPaths data directory.
exec "$app_root/build/bin/vision-studio" "$@"
