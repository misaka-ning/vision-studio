#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qt_prefix="${QT_PREFIX:-$HOME/Qt/6.8.3/gcc_64}"
export VISION_STUDIO_HOME="$project_dir"
export LD_LIBRARY_PATH="$qt_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$qt_prefix/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$qt_prefix/plugins/platforms"
if [[ ! -x "$project_dir/build/bin/vision-studio" ]]; then
    "$project_dir/scripts/build.sh"
fi
# A versioned home-directory build can carry its own CPU environment while
# older builds and the user's shared runtime remain available.
versioned_binary="$(readlink -f -- "$project_dir/build/bin/vision-studio")"
versioned_python="$(dirname -- "$versioned_binary")/../runtime/bin/python"
if [[ -z "${VISION_STUDIO_PYTHON:-}" && -x "$versioned_python" ]]; then
    export VISION_STUDIO_PYTHON="$versioned_python"
fi
exec "$project_dir/build/bin/vision-studio" "$@"
