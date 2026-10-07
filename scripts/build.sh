#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
qt_prefix="${QT_PREFIX:-$HOME/Qt/6.8.3/gcc_64}"
cmake -S "$project_dir" -B "$project_dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$qt_prefix" -DBUILD_TESTING=ON
cmake --build "$project_dir/build" --parallel 4
ctest --test-dir "$project_dir/build" --output-on-failure
