#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_dir/runtime"
base_python="${VISION_STUDIO_BASE_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$runtime_dir/bin/python" ]]; then
    "$base_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), "The bundled legacy YOLOv5 runtime requires Python 3.10 or 3.11; set VISION_STUDIO_BASE_PYTHON to a supported interpreter"'
    "$base_python" -m venv "$runtime_dir"
fi
runtime_python="$runtime_dir/bin/python"
"$runtime_python" -m pip install --upgrade pip
"$runtime_python" -m pip install -r "$project_dir/requirements-pt.txt" -c "$project_dir/requirements-pt.lock.txt"
"$runtime_python" -c 'import torch, torchvision, ultralytics, netron; print("PT / Netron runtime ready:", torch.__version__, torchvision.__version__, ultralytics.__version__, netron.__version__); assert torch.version.cuda is None, "Expected CPU-only torch"; assert netron.__version__ == "9.3.1", "Expected pinned Netron version"'
"$runtime_python" -m pip freeze > "$runtime_dir/installed-packages.txt"
printf 'PT 推理与模型显示环境准备完成：%s\n' "$runtime_python"
