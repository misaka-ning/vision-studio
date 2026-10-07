#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Explicit system interpreter: PATH may refer to an incompatible Conda Python.
exec /usr/bin/python3.10 "$script_dir/gpu_setup.py" "$@"
