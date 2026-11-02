#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the audited CTest suites and reject missing, skipped or empty results."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    logs = build / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a successful summary from an older run.
    (logs / "summary.txt").unlink(missing_ok=True)
    expected = (build / "pr-core-tests.txt").read_text().splitlines()
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("The audited test manifest is empty or has duplicates")

    command = ["ctest", "--test-dir", str(build), "-L", "^pr-core$"]
    discovery = subprocess.run(command + ["--show-only=json-v1"], check=True,
                               text=True, capture_output=True)
    (logs / "discovery.json").write_text(discovery.stdout)
    discovered = json.loads(discovery.stdout)["tests"]
    if sorted(t["name"] for t in discovered) != sorted(expected):
        raise RuntimeError("CTest discovery differs from the audited test manifest")
    for test in discovered:
        properties = {p["name"]: p["value"] for p in test.get("properties", [])}
        if properties.get("DISABLED") or not test.get("command"):
            raise RuntimeError(f"Test is disabled or not built: {test['name']}")

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    for key in ("VISION_STUDIO_GPU_TESTS", "VISION_STUDIO_NETRON_WORKER", "VISION_STUDIO_CONVERSION_WORKER",
                "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QML2_IMPORT_PATH"):
        env.pop(key, None)
    # Keep personal data and optional installed runtimes out of the test run.
    root = Path(__file__).resolve().parents[2]
    env.update(VISION_STUDIO_HOME=str(root), VISION_STUDIO_PYTHON=sys.executable,
               VISION_STUDIO_EVALUATION_PYTHON=sys.executable,
               VISION_STUDIO_GPU_RUNTIME_DIR=str(build / "absent-gpu-runtime"),
               QT_QPA_PLATFORM="offscreen", QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu",
               LIBGL_ALWAYS_SOFTWARE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for key, directory in (("VISION_STUDIO_DATA_DIR", "data"), ("XDG_CONFIG_HOME", "config"),
                           ("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "cache"),
                           ("XDG_RUNTIME_DIR", "runtime")):
        path = build / "test-environment" / directory
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    # CTest serial execution keeps GUI responsiveness assertions meaningful.
    result = subprocess.run(command + ["--no-tests=error", "--parallel", "1", "--timeout", "180",
                            "--verbose", "--output-on-failure", "--test-output-size-passed", "1048576",
                            "--test-output-size-failed", "1048576", "--output-log", str(logs / "ctest.log"),
                            "--output-junit", str(logs / "ctest.xml")], env=env)
    if result.returncode:
        return result.returncode

    cases = ET.parse(logs / "ctest.xml").getroot().findall(".//testcase")
    if sorted(c.get("name") for c in cases) != sorted(expected):
        raise RuntimeError("JUnit does not contain every audited suite exactly once")
    count = 0
    for case in cases:
        if case.get("status") != "run" or any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise RuntimeError(f"Suite did not pass: {case.get('name')}")
        output = case.findtext("system-out", "")
        totals = re.findall(r"Totals: (\d+) passed, (\d+) failed, (\d+) skipped, (\d+) blacklisted,", output)
        if len(totals) != 1 or int(totals[0][0]) < 3 or totals[0][1:] != ("0", "0", "0"):
            raise RuntimeError(f"Missing/empty/failed/skipped QtTest results: {case.get('name')}")
        count += int(totals[0][0])
    summary = f"{len(cases)} CTest suites; {count} QtTest passes (including init/cleanup); zero failures/skips.\n"
    (logs / "summary.txt").write_text(summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ET.ParseError, subprocess.CalledProcessError) as error:
        sys.exit(f"PR core acceptance failed: {error}")
