#!/usr/bin/env python3
"""GPU configuration safety tests; no downloads, GPU, or driver changes."""
import importlib.util
import hashlib
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gpu_probe
import gpu_setup


class GpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vision-gpu-tests-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def managed_runtime(self, name="gpu-runtime"):
        target = self.root / name
        target.mkdir()
        (target / "pyvenv.cfg").write_text("home=/usr/bin\n")
        (target / "ready.json").write_text(json.dumps({"schema": 1, "installer": gpu_probe.INSTALLER, **gpu_probe.PINS}))
        return target

    def test_missing_runtime_does_not_run_an_interpreter(self):
        with mock.patch.object(gpu_probe, "hardware", return_value=([{"index": 0, "name": "GPU", "total_memory_mb": 8000}], "595")), mock.patch.object(gpu_probe, "managed_command", side_effect=AssertionError("must not execute")):
            result = gpu_probe.probe(self.root / "missing")
        self.assertFalse(result["prepared"])
        self.assertFalse(result["cuda_available"])
        self.assertEqual(result["driver_version"], "595")
        self.assertEqual(result["devices"][0]["index"], 0)

    def test_unknown_directory_is_preserved(self):
        target = self.root / "unknown"
        target.mkdir()
        (target / "personal.txt").write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "未知内容"):
            gpu_setup.validate_destination(target)
        self.assertEqual((target / "personal.txt").read_text(), "keep")

    def test_symlink_directory_is_rejected(self):
        target = self.managed_runtime()
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        self.assertFalse(gpu_probe.owned_runtime(alias))
        with self.assertRaisesRegex(RuntimeError, "符号链接"):
            gpu_setup.validate_destination(alias)

    def test_wrong_pins_are_not_a_managed_runtime(self):
        target = self.managed_runtime()
        metadata = json.loads((target / "ready.json").read_text())
        metadata["torch_version"] = "2.9.1+cpu"
        (target / "ready.json").write_text(json.dumps(metadata))
        self.assertFalse(gpu_setup.managed(target))

    def test_ready_runtime_is_reused_without_download(self):
        target = self.managed_runtime()
        with mock.patch.object(gpu_setup, "probe", return_value={"ok": True}), mock.patch.object(gpu_setup, "run", side_effect=AssertionError("must not install")), mock.patch.object(gpu_setup, "event") as event:
            gpu_setup.setup(target, Path(sys.executable))
        event.assert_called_once_with("ready", runtime_dir=str(target), reused=True)

    def test_atomic_publication_never_overwrites_unknown_destination(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir(); second.mkdir()
        (first / "new").write_text("new"); (second / "old").write_text("old")
        with self.assertRaises(OSError):
            gpu_setup.atomic_rename(first, second, 1)
        self.assertTrue((first / "new").exists())
        self.assertTrue((second / "old").exists())
        gpu_setup.exchange(first, second)
        self.assertTrue((first / "old").exists())
        self.assertTrue((second / "new").exists())

    def test_relocation_updates_only_generated_entry_scripts(self):
        stage = self.root / "stage"; (stage / "bin").mkdir(parents=True)
        entry = stage / "bin/pip"
        entry.write_text("#!" + str(stage / "bin/python") + "\n")
        binary = stage / "bin/proprietary.so"
        binary.write_bytes(b"\x7fELF\0" + str(stage).encode())
        original = binary.read_bytes()
        gpu_setup.relocate_entries(stage, self.root / "final")
        self.assertIn(str(self.root / "final/bin/python"), entry.read_text())
        self.assertEqual(binary.read_bytes(), original)

    def test_complete_lock_uses_hashes_and_expected_cuda_pins(self):
        lock = ROOT / "requirements-gpu.lock.txt"
        pins = gpu_setup.read_pins(lock)
        self.assertEqual(pins["torch"], "2.9.1+cu128")
        self.assertEqual(pins["onnxruntime-gpu"], "1.23.2")
        self.assertEqual(pins["nvidia-cudnn-cu12"], "9.10.2.21")
        self.assertNotIn("opencv-python", pins)
        self.assertNotIn("netron", pins)
        for block in lock.read_text().split("==")[1:]:
            self.assertIn("--hash=sha256:", block)

    def test_default_cuda_links_use_canonical_official_host(self):
        text = gpu_setup.canonical_links((ROOT / "requirements-gpu.lock.txt").read_text())
        self.assertEqual(text.count('href="https://download.pytorch.org/whl/cu128/'), 4)
        self.assertNotIn("download-r2", text)
        self.assertNotIn("extra-index-url", (ROOT / "requirements-gpu.txt").read_text())
        with self.assertRaisesRegex(RuntimeError, "哈希锁"):
            gpu_setup.canonical_links("missing hashes")

    def test_complete_offline_wheelhouse_checks_python_and_publisher_hash(self):
        wheel = self.root / "example-1.0-cp310-cp310-manylinux_2_28_x86_64.whl"
        wheel.write_bytes(b"test wheel")
        lock = "--hash=sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
        self.assertTrue(gpu_setup.complete_wheelhouse(self.root, {"example": "1.0"}, [3, 10], lock))
        self.assertFalse(gpu_setup.complete_wheelhouse(self.root, {"example": "1.0"}, [3, 11], lock))
        self.assertFalse(gpu_setup.complete_wheelhouse(self.root, {"example": "1.0"}, [3, 10], "wrong hash"))
        self.assertFalse(gpu_setup.complete_wheelhouse(self.root, {"example": "1.0", "missing": "1.0"}, [3, 10], lock))

    def test_cancel_stops_private_child_process_group(self):
        marker = self.root / "child.pid"
        launcher = self.root / "launcher.py"
        launcher.write_text("import sys,signal\nsys.path.insert(0," + repr(str(ROOT / "scripts")) + ")\nimport gpu_setup\nsignal.signal(signal.SIGTERM,gpu_setup.cancelled)\ntry:\n gpu_setup.run([sys.executable,'-c'," + repr("import os,time,pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)") + "])\nexcept gpu_setup.Cancelled:\n sys.exit(130)\n")
        process = subprocess.Popen([sys.executable, str(launcher)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):
                if marker.exists(): break
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            child = int(marker.read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 130)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()

    def test_cancel_during_probe_stops_cuda_child(self):
        target = self.managed_runtime()
        (target / "bin").mkdir()
        (target / "bin/python").symlink_to(sys.executable)
        marker = self.root / "probe-child.pid"
        child_code = "import os,time,pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
        launcher = self.root / "probe-launcher.py"
        launcher.write_text("import sys,signal\nsys.path.insert(0," + repr(str(ROOT / "scripts")) + ")\nimport gpu_probe,gpu_setup\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,gpu_setup.cancelled)\ngpu_probe.hardware=lambda:([], '')\ngpu_probe.CHILD_PROBE=" + repr(child_code) + "\ntry:\n gpu_probe.probe(Path(" + repr(str(target)) + "))\nexcept gpu_setup.Cancelled:\n sys.exit(130)\n")
        process = subprocess.Popen([sys.executable, str(launcher)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):
                if marker.exists(): break
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            child = int(marker.read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 130)
            with self.assertRaises(ProcessLookupError): os.kill(child, 0)
        finally:
            if process.poll() is None: process.kill(); process.wait()

    def fake_install(self, command, **kwargs):
        if "venv" in command:
            stage = Path(command[-1])
            (stage / "bin").mkdir()
            (stage / "pyvenv.cfg").write_text("home=/usr/bin\n")
            (stage / "bin/python").write_text("#!/usr/bin/python3.10\n")

    def test_failed_final_probe_retracts_first_publication(self):
        target = self.root / "first-runtime"
        with mock.patch.object(gpu_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="[3, 10]", stderr="")), mock.patch.object(gpu_setup, "run", side_effect=self.fake_install), mock.patch.object(gpu_setup, "event"), mock.patch.object(gpu_setup, "probe", side_effect=[{"ok": True}, {"ok": False, "reason": "final failed"}]):
            with self.assertRaisesRegex(RuntimeError, "final failed"):
                gpu_setup.setup(target, Path(sys.executable))
        self.assertFalse(target.exists())

    def test_failed_replacement_probe_restores_old_runtime(self):
        target = self.managed_runtime()
        (target / "old-data").write_text("preserved")
        with mock.patch.object(gpu_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="[3, 10]", stderr="")), mock.patch.object(gpu_setup, "run", side_effect=self.fake_install), mock.patch.object(gpu_setup, "event"), mock.patch.object(gpu_setup, "probe", side_effect=[{"ok": False}, {"ok": True}, {"ok": False, "reason": "final failed"}]):
            with self.assertRaisesRegex(RuntimeError, "final failed"):
                gpu_setup.setup(target, Path(sys.executable))
        self.assertEqual((target / "old-data").read_text(), "preserved")
        self.assertTrue(gpu_setup.managed(target))

    def test_cancel_at_candidate_probe_keeps_old_environment(self):
        target = self.managed_runtime()
        (target / "old-data").write_text("preserved")
        with mock.patch.object(gpu_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="[3, 10]", stderr="")), mock.patch.object(gpu_setup, "run", side_effect=self.fake_install), mock.patch.object(gpu_setup, "event"), mock.patch.object(gpu_setup, "probe", side_effect=[{"ok": False}, gpu_setup.Cancelled("cancelled")]):
            with self.assertRaises(gpu_setup.Cancelled):
                gpu_setup.setup(target, Path(sys.executable))
        self.assertEqual((target / "old-data").read_text(), "preserved")
        self.assertTrue(gpu_setup.managed(target))

    def test_cancel_at_exchange_boundary_restores_old_environment(self):
        stage = self.managed_runtime("stage")
        target = self.managed_runtime()
        (target / "old-data").write_text("preserved")
        original = gpu_setup.exchange
        count = 0
        def exchange_then_signal(first, second):
            nonlocal count
            original(first, second)
            count += 1
            if count == 1: os.kill(os.getpid(), signal.SIGTERM)
        handler = signal.signal(signal.SIGTERM, gpu_setup.cancelled)
        try:
            with mock.patch.object(gpu_setup, "exchange", side_effect=exchange_then_signal):
                with self.assertRaises(gpu_setup.Cancelled):
                    gpu_setup.publish_checked(stage, target, {stage})
        finally:
            signal.signal(signal.SIGTERM, handler)
        self.assertEqual((target / "old-data").read_text(), "preserved")

    def test_cancel_at_first_rename_boundary_retracts_runtime(self):
        stage = self.managed_runtime("stage")
        target = self.root / "new-runtime"
        original = gpu_setup.atomic_rename
        count = 0
        def rename_then_signal(first, second, flags):
            nonlocal count
            original(first, second, flags)
            count += 1
            if count == 1: os.kill(os.getpid(), signal.SIGTERM)
        handler = signal.signal(signal.SIGTERM, gpu_setup.cancelled)
        try:
            with mock.patch.object(gpu_setup, "atomic_rename", side_effect=rename_then_signal):
                with self.assertRaises(gpu_setup.Cancelled):
                    gpu_setup.publish_checked(stage, target, {stage})
        finally:
            signal.signal(signal.SIGTERM, handler)
        self.assertFalse(target.exists())
        self.assertTrue(stage.exists())

    def test_failed_rollback_keeps_unique_old_backup_out_of_cleanup(self):
        stage = self.managed_runtime("stage")
        target = self.managed_runtime()
        (target / "old-data").write_text("preserved")
        stages = {stage}
        original = gpu_setup.exchange
        count = 0
        def fail_rollback(first, second):
            nonlocal count
            count += 1
            if count == 2: raise OSError("rollback fault")
            original(first, second)
        with mock.patch.object(gpu_setup, "exchange", side_effect=fail_rollback), mock.patch.object(gpu_setup, "probe", return_value={"ok": False, "reason": "fault"}), mock.patch.object(gpu_setup, "event"):
            with self.assertRaisesRegex(RuntimeError, "环境已保留"):
                gpu_setup.publish_checked(stage, target, stages)
        self.assertNotIn(stage, stages)
        self.assertEqual((stage / "old-data").read_text(), "preserved")

    def verify_grandchild_cleanup(self, mode):
        # Adopt the test-only grandchild so it can be reaped deterministically.
        libc = ctypes.CDLL(None)
        original_subreaper = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(original_subreaper), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        marker = self.root / (mode + "-grandchild.pid")
        grandchild_code = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
        leader_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(grandchild_code) + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(30)"
        launcher = self.root / (mode + "-grandchild-launcher.py")
        launcher.write_text("import sys,signal\nsys.path.insert(0," + repr(str(ROOT / "scripts")) + ")\nimport gpu_probe,gpu_setup\nsignal.signal(signal.SIGTERM,gpu_setup.cancelled)\ntry:\n " + ("gpu_setup.run" if mode == "setup" else "gpu_probe.managed_command") + "([sys.executable,'-c'," + repr(leader_code) + "])\nexcept gpu_setup.Cancelled:\n sys.exit(130)\n")
        process = subprocess.Popen([sys.executable, str(launcher)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child, reaped = None, False
        try:
            for _ in range(60):
                if marker.exists(): break
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            child = int(marker.read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 130)
            for _ in range(40):
                found, status = os.waitpid(child, os.WNOHANG)
                if found:
                    reaped = True
                    self.assertEqual(os.waitstatus_to_exitcode(status), -signal.SIGKILL)
                    break
                time.sleep(0.05)
            self.assertTrue(reaped, "SIGTERM-ignoring grandchild survived cancellation")
        finally:
            if process.poll() is None: process.kill(); process.wait()
            if child is not None and not reaped:
                try: os.kill(child, signal.SIGKILL); os.waitpid(child, 0)
                except (ProcessLookupError, ChildProcessError): pass
            libc.prctl(36, original_subreaper.value, 0, 0, 0)

    def test_setup_cancellation_kills_sigterm_ignoring_grandchild(self):
        self.verify_grandchild_cleanup("setup")

    def test_probe_cancellation_kills_sigterm_ignoring_grandchild(self):
        self.verify_grandchild_cleanup("probe")

    def test_inventory_preserves_ort_third_party_notice(self):
        site = self.root / "site"
        notice = site / "onnxruntime/ThirdPartyNotices.txt"
        notice.parent.mkdir(parents=True)
        original = b"Original upstream third-party copyright and license notice.\n"
        notice.write_bytes(original)

        class Distribution:
            metadata = {"Name": "onnxruntime-gpu", "License": "MIT"}
            version = "1.23.2"
            files = [Path("onnxruntime/ThirdPartyNotices.txt")]

            def locate_file(self, item):
                return site / item

        with mock.patch("importlib.metadata.distributions", return_value=[Distribution()]), \
             mock.patch.object(sys, "argv", ["inventory", str(self.root), '{"onnxruntime-gpu":"1.23.2"}']):
            exec(gpu_setup.INVENTORY, {})
        copied = self.root / "licenses/onnxruntime-gpu/onnxruntime/ThirdPartyNotices.txt"
        self.assertEqual(copied.read_bytes(), original)
        record = json.loads((self.root / "gpu-license-inventory.json").read_text())[0]
        self.assertIn(str(copied.relative_to(self.root)), record["license_files"])


if __name__ == "__main__":
    unittest.main()
