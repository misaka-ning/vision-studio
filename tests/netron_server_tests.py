#!/usr/bin/env python3
"""Real HTTP/protocol tests for the official Netron 9.3.1 session helper.

Run with the prepared runtime's Python. These tests do not execute model files
or use PyTorch/ONNX loaders; parsing remains a separate browser responsibility.
"""

from contextlib import contextmanager
import html
import json
import os
from pathlib import Path
import pickle
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

PROJECT = Path(__file__).resolve().parent.parent
HELPER = PROJECT / "scripts" / "netron_server.py"
DEFAULT_PYTHON = next((str(path) for path in (PROJECT / "runtime" / "bin" / "python",
                                            PROJECT / "build" / "release-runtime" / "bin" / "python")
                       if path.is_file()), sys.executable)
PYTHON = (os.environ.get("VISION_STUDIO_NETRON_PYTHON")
          or os.environ.get("VISION_STUDIO_PYTHON") or DEFAULT_PYTHON)
VERSION = "9.3.1"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class UnsafePickleProbe:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # Constructing this pickle does not run the expression. If the server
        # were to deserialize it, the marker would reveal that regression.
        return eval, ("open(" + repr(str(self.marker)) + ", 'w').write('executed')",)


def read_message(process, timeout=8.0):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise AssertionError("Netron helper did not emit a protocol message")
    line = process.stdout.readline()
    if not line:
        raise AssertionError("Netron helper exited without a protocol message")
    value = json.loads(line.decode("utf-8"))
    if value.get("protocol") != 1:
        raise AssertionError("Unexpected protocol version: " + repr(value))
    return value


@contextmanager
def spawn(model, *, interpreter=None, environment=None, arguments=None):
    process = subprocess.Popen(
        [interpreter or PYTHON, "-u", str(HELPER)]
        + (arguments if arguments is not None else ["--model", str(model)]),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, bufsize=0,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def fetch(url):
    with HTTP.open(url, timeout=3) as response:
        return response.status, response.headers, response.read()


class NetronServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = subprocess.run([PYTHON, "-c", "import netron; print(netron.__version__)"],
                               check=True, capture_output=True, text=True, timeout=5)
        if probe.stdout.strip() != VERSION:
            raise RuntimeError("These integration tests require official netron==" + VERSION)
        for model in ("yolov5n.onnx", "yolov8n.pt"):
            if not (PROJECT / "models" / model).is_file():
                raise RuntimeError("Missing real model fixture: " + model)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vision-netron-tests-")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def model(self, filename="模型 含空格.onnx", source="yolov5n.onnx"):
        target = self.directory / "中文 路径" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / "models" / source, target)
        return target

    def assert_ready(self, process, path):
        ready = read_message(process)
        self.assertEqual(ready["type"], "ready", ready)
        self.assertEqual(ready["version"], VERSION)
        self.assertEqual(ready["model_path"], str(path.resolve()))
        url = urllib.parse.urlparse(ready["url"])
        self.assertEqual(url.scheme, "http")
        self.assertEqual(url.hostname, "127.0.0.1")
        self.assertGreater(url.port, 0)
        self.assertNotIn(url.port, (8080, 8081))
        return ready

    def stop(self, process, command=b'{"type":"quit"}\n'):
        process.stdin.write(command)
        process.stdin.flush()
        self.assertEqual(process.wait(timeout=3), 0)
        self.assertEqual(process.stdout.read(), b"")

    def assert_closed(self, url):
        parsed = urllib.parse.urlparse(url)
        with self.assertRaises(OSError):
            with socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
                pass

    def assert_denied(self, url):
        with self.assertRaises(urllib.error.HTTPError) as error:
            fetch(url)
        self.assertEqual(error.exception.code, 404)

    def test_real_onnx_static_assets_model_bytes_and_json_quit(self):
        path = self.model()
        with spawn(path) as process:
            ready = self.assert_ready(process, path)
            status, headers, data = fetch(ready["url"])
            self.assertEqual(status, 200)
            self.assertTrue(headers["Content-Type"].startswith("text/html"))
            page = data.decode("utf-8")
            self.assertIn('name="version" content="' + VERSION + '"', page)
            match = re.search(r'<meta name="file" content="([^"]+)"', page)
            self.assertIsNotNone(match)
            model_url = ready["url"].rstrip("/") + urllib.parse.quote(html.unescape(match.group(1)), safe="/")
            status, headers, data = fetch(model_url)
            self.assertEqual(status, 200)
            self.assertEqual(data, path.read_bytes())
            self.assertEqual(headers["Content-Type"], "application/octet-stream")
            for resource in ("view.js", "browser.js", "onnx.js", "grapher.css"):
                status, _, data = fetch(ready["url"] + resource)
                self.assertEqual(status, 200)
                self.assertGreater(len(data), 100)
            self.stop(process)
            self.assert_closed(ready["url"])

    def test_real_pt_raw_bytes_and_plain_quit(self):
        path = self.model("检测 模型.pt", "yolov8n.pt")
        with spawn(path) as process:
            ready = self.assert_ready(process, path)
            _, _, data = fetch(ready["url"] + "data/" + urllib.parse.quote(path.name))
            self.assertEqual(data, path.read_bytes())
            self.stop(process, b"quit\n")

    def test_unknown_suffix_and_corrupt_bytes_are_served_without_parsing(self):
        for name, payload in (("model.unknown", b"not a model"), ("broken.onnx", b"bad protobuf"),
                              ("empty.pt", b"")):
            with self.subTest(name=name):
                path = self.directory / name
                path.write_bytes(payload)
                with spawn(path) as process:
                    ready = self.assert_ready(process, path)
                    if payload:
                        _, _, data = fetch(ready["url"] + "data/" + name)
                        self.assertEqual(data, payload)
                    # Empty/corrupt formats are deliberately not accepted as a
                    # parsed graph merely because an HTTP session is ready.
                    self.stop(process)

    def test_external_data_stays_within_model_directory(self):
        path = self.model()
        external = path.parent / "权重 data.bin"
        external.write_bytes(b"external ONNX tensor bytes")
        secret = self.directory / "outside-secret.bin"
        secret.write_bytes(b"must not be served")
        (path.parent / "outside-link.bin").symlink_to(secret)
        with spawn(path) as process:
            ready = self.assert_ready(process, path)
            _, _, data = fetch(ready["url"] + "data/" + urllib.parse.quote(external.name))
            self.assertEqual(data, external.read_bytes())
            self.assert_denied(ready["url"] + "data/" + urllib.parse.quote("../outside-secret.bin", safe=""))
            self.assert_denied(ready["url"] + "data/outside-link.bin")
            self.assert_denied(ready["url"] + urllib.parse.quote("../outside-secret.bin", safe=""))
            self.stop(process)

    def test_pickle_payload_is_never_deserialized_or_framework_imported(self):
        marker = self.directory / "execution-marker"
        path = self.directory / "checkpoint.pt"
        payload = pickle.dumps(UnsafePickleProbe(marker))
        path.write_bytes(payload)
        traps = self.directory / "framework-import-traps"
        traps.mkdir()
        for module in ("torch", "onnx", "ultralytics"):
            (traps / (module + ".py")).write_text(
                "raise RuntimeError('model framework must not be imported by viewer service')\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(traps)
        with spawn(path, environment=environment) as process:
            ready = self.assert_ready(process, path)
            self.assertEqual(fetch(ready["url"] + "data/checkpoint.pt")[2], payload)
            self.stop(process)
        self.assertFalse(marker.exists())

    def test_eof_and_sigterm_close_session(self):
        path = self.model()
        for method in ("eof", "signal"):
            with self.subTest(method=method), spawn(path) as process:
                ready = self.assert_ready(process, path)
                if method == "eof":
                    process.stdin.close()
                else:
                    process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=3), 0)
                self.assertEqual(process.stdout.read(), b"")
                self.assert_closed(ready["url"])

    def test_concurrent_sessions_have_different_ephemeral_ports(self):
        first = self.model()
        second = self.model("second.pt", "yolov8n.pt")
        with spawn(first) as a, spawn(second) as b:
            ready_a = self.assert_ready(a, first)
            ready_b = self.assert_ready(b, second)
            self.assertNotEqual(ready_a["url"], ready_b["url"])
            self.stop(a)
            self.assertEqual(fetch(ready_b["url"])[0], 200)
            self.stop(b)

    def test_invalid_path_directory_and_arguments_emit_errors(self):
        for path in (self.directory / "不存在.pt", self.directory):
            with self.subTest(path=path), spawn(path) as process:
                message = read_message(process)
                self.assertEqual(message["type"], "error")
                self.assertEqual(message["code"], "invalid_model")
                self.assertEqual(process.wait(timeout=3), 2)
                self.assertEqual(process.stdout.read(), b"")
        with spawn(None, arguments=[]) as process:
            self.assertEqual(read_message(process)["code"], "invalid_arguments")
            self.assertEqual(process.wait(timeout=3), 2)

    def test_missing_module_and_wrong_version_are_clear_errors(self):
        path = self.model()
        # -S prevents loading the runtime's site-packages, genuinely making
        # Netron unavailable without changing the installed environment.
        isolated = subprocess.Popen(
            [PYTHON, "-S", "-u", str(HELPER), "--model", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        try:
            message = read_message(isolated)
            self.assertEqual(message["code"], "missing_module")
            self.assertIn("Netron", message["message"])
            self.assertEqual(isolated.wait(timeout=3), 2)
        finally:
            if isolated.poll() is None:
                isolated.kill()
                isolated.wait(timeout=3)
            for stream in (isolated.stdin, isolated.stdout, isolated.stderr):
                stream.close()
        module = self.directory / "wrong-version" / "netron"
        module.mkdir(parents=True)
        (module / "__init__.py").write_text(
            "print('a library log must not corrupt stdout')\n__version__ = '0.0.0'\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(module.parent)
        with spawn(path, environment=environment) as process:
            message = read_message(process)
            self.assertEqual(message["code"], "unsupported_version")
            self.assertIn("0.0.0", message["message"])
            self.assertEqual(process.wait(timeout=3), 2)
            self.assertEqual(process.stdout.read(), b"")
            self.assertIn(b"library log", process.stderr.read())

    def test_invalid_command_releases_port(self):
        path = self.model()
        with spawn(path) as process:
            ready = self.assert_ready(process, path)
            process.stdin.write(b'{"type":"unknown"}\n')
            process.stdin.flush()
            self.assertEqual(read_message(process)["code"], "invalid_command")
            self.assertEqual(process.wait(timeout=3), 2)
            self.assert_closed(ready["url"])

    def test_slow_http_client_does_not_block_quit(self):
        path = self.model("large.pt", "yolov8n.pt")
        with spawn(path) as process:
            ready = self.assert_ready(process, path)
            address = urllib.parse.urlparse(ready["url"])
            with socket.create_connection((address.hostname, address.port), timeout=1) as client:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                client.sendall(b"GET /data/large.pt HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                time.sleep(0.1)
                started = time.monotonic()
                self.stop(process)
                self.assertLess(time.monotonic() - started, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
