#!/usr/bin/env python3
"""Serve one local model with the pinned official Netron package.

stdout is a UTF-8 JSON-lines control protocol, never a logging stream. ``ready``
only means the HTTP service is available: the browser performs model parsing.
The official server also serves relative external tensor files within the model
directory, while rejecting paths and symlinks that resolve outside that directory.
This process never deserializes a model with Python or imports a model framework.
"""

import argparse
import errno
import importlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

NETRON_VERSION = "9.3.1"
PROTOCOL_VERSION = 1
MAX_COMMAND_BYTES = 8192


class ProtocolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Shutdown(BaseException):
    pass


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ProtocolError("invalid_arguments", "启动参数无效：" + message)

    def _print_message(self, message, file=None):
        if message:
            sys.stderr.write(message)


def _shutdown_signal(signum, frame):
    raise Shutdown()


def _model_path(value):
    try:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ProtocolError("invalid_model", "请选择现有的模型文件，不能选择目录或设备。")
        # Check readability without parsing or copying the model into memory.
        with path.open("rb") as stream:
            stream.read(1)
        return path
    except ProtocolError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ProtocolError("invalid_model", "无法读取所选模型文件：" + str(error)) from error


def _load_netron():
    try:
        netron = importlib.import_module("netron")
    except ImportError as error:
        raise ProtocolError(
            "missing_module",
            "Python 运行环境缺少可用的 Netron 9.3.1。请使用应用附带的运行环境。"
            "（" + str(error) + "）",
        ) from error
    version = str(getattr(netron, "__version__", "未知"))
    if version != NETRON_VERSION:
        raise ProtocolError(
            "unsupported_version",
            "Netron 版本不兼容：需要 " + NETRON_VERSION + "，当前为 " + version + "。",
        )
    for name in ("start", "stop", "status"):
        if not callable(getattr(netron, name, None)):
            raise ProtocolError("unsupported_version", "Netron 安装不完整，缺少本地服务接口。")

    # Netron 9.3.1 uses ThreadingMixIn + HTTPServer. Its default non-daemon
    # request threads can otherwise outlive stop() when a slow client has not
    # read a large response. The standard-library flag makes process shutdown
    # bounded; resource handling and serving still use the official API.
    server = importlib.import_module("netron.server")
    server_class = getattr(server, "_ThreadedHTTPServer", None)
    if server_class is None:
        raise ProtocolError("unsupported_version", "Netron 本地 HTTP 服务接口不兼容。")
    server_class.daemon_threads = True
    return netron


def _start(netron, path):
    # The official ephemeral-port allocator briefly releases its probe socket.
    # Retry only the possible bind race, always using a fresh ephemeral port.
    for attempt in range(5):
        try:
            return netron.start(str(path), address=("127.0.0.1", 0), browse=False)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or attempt == 4:
                raise
    raise RuntimeError("无法分配本地端口。")


def _continue_session():
    line = sys.stdin.buffer.readline(MAX_COMMAND_BYTES + 1)
    if not line:
        return False
    if len(line) > MAX_COMMAND_BYTES:
        raise ProtocolError("invalid_command", "控制命令过长。")
    try:
        command = line.decode("utf-8").strip()
        if not command:
            return True
        if command == "quit":
            return False
        value = json.loads(command)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid_command", "控制命令必须是 UTF-8 JSON 行或 quit。") from error
    if not isinstance(value, dict) or value.get("type") != "quit":
        raise ProtocolError("invalid_command", "不支持的控制命令；请发送 {\"type\":\"quit\"}。")
    return False


def main(argv=None):
    # Preserve an unbuffered protocol descriptor, then redirect both Python
    # prints and native writes to stdout before importing the library.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    protocol = os.fdopen(protocol_fd, "wb", buffering=0)

    def reply(value):
        value["protocol"] = PROTOCOL_VERSION
        protocol.write((json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")) + "\n").encode("utf-8"))

    netron = None
    address = None
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _shutdown_signal)
    try:
        parser = ArgumentParser(description=__doc__)
        parser.add_argument("--model", required=True, help="selected local model path")
        args = parser.parse_args(argv)
        path = _model_path(args.model)
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")
        netron = _load_netron()
        address = _start(netron, path)
        if not (isinstance(address, tuple) and len(address) == 2
                and address[0] == "127.0.0.1" and isinstance(address[1], int)
                and 0 < address[1] < 65536 and netron.status(address)):
            raise RuntimeError("Netron 没有启动有效的本机 HTTP 会话。")
        reply({"type": "ready", "url": "http://127.0.0.1:" + str(address[1]) + "/",
               "model_path": str(path), "version": NETRON_VERSION})
        while _continue_session():
            pass
        return 0
    except (Shutdown, KeyboardInterrupt):
        return 0
    except SystemExit as error:
        return int(error.code or 0)
    except Exception as error:
        code = error.code if isinstance(error, ProtocolError) else "start_failed"
        message = str(error) if isinstance(error, ProtocolError) else "模型结构服务启动失败：" + str(error)
        try:
            reply({"type": "error", "code": code, "message": message})
        except OSError:
            pass  # The controlling application may already have closed stdout.
        return 2
    finally:
        if netron is not None:
            try:
                netron.stop(address)
                deadline = time.monotonic() + 1.0
                while netron.status(address) and time.monotonic() < deadline:
                    time.sleep(0.01)
            except Exception as error:
                print("Netron 停止服务时出现错误：" + str(error), file=sys.stderr)
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
