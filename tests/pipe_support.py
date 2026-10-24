# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded readiness checks for subprocess pipes on POSIX and Windows."""
import os
import selectors
import time


def wait_readable(process, timeout):
    if os.name != 'nt':
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            return bool(selector.select(timeout))
    import ctypes
    from ctypes import wintypes
    import msvcrt
    peek = ctypes.WinDLL('kernel32', use_last_error=True).PeekNamedPipe
    peek.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                     ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
    peek.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(process.stdout.fileno())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            return False
        if available.value: return True
        if process.poll() is not None: return False
        time.sleep(.01)
    return False
