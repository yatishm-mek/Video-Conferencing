"""Keeps Windows from slowing this process down while another window is in front.

Windows 11 throttles background processes (EcoQoS: lower CPU speed, coarser timers),
which makes training and speech generation much slower when their window is not in
front. Only the standard library is used.
"""

from __future__ import annotations

import ctypes
import logging
import sys

logger = logging.getLogger("full-speed")

_PROCESS_POWER_THROTTLING = 4               # PROCESS_INFORMATION_CLASS
_THROTTLE_EXECUTION_SPEED = 0x1
_THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4
_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


class _ThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def run_at_full_speed() -> None:
    """Opt this process out of background throttling and raise its priority a little."""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    process = kernel32.GetCurrentProcess()
    # The control mask names the settings we decide; a clear state bit turns them off.
    state = _ThrottlingState(1, _THROTTLE_EXECUTION_SPEED | _THROTTLE_IGNORE_TIMER_RESOLUTION, 0)
    if not kernel32.SetProcessInformation(process, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)):
        logger.warning("could not turn off power throttling (error %d)", ctypes.get_last_error())
    if not kernel32.SetPriorityClass(process, _ABOVE_NORMAL_PRIORITY_CLASS):
        logger.warning("could not raise process priority (error %d)", ctypes.get_last_error())
