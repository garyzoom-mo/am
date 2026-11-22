"""Windows 进程管控：Job Object、优雅停止、进程树强杀、窗口探测。

三件事是「关掉程序就全部停止」的技术保证：

1. **Job Object + KILL_ON_JOB_CLOSE**
   把所有子进程塞进一个作业对象。只要本进程消失（正常退出、崩溃、被任务
   管理器结束都算），内核就会连带杀掉作业内所有进程。这是唯一不依赖
   「我们记得去清理」的兜底。

2. **模拟 Ctrl+C 优雅停止**
   实盘 loop 只在 `KeyboardInterrupt` 里做清理（写 stopped 心跳、平仓判断），
   直接 TerminateProcess 会跳过这些逻辑。做法是 AttachConsole 到子进程的
   隐藏控制台，再 GenerateConsoleCtrlEvent 投递 CTRL_C，等它自己退。

3. **窗口探测**
   exe 本身没有窗口，界面是 Edge 的 --app 窗口。用 FindWindowW 按标题找，
   找不到就是用户把窗口关了。
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes

IS_WIN = sys.platform == "win32"

# ── 常量 ────────────────────────────────────────────────────────────────
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1

# 子进程创建标志：
#   CREATE_NEW_PROCESS_GROUP  让它成为新进程组，Ctrl+C 可以只投给它
#   CREATE_NEW_CONSOLE        给它**自己的**控制台窗口（配合 STARTUPINFO 隐藏）
#
# 为什么必须有独立控制台：GenerateConsoleCtrlEvent 只能投递给「与调用者共享
# 同一个控制台」的进程。早先只加 CREATE_NO_WINDOW，子进程往往沿用父进程的
# 控制台，一次 Ctrl+C 会把同控制台的所有进程（包括启动器和别的训练）一起
# 打死 —— 实测误杀了正在跑的 XAUUSD 训练。
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NEW_CONSOLE = 0x00000010
CREATE_NO_WINDOW = 0x08000000
CHILD_FLAGS = (CREATE_NEW_PROCESS_GROUP | CREATE_NEW_CONSOLE) if IS_WIN else 0


def child_startupinfo():
    """隐藏子进程的控制台窗口（要 CREATE_NEW_CONSOLE 时才有效）。"""
    if not IS_WIN:
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _k32():
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.CreateJobObjectW.restype = wintypes.HANDLE
    dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    dll.SetInformationJobObject.restype = wintypes.BOOL
    dll.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    dll.AssignProcessToJobObject.restype = wintypes.BOOL
    dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    dll.OpenProcess.restype = wintypes.HANDLE
    dll.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    dll.CloseHandle.restype = wintypes.BOOL
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.GetExitCodeProcess.restype = wintypes.BOOL
    dll.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    dll.AttachConsole.restype = wintypes.BOOL
    dll.AttachConsole.argtypes = [wintypes.DWORD]
    dll.FreeConsole.restype = wintypes.BOOL
    dll.FreeConsole.argtypes = []
    dll.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    dll.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    dll.SetConsoleCtrlHandler.restype = wintypes.BOOL
    dll.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    return dll


class JobObject:
    """把子进程绑在一起，本进程一死它们全死。"""

    def __init__(self) -> None:
        self._handle = None
        if not IS_WIN:
            return
        try:
            k32 = _k32()
            self._handle = k32.CreateJobObjectW(None, None)
            if not self._handle:
                return
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = k32.SetInformationJobObject(
                self._handle, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info),
            )
            if not ok:
                self._handle = None
        except OSError:
            self._handle = None

    @property
    def ok(self) -> bool:
        return self._handle is not None

    def assign(self, pid: int) -> bool:
        if not self._handle or not IS_WIN:
            return False
        try:
            k32 = _k32()
            h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            if not h:
                return False
            try:
                return bool(k32.AssignProcessToJobObject(self._handle, h))
            finally:
                k32.CloseHandle(h)
        except OSError:
            return False

    def close(self) -> None:
        """关闭句柄 → 内核杀掉作业内所有进程。"""
        if not self._handle:
            return
        try:
            _k32().CloseHandle(self._handle)
        except OSError:
            pass
        self._handle = None


def is_alive(pid: int) -> bool:
    if not pid:
        return False
    if not IS_WIN:
        try:
            import os
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        k32 = _k32()
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    except OSError:
        return False


def send_ctrl_c(pid: int) -> bool:
    """向指定进程组投递 CTRL_BREAK，触发子进程的 KeyboardInterrupt 分支。

    为什么是 BREAK 而不是 C —— 实测三种组合（子进程用 CREATE_NEW_CONSOLE
    独立控制台，避免误伤同一控制台上的其他进程）：

        独立控制台 + CTRL_C 广播        → 投递成功但子进程毫无反应
        独立控制台 + 新进程组 + CTRL_C  → 同上（新进程组默认禁用 Ctrl+C）
        独立控制台 + 新进程组 + CTRL_BREAK 定向到组 → 子进程立刻收到

    Python 对 SIGBREAK 的默认动作是直接终止，所以 factor_governor.executor
    里把 SIGBREAK 映射成了 KeyboardInterrupt，才能走原有清理逻辑。
    """
    if not IS_WIN:
        return False
    try:
        k32 = _k32()
        k32.FreeConsole()          # 先断开自己，否则 AttachConsole 会失败
        if not k32.AttachConsole(pid):
            return False
        try:
            k32.SetConsoleCtrlHandler(None, True)      # 自己忽略，双保险
            ok = bool(k32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid))
        finally:
            k32.FreeConsole()
            k32.SetConsoleCtrlHandler(None, False)
        return ok
    except OSError:
        return False


def kill_tree(pid: int, timeout: float = 20.0) -> bool:
    if not pid:
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, timeout=timeout,
            creationflags=CREATE_NO_WINDOW if IS_WIN else 0,
        )
        return True
    except Exception:
        return False


def find_window(title: str) -> int:
    """按标题精确查找窗口，返回 HWND（0 表示不存在）。"""
    if not IS_WIN:
        return 0
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.restype = wintypes.HWND
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        return int(user32.FindWindowW(None, title) or 0)
    except OSError:
        return 0


def kill_by_port(port: int) -> list[int]:
    """兜底：按监听端口找进程并杀掉（防子进程逃逸出作业对象）。"""
    if not IS_WIN:
        return []
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
    except Exception:
        return []
    pids: list[int] = []
    needle = f":{port}"
    for line in out.splitlines():
        if "LISTENING" not in line.upper() or needle not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    for pid in pids:
        kill_tree(pid)
    return pids
