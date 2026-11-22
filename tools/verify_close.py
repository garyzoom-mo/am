"""端到端验证：关窗口 → 子进程是否全部退出。

流程：起一个 BTC dry-run → 记录 pid → 给窗口发 WM_CLOSE（等价于点右上角 X）
→ 观察 exe 与子进程是否消失、端口是否释放。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import time
import urllib.request
from ctypes import wintypes

from cockpit import winproc

BASE = "http://127.0.0.1:18888"
WM_CLOSE = 0x0010


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Cockpit": "1"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode())


def exe_pids() -> list[int]:
    # Windows 的 tasklist 输出是本地代码页（中文系统为 GBK），不能按 utf-8 解码
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq AlphaMaster.exe", "/FO", "CSV", "/NH"],
        capture_output=True, creationflags=winproc.CREATE_NO_WINDOW,
    ).stdout.decode("mbcs", errors="replace")
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[0].lower().startswith("alphamaster"):
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def main() -> int:
    from pathlib import Path
    exe = Path(__file__).resolve().parents[1] / "AlphaMaster.exe"
    if not exe.exists():
        print(f"找不到 {exe}，先跑 python build_exe.py")
        return 2
    if exe_pids():
        print("已有 AlphaMaster.exe 在运行，先关掉再测")
        return 2

    print(f"启动 {exe.name} …")
    proc = subprocess.Popen([str(exe)], cwd=str(exe.parent),
                            creationflags=winproc.CHILD_FLAGS,
                            startupinfo=winproc.child_startupinfo())
    for _ in range(60):
        time.sleep(1.0)
        try:
            get("/health")
            break
        except Exception:
            continue
    else:
        print("服务未就绪（60s 超时）")
        return 1
    print("服务已就绪")

    before = exe_pids()
    print(f"exe 进程: {before}")

    r = post("/api/service/start", {"key": "live_btcusdt", "dry_run": True})
    child = r.get("pid")
    print(f"启动子服务 pid={child}")
    time.sleep(10)

    st = get("/api/state")
    svc = next(s for s in st["services"] if s["key"] == "live_btcusdt")
    print(f"子服务状态 running={svc['running']} pid={svc['pid']}")

    hwnd = winproc.find_window("AlphaMaster 控制台")
    print(f"窗口 HWND={hwnd}，发送 WM_CLOSE …")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)

    deadline = time.time() + 40
    gone_at = None
    while time.time() < deadline:
        time.sleep(1.0)
        if not exe_pids():
            gone_at = time.time()
            break
    print(f"exe 退出: {'是' if gone_at else '否（超时）'}")

    time.sleep(2)
    child_alive = winproc.is_alive(child) if child else None
    port_ok = True
    try:
        get("/health")
    except Exception:
        port_ok = False

    print(f"子进程存活: {child_alive}")
    print(f"端口 18888 仍在监听: {port_ok}")
    print(f"残留 exe: {exe_pids()}")
    ok = bool(gone_at) and not child_alive and not port_ok
    print("\n结论:", "PASS 关窗后全部退出" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
