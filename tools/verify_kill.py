"""验证兜底：强杀 exe（不经过任何清理逻辑）时子进程是否被连带终止。

模拟场景：用户从任务管理器直接结束 AlphaMaster.exe，或进程崩溃。
这时没有机会执行 stop_all，唯一指望就是 Job Object 的 KILL_ON_JOB_CLOSE。
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from cockpit import winproc

BASE = "http://127.0.0.1:18888"


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


def main() -> int:
    exe = Path(__file__).resolve().parents[1] / "AlphaMaster.exe"
    if not exe.exists():
        print("找不到 exe")
        return 2

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
        print("服务未就绪")
        return 1

    child = post("/api/service/start", {"key": "live_btcusdt", "dry_run": True})["pid"]
    time.sleep(9)
    alive_before = winproc.is_alive(child)
    print(f"子进程 pid={child} 启动后存活={alive_before}")

    # 强杀整棵树：不加 /T 只杀 bootloader 会留下子进程，这里模拟最彻底的结束
    subprocess.run(["taskkill", "/F", "/T", "/IM", "AlphaMaster.exe"],
                   capture_output=True, creationflags=winproc.CREATE_NO_WINDOW)
    time.sleep(3)

    alive_after = winproc.is_alive(child)
    print(f"强杀后子进程存活={alive_after}")
    ok = (not alive_after) and alive_before
    print("\n结论:", "PASS 强杀也会连带清理" if ok else "FAIL 有进程逃逸")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
