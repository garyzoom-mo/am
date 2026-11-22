"""AlphaMaster 控制台 —— 单入口启动器。

职责：起本地 API → 打开一个无地址栏的应用窗口 → 盯着这个窗口。
窗口一旦消失（用户点了关闭），先优雅停止所有子进程，再关闭 Job Object
兜底杀干净，最后退出。

“关掉程序就全部停止”由三层保证：
  1. 窗口消失 → 主动 stop_all()（Ctrl+C 让实盘 loop 自己收尾）
  2. stop_all 完成后 job.close()（内核杀掉漏网进程）
  3. 即使本进程被强杀，Job Object 的 KILL_ON_JOB_CLOSE 仍会连带清理
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from . import config as cfgmod
from . import state as state_mod
from . import winproc
from .server import CockpitServer
from .services import Supervisor

EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


class App:
    def __init__(self) -> None:
        self.cfg = cfgmod.load_config()
        self.am = Path(self.cfg["alphamaster"])
        self.log_dir = cfgmod.log_dir(self.am)
        self.log_path = self.log_dir / "cockpit.log"
        self.supervisor = Supervisor(self.cfg, self.log_dir)
        self.port = 0
        self.server: CockpitServer | None = None
        self._browser: subprocess.Popen | None = None
        self._shutting_down = False
        self.no_window = False
        self.force_port: int | None = None

    # ── 日志 ────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass

    # ── 端口 ────────────────────────────────────────────────────────
    def _probe_existing(self) -> str | None:
        """已有实例在跑就返回它的地址，避免开出第二个进程树。"""
        base = int(self.cfg.get("port") or cfgmod.DEFAULT_PORT)
        for port in range(base, base + 5):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.35):
                    pass
            except OSError:
                continue
            import urllib.request
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as r:
                    if b'"ok"' in r.read(64):
                        return f"http://127.0.0.1:{port}/"
            except Exception:  # noqa: BLE001
                continue
        return None

    def _pick_port(self) -> int:
        if self.force_port:
            return self.force_port
        base = int(self.cfg.get("port") or cfgmod.DEFAULT_PORT)
        for port in range(base, base + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("端口被占满，无法启动")

    # ── 窗口 ────────────────────────────────────────────────────────
    def _open_window(self, url: str) -> None:
        exe = next((p for p in EDGE + CHROME if Path(p).exists()), None)
        if not exe:
            self.log("未找到 Edge/Chrome，退回默认浏览器（关窗无法自动停止服务）")
            webbrowser.open(url)
            return

        profile = self.log_dir / "browser_profile"
        profile.mkdir(parents=True, exist_ok=True)
        args = [
            exe,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-mode",
            "--disable-features=Translate,msEdgeWelcomePage",
            "--window-size=1460,940",
        ]
        try:
            self._browser = subprocess.Popen(
                args, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=winproc.CREATE_NO_WINDOW if winproc.IS_WIN else 0,
            )
            self.log(f"已打开应用窗口: {exe}")
        except OSError as exc:
            self.log(f"打开窗口失败: {exc}，退回默认浏览器")
            webbrowser.open(url)

    def _watch_window(self, quit_event: threading.Event) -> str:
        found = False
        miss = 0
        deadline = time.time() + 120.0
        while not quit_event.is_set():
            time.sleep(1.2)
            if quit_event.is_set():
                return "收到退出请求"
            hwnd = winproc.find_window(cfgmod.APP_TITLE)
            if hwnd:
                if not found:
                    self.log(f"窗口已就绪 hwnd={hwnd}")
                found, miss = True, 0
                continue
            if found:
                miss += 1
                if miss >= 2:
                    return "窗口已关闭"
                continue
            if self._browser is not None and self._browser.poll() is not None:
                return "浏览器进程已退出"
            if time.time() > deadline:
                return "窗口未出现（超时）"
        return "收到退出请求"

    # ── 生命周期 ────────────────────────────────────────────────────
    def shutdown(self, reason: str) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.log(f"开始退出：{reason}")
        try:
            results = self.supervisor.stop_all()
            self.log(f"子进程停止结果: {results}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"stop_all 异常: {exc}")
        try:
            if self.server:
                self.server.stop()
        except Exception:  # noqa: BLE001
            pass
        # 兜底：按端口清理（防止有进程脱离了作业对象）
        try:
            for svc in self.supervisor.definitions().values():
                if svc.port:
                    winproc.kill_by_port(svc.port)
        except Exception:  # noqa: BLE001
            pass
        self.log("已退出")
        time.sleep(0.2)
        os._exit(0)

    def run(self) -> int:
        existing = None if self.force_port else self._probe_existing()
        if existing:
            self.log(f"已有实例在运行 {existing}，只打开窗口")
            self._open_window(existing)
            return 0

        self.port = self._pick_port()
        self.server = CockpitServer(self.cfg, self.supervisor, self.port)
        self.server.start()
        self.log(f"服务已启动 {self.server.url}（Job Object={self.supervisor.job.ok}）")

        # 预热一次品种快照，减少首屏等待
        threading.Thread(
            target=lambda: state_mod.symbol_payload(Path(self.cfg["factor_governor"])),
            daemon=True,
        ).start()

        self._open_window(self.server.url)

        if self.no_window:
            self.log("无窗口模式：等待退出请求（Ctrl+C 或 POST /api/quit）")
            try:
                while not self.server.quit_event.wait(1.0):
                    pass
            except KeyboardInterrupt:
                self.shutdown("Ctrl+C")
                return 0
            self.shutdown("收到退出请求")
            return 0

        reason = self._watch_window(self.server.quit_event)
        self.shutdown(reason)
        return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="AlphaMaster 控制台")
    ap.add_argument("--no-window", action="store_true", help="不打开界面，只起服务（调试用）")
    ap.add_argument("--port", type=int, default=None, help="强制端口（调试用）")
    args = ap.parse_args()

    app = App()
    app.no_window = args.no_window
    app.force_port = args.port
    try:
        return app.run()
    except KeyboardInterrupt:
        app.shutdown("Ctrl+C")
        return 0
    except Exception as exc:  # noqa: BLE001
        app.log(f"启动失败: {type(exc).__name__}: {exc}")
        app.shutdown(f"启动失败 {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
