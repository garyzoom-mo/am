"""本地 HTTP 服务：给控制台页面提供 API。

只监听 127.0.0.1。所有 POST 要求带 `X-Cockpit: 1` 头 —— 浏览器跨源请求会先
发预检，而我们不应答预检，别处的网页就没法偷偷调这些接口（防止开着控制台时
被别的页面 POST 一个「全部平仓」）。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import state as state_mod

PAGE_FILE = Path(__file__).with_name("page.html")


class CockpitServer:
    def __init__(self, cfg: dict[str, Any], supervisor, port: int) -> None:
        self.cfg = cfg
        self.sup = supervisor
        self.port = port
        self.quit_event = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "AlphaMasterCockpit/1.0"

            # ── 工具 ────────────────────────────────────────────
            def log_message(self, fmt: str, *args) -> None:  # noqa: A003
                pass  # 静默：请求太密，写了会刷屏

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _json(self, obj: Any, code: int = 200) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return {}

            def _guard(self) -> bool:
                if self.headers.get("X-Cockpit") != "1":
                    self._json({"ok": False, "error": "缺少 X-Cockpit 头"}, 403)
                    return False
                return True

            # ── GET ─────────────────────────────────────────────
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path

                if path in ("/", "/index.html"):
                    try:
                        html = PAGE_FILE.read_text(encoding="utf-8")
                    except OSError as exc:
                        self._send(500, f"page.html 读取失败: {exc}".encode(), "text/plain; charset=utf-8")
                        return
                    self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                    return

                if path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                    return

                if path == "/health":
                    self._json({"ok": True})
                    return

                if path == "/api/state":
                    self._json(state_mod.build_state(outer.sup, outer.cfg))
                    return

                if path == "/api/train/inspect":
                    qs = parse_qs(parsed.query)
                    data_file = (qs.get("data_file") or [""])[0]
                    self._json(outer.sup.inspect_data(data_file))
                    return

                if path == "/api/train/plan":
                    qs = parse_qs(parsed.query)
                    data_file = (qs.get("data_file") or [""])[0]
                    raw_add = (qs.get("add") or [""])[0]
                    try:
                        add = int(raw_add) if raw_add else None
                    except ValueError:
                        add = None
                    self._json(outer.sup.training_plan(
                        data_file, add_steps=add,
                        from_scratch=(qs.get("fs") or ["0"])[0] in ("1", "true", "yes"),
                    ))
                    return

                if path == "/api/log":
                    qs = parse_qs(parsed.query)
                    key = (qs.get("key") or ["web"])[0]
                    try:
                        n = min(int((qs.get("n") or ["200"])[0]), 2000)
                    except ValueError:
                        n = 200
                    self._json({"key": key, "lines": outer.sup.tail(key, n)})
                    return

                self._json({"ok": False, "error": "not found"}, 404)

            # ── POST ────────────────────────────────────────────
            def do_POST(self) -> None:  # noqa: N802
                if not self._guard():
                    return
                path = urlparse(self.path).path
                body = self._read_json()

                if path == "/api/service/start":
                    key = str(body.get("key") or "")
                    extra = ["--dry-run"] if body.get("dry_run") else []
                    self._json(outer.sup.start(key, extra_args=extra))
                    return

                if path == "/api/service/stop":
                    key = str(body.get("key") or "")
                    self._json(outer.sup.stop(key))
                    return

                if path == "/api/train/start":
                    res = outer.sup.start_training(
                        data_file=str(body.get("data_file") or ""),
                        steps=body.get("steps"),
                        cost=body.get("cost"),
                        from_scratch=bool(body.get("from_scratch")),
                    )
                    self._json(res)
                    return

                if path == "/api/train/stop":
                    self._json(outer.sup.stop("train", grace=15.0))
                    return

                if path == "/api/quit":
                    self._json({"ok": True, "stopping": True})
                    outer.quit_event.set()
                    return

                if path == "/api/service/stop-all":
                    self._json({"ok": True, "results": outer.sup.stop_all()})
                    return

                self._json({"ok": False, "error": "not found"}, 404)

            def do_OPTIONS(self) -> None:  # noqa: N802
                # 不应答预检：跨源请求会在这一步失败
                self._send(403, b"", "text/plain")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="cockpit-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        self._httpd = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"
