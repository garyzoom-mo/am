"""复现「点击开始训练没反应」：真浏览器点按钮，抓网络/控制台/toast。"""
from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18888/"


def main() -> int:
    reqs: list[tuple[str, str, str]] = []
    console: list[str] = []
    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        pg = br.new_page(viewport={"width": 1460, "height": 940})
        pg.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))

        def on_resp(r):
            if "/api/" in r.url:
                body = ""
                try:
                    body = r.text()[:300]
                except Exception:
                    pass
                reqs.append((r.request.method, r.url.split("18888")[-1], f"{r.status} {body}"))

        pg.on("response", on_resp)
        pg.goto(URL, wait_until="load")
        pg.wait_for_timeout(2500)

        # 切到训练页
        pg.click('.tab[data-tab="train"]')
        pg.wait_for_timeout(800)
        print("== 训练页初始 ==")
        print("  按钮 disabled:", pg.eval_on_selector("#btnTrain", "el=>el.disabled"))
        print("  下拉值:", pg.eval_on_selector("#dataFile", "el=>el.value"))
        print("  选项数:", pg.eval_on_selector("#dataFile", "el=>el.options.length"))
        print("  当前任务:", pg.inner_text("#trainNow"))

        # 填参数：步数 30，方便快速验证链路
        pg.fill("#steps", "30")
        pg.fill("#cost", "0.0005")
        reqs.clear()
        print("== 点击开始训练 ==")
        pg.click("#btnTrain")
        pg.wait_for_timeout(900)
        print("  toast:", repr(pg.inner_text("#toast")), "class=", pg.eval_on_selector("#toast", "el=>el.className"))
        for m, u, b in reqs:
            print(f"  {m} {u} -> {b}")
        pg.wait_for_timeout(4000)
        print("  3s 后 toast:", repr(pg.inner_text("#toast")))
        print("  按钮 disabled:", pg.eval_on_selector("#btnTrain", "el=>el.disabled"))
        print("  当前任务:", pg.inner_text("#trainNow").replace("\n", " | "))
        print("  进度条宽度:", pg.eval_on_selector("#trainBar", "el=>el.style.width"))
        print("  日志尾部:", pg.inner_text("#trainLog")[-400:].replace("\n", " | "))
        pg.screenshot(path="logs/cockpit/probe_train.png")
        if console:
            print("== console ==")
            for c in console[-10:]:
                print("  ", c)
        br.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
