"""训练页两种状态的截图与断言：运行中 / 启动失败反馈。"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18890/"
MODE = sys.argv[2] if len(sys.argv) > 2 else "running"


def dump(pg, tag: str) -> None:
    print(f"-- {tag} --")
    print("   按钮:", pg.eval_on_selector("#btnTrain", "el=>[el.textContent, el.disabled]"))
    print("   计划:", pg.inner_text("#planHint"))
    print("   当前任务:", pg.inner_text("#trainNow").replace("\n", " | "))
    print("   进度条:", pg.eval_on_selector("#trainBar", "el=>el.style.width"))
    print("   toast:", repr(pg.inner_text("#toast")), "|", pg.eval_on_selector("#toast", "el=>el.className"))


def main() -> int:
    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        pg = br.new_page(viewport={"width": 1460, "height": 940})
        pg.goto(URL, wait_until="load")
        pg.wait_for_timeout(2000)
        pg.click('.tab[data-tab="train"]')
        pg.wait_for_timeout(600)

        if MODE == "running":
            dump(pg, "训练运行中")
            pg.screenshot(path="logs/cockpit/verify_running.png")
        else:
            # 造一个不存在的数据文件，验证失败反馈
            pg.evaluate("""() => {
                const s = document.querySelector('#dataFile');
                const o = document.createElement('option');
                o.value = 'E:\\\\工作\\\\不存在的数据.parquet';
                o.textContent = 'FAKE 不存在 — 不存在的数据.parquet';
                s.appendChild(o); s.value = o.value;
            }""")
            pg.wait_for_timeout(300)
            pg.fill("#steps", "20")
            pg.click("#btnTrain")
            pg.wait_for_timeout(3000)
            dump(pg, "启动失败")
            pg.screenshot(path="logs/cockpit/verify_fail.png")
        br.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
