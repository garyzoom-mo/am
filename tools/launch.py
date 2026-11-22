"""launch.py — AlphaMaster 一键启动 / 停止。

设计原则：
  1. 幂等：已经在跑的服务和训练不会被重复启动（这是之前出现双进程抢
     检查点的根因，必须防住）。
  2. 脱离父进程：Web 和训练都用 DETACHED_PROCESS 启动，关掉这个启动
     窗口不会影响后台任务。
  3. 单一入口：训练统一走 Web 的 /api/training/start，这样监控页能认到
     任务（之前 CLI 直接启动会被误判成「外部进程」）。

用法:
    python tools/launch.py              # 起 Web + 续训，并打开浏览器
    python tools/launch.py --no-train   # 只起 Web
    python tools/launch.py --stop       # 停止训练 + 关闭 Web
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PORT = 18765
DEFAULT_DATA_FILE = r"E:\工作\缠论助手\XAUUSD_M30.parquet"


# ── HTTP 小工具（标准库，不引入 requests 依赖）──────────────────────────
def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 20.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _base(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _is_healthy(port: int) -> bool:
    try:
        _get_json(f"{_base(port)}/api/health", timeout=2.0)
        return True
    except Exception:
        return False


# ── 进程操作 ────────────────────────────────────────────────────────────
# 关键：只加 DETACHED_PROCESS 是不够的。Windows 11 的 Windows Terminal /
# conhost 会把控制台里的进程放进一个 job 对象（JOB_OBJECT_LIMIT_KILL_ON_
# JOB_CLOSE），关窗时整个 job 被销毁，DETACHED 的子进程照样被连带杀掉——
# 实测「关掉黑窗口，Web 服务就掉线」就是这个原因。
# CREATE_BREAKAWAY_FROM_JOB (0x01000000) 让子进程脱离该 job，才能真正存活。
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _detached_flags() -> int:
    if sys.platform != "win32":
        return 0
    return (subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | _CREATE_BREAKAWAY_FROM_JOB)


def _detached_flags_fallback() -> int:
    """若所在 job 不允许 breakaway，CreateProcess 会报拒绝访问，退回两步走。"""
    if sys.platform != "win32":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def _spawn_detached(cmd: list[str], cwd: str, stdout) -> subprocess.Popen:
    env = _child_env()
    try:
        return subprocess.Popen(
            cmd, cwd=cwd,
            stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env,
            creationflags=_detached_flags(),
            close_fds=True,
        )
    except OSError:
        return subprocess.Popen(
            cmd, cwd=cwd,
            stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env,
            creationflags=_detached_flags_fallback(),
            close_fds=True,
        )


def _child_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _start_web(port: int) -> None:
    log_path = ROOT / "logs" / "web_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(log_path, "a", encoding="utf-8", buffering=1)
    # 优先用 pythonw.exe（GUI 子系统）：它天生不依附任何控制台，即使
    # BREAKAWAY 被 job 拒绝，关窗也收不到 CTRL_CLOSE。stdout 已重定向到
    # 日志文件，pythonw 照样能正常写日志。
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    launcher = str(pyw) if pyw.exists() else str(exe)
    _spawn_detached([launcher, "run_web.py", "--port", str(port)], str(ROOT), fp)


def _pids_on_port(port: int) -> list[int]:
    """通过 netstat 找到监听指定端口的 PID（taskkill 用）。"""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=15,
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
    return pids


def _kill_tree(pid: int) -> bool:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True, timeout=20,
        )
        return True
    except Exception:
        return False


# ── 训练状态 ────────────────────────────────────────────────────────────
def _training_overview(port: int) -> dict | None:
    """优先用监控聚合接口（它带「外部进程」识别），失败再退回基础接口。"""
    try:
        return _get_json(f"{_base(port)}/api/monitor/overview", timeout=8.0).get("training", {})
    except Exception:
        pass
    try:
        st = _get_json(f"{_base(port)}/api/training/status", timeout=8.0)
        return {
            "active": bool(st.get("active")),
            "state": (st.get("job") or {}).get("state") or "idle",
            "current_step": st.get("current_step"),
            "train_steps": st.get("train_steps"),
            "best_score": st.get("best_score"),
        }
    except Exception:
        return None


def _fmt_step(t: dict) -> str:
    cur = t.get("current_step")
    tot = t.get("train_steps")
    if cur is None:
        return "步数未知"
    return f"{cur}/{tot}" if tot else str(cur)


# ── 主流程 ──────────────────────────────────────────────────────────────
def do_start(port: int, data_file: str, want_train: bool, open_browser: bool) -> int:
    print("=" * 58)
    print("  AlphaMaster 启动器")
    print("=" * 58)

    # 1. Web 服务
    if _is_healthy(port):
        print(f"[1/3] Web 服务已在运行  →  {_base(port)}")
    else:
        print(f"[1/3] 启动 Web 服务（端口 {port}）…")
        _start_web(port)
        ok = False
        for _ in range(40):
            time.sleep(1.0)
            if _is_healthy(port):
                ok = True
                break
        if not ok:
            print("      ✗ 启动失败，请看 logs/web_server.log")
            return 1
        print(f"      ✓ 已就绪  →  {_base(port)}")

    # 2. 浏览器
    if open_browser:
        webbrowser.open(_base(port))
        print("[2/3] 已打开浏览器")
    else:
        print("[2/3] 跳过打开浏览器")

    # 3. 训练
    ov = _training_overview(port)
    if ov is None:
        print("[3/3] 无法读取训练状态，跳过自动训练")
        return 0

    if ov.get("active"):
        extra = "（外部进程）" if ov.get("external") else ""
        print(f"[3/3] 训练已在运行{extra}：{_fmt_step(ov)}  最优={ov.get('best_score')}")
        print("      不重复启动，避免两个进程抢同一份 checkpoint")
        return 0

    if not want_train:
        print("[3/3] 已指定 --no-train，跳过训练")
        return 0

    print(f"[3/3] 启动训练（自动续训）：{data_file}")
    try:
        res = _post_json(f"{_base(port)}/api/training/start",
                         {"data_file": data_file, "from_scratch": False})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        print(f"      ✗ 启动失败 HTTP {e.code}: {detail}")
        return 1
    except Exception as e:
        print(f"      ✗ 启动失败: {type(e).__name__}: {e}")
        return 1

    job = res.get("job") or {}
    info = res.get("data_file") or {}
    print(f"      ✓ 已提交  {info.get('symbol', '?')} {info.get('timeframe', '')}"
          f"  pid={job.get('pid')}")
    print(f"      日志: {job.get('log_path')}")

    time.sleep(4)
    ov2 = _training_overview(port) or {}
    print(f"      当前进度: {_fmt_step(ov2)}  状态={ov2.get('state')}")
    print()
    print("启动完成。本窗口可以直接关闭，后台任务不受影响。")
    return 0


def do_stop(port: int) -> int:
    print("=" * 58)
    print("  AlphaMaster 停止器")
    print("=" * 58)

    if not _is_healthy(port):
        print(f"Web 服务未在运行（端口 {port}）")
        return 0

    # 1. 训练
    try:
        st = _get_json(f"{_base(port)}/api/training/status", timeout=8.0)
        job = st.get("job") or {}
        pid = job.get("pid")
        if pid:
            print(f"[1/2] 停止训练 pid={pid} …")
            try:
                _post_json(f"{_base(port)}/api/training/stop", {}, timeout=20.0)
            except Exception:
                pass
            _kill_tree(int(pid))
        else:
            print("[1/2] 没有由 Web 管理的训练任务")
    except Exception as e:
        print(f"[1/2] 读取训练状态失败: {e}")

    # 2. Web
    time.sleep(1.0)
    pids = _pids_on_port(port)
    if pids:
        for pid in pids:
            print(f"[2/2] 关闭 Web 服务 pid={pid} …")
            _kill_tree(pid)
    else:
        print("[2/2] 未找到监听进程")

    time.sleep(1.0)
    print("完成。" if not _is_healthy(port) else "警告：端口仍在监听，可能被其他程序占用")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AlphaMaster 一键启动/停止")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    ap.add_argument("--no-train", action="store_true", help="只启动 Web，不自动训练")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--stop", action="store_true", help="停止训练并关闭 Web")
    args = ap.parse_args()

    if args.stop:
        return do_stop(args.port)
    return do_start(args.port, args.data_file, not args.no_train, not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
