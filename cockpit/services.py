"""服务定义与进程管理。

一个「服务」= 一条常驻命令。所有服务共用 venv 里的 python，区别只在
cwd / 命令行 / 环境变量。停止时先尝试 Ctrl+C 让对端自己收尾，超时才强杀。
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import winproc


@dataclass
class Service:
    key: str
    title: str
    desc: str
    kind: str                      # web | live | train
    cwd: Path
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None
    health_path: str | None = None
    log_name: str = ""


@dataclass
class Running:
    proc: subprocess.Popen
    service: Service
    started_at: float
    log_file: Any = None


class Supervisor:
    def __init__(self, cfg: dict[str, Any], log_dir: Path) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self.job = winproc.JobObject()
        self._running: dict[str, Running] = {}
        self._last_exit: dict[str, int | None] = {}
        # 本次训练的目标品种与步数：进度要从对应品种的历史文件里读，
        # 不能盲取「最近更新的那个」，否则多品种训练时会串台
        self._train_meta: dict[str, Any] = {}

    # ── 服务定义 ────────────────────────────────────────────────────
    def definitions(self) -> dict[str, Service]:
        am = Path(self.cfg["alphamaster"])
        fg = Path(self.cfg["factor_governor"])
        py = str(self.cfg["python"])
        web_port = int(self.cfg.get("web_port", 18765))

        base_env = {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        fg_env = dict(base_env)
        fg_env["PYTHONPATH"] = str(fg / "src")

        svcs: dict[str, Service] = {
            "web": Service(
                key="web", title="训练台", kind="web",
                desc="AlphaMaster Web 控制台：曲线、策略、回测",
                cwd=am, port=web_port, health_path="/api/health",
                cmd=[py, "run_web.py", "--port", str(web_port)],
                env=base_env, log_name="web.log",
            ),
        }

        for key, spec in (self.cfg.get("live") or {}).items():
            if not spec.get("enabled", True):
                continue
            cmd = [
                py, "-m", "factor_governor.executor", "loop",
                "--broker", str(spec.get("broker", "mt5")),
                "--symbol", str(spec["symbol"]),
                "--timeframe", str(spec["timeframe"]),
                "--db", str(spec["db"]),
                *[str(x) for x in spec.get("extra", [])],
            ]
            svcs[f"live_{key}"] = Service(
                key=f"live_{key}",
                title=f"{spec['symbol']} 实盘",
                desc=f"{spec.get('broker', 'mt5')} · {spec['timeframe']} · loop",
                kind="live", cwd=fg, cmd=cmd, env=fg_env,
                log_name=f"{key}.log",
            )
        return svcs

    # ── 状态 ────────────────────────────────────────────────────────
    def _healthy(self, svc: Service, timeout: float = 1.0) -> bool | None:
        if not svc.port or not svc.health_path:
            return None
        url = f"http://127.0.0.1:{svc.port}{svc.health_path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, svc in self.definitions().items():
            run = self._running.get(key)
            alive = bool(run and run.proc.poll() is None)
            if run and not alive and key not in self._last_exit:
                self._last_exit[key] = run.proc.returncode
            out.append({
                "key": key,
                "title": svc.title,
                "desc": svc.desc,
                "kind": svc.kind,
                "running": alive,
                "pid": run.proc.pid if (run and alive) else None,
                "uptime_sec": int(time.time() - run.started_at) if (run and alive) else 0,
                "port": svc.port,
                "healthy": self._healthy(svc) if alive else False,
                "last_exit": self._last_exit.get(key),
                "log": svc.log_name,
            })
        return out

    def is_running(self, key: str) -> bool:
        run = self._running.get(key)
        return bool(run and run.proc.poll() is None)

    # ── 启停 ────────────────────────────────────────────────────────
    def start(self, key: str, extra_args: list[str] | None = None) -> dict[str, Any]:
        svcs = self.definitions()
        svc = svcs.get(key)
        if svc is None:
            return {"ok": False, "error": f"未知服务 {key}"}
        if self.is_running(key):
            return {"ok": True, "pid": self._running[key].proc.pid, "already": True}

        cmd = list(svc.cmd) + list(extra_args or [])
        env = os.environ.copy()
        env.update(svc.env)
        log_path = self.log_dir / (svc.log_name or f"{key}.log")
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        log_file.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 {key} =====\n"
            f"     {' '.join(cmd)}\n"
        )
        log_file.flush()

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(svc.cwd),
                stdout=log_file, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=winproc.CHILD_FLAGS,
                startupinfo=winproc.child_startupinfo(),
                close_fds=True,
            )
        except OSError as exc:
            log_file.write(f"[启动失败] {exc}\n")
            log_file.close()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.job.assign(proc.pid)
        self._running[key] = Running(proc, svc, time.time(), log_file)
        self._last_exit.pop(key, None)
        return {"ok": True, "pid": proc.pid, "cmd": cmd}

    def stop(self, key: str, grace: float = 12.0) -> dict[str, Any]:
        run = self._running.get(key)
        if not run or run.proc.poll() is not None:
            self._running.pop(key, None)
            return {"ok": True, "already": True}

        pid = run.proc.pid
        graceful = winproc.send_ctrl_c(pid)
        try:
            run.proc.wait(timeout=grace)
            how = "ctrl-c"
        except subprocess.TimeoutExpired:
            winproc.kill_tree(pid)
            try:
                run.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
            how = "killed"

        self._last_exit[key] = run.proc.returncode
        if run.log_file:
            try:
                run.log_file.write(f"===== {key} 已停止（{how}）=====\n")
                run.log_file.close()
            except OSError:
                pass
        self._running.pop(key, None)
        return {"ok": True, "how": how, "code": run.proc.returncode, "graceful": graceful}

    def stop_all(self, grace: float = 10.0) -> list[dict[str, Any]]:
        results = []
        # 实盘先停（涉及持仓），再停训练，最后停 web
        order = [k for k in self._running if k.startswith("live_")]
        order += [k for k in ("train",) if k in self._running]
        order += [k for k in self._running if k not in order]
        for key in order:
            try:
                results.append({"key": key, **self.stop(key, grace=grace)})
            except Exception as exc:  # noqa: BLE001
                results.append({"key": key, "ok": False, "error": str(exc)})
        self.job.close()
        return results

    # ── 日志 ────────────────────────────────────────────────────────
    def tail(self, key: str, n: int = 200) -> list[str]:
        svcs = self.definitions()
        svc = svcs.get(key)
        name = svc.log_name if svc else None
        if key == "train":
            name = "train.log"
        if not name:
            return []
        path = self.log_dir / name
        if not path.exists():
            return []
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                block = min(size, max(4096, n * 220))
                fh.seek(size - block)
                raw = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        lines = raw.splitlines()
        return lines[-n:]

    # ── 训练（单独处理：命令行是动态的）──────────────────────────────
    def start_training(self, data_file: str, steps: int | None,
                       cost: float | None, from_scratch: bool) -> dict[str, Any]:
        if self.is_running("train"):
            return {"ok": False, "error": "已有训练在跑，先停止"}

        am = Path(self.cfg["alphamaster"])
        fg = Path(self.cfg["factor_governor"])
        py = str(self.cfg["python"])
        script = fg / "tools" / "train_with_cost.py"
        if not script.exists():
            return {"ok": False, "error": f"找不到训练入口 {script}"}
        if not Path(data_file).exists():
            return {"ok": False, "error": f"数据文件不存在 {data_file}"}

        cmd = [py, str(script), "--data-file", data_file]
        if steps:
            cmd += ["--steps", str(int(steps))]
        if cost is not None:
            cmd += ["--cost", str(float(cost))]
        if from_scratch:
            cmd.append("--from-scratch")

        svc = Service(
            key="train", title="训练任务", kind="train", desc="模型训练",
            cwd=am, cmd=cmd,
            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            log_name="train.log",
        )
        # 复用统一启动逻辑，但命令行是本次动态生成的
        env = os.environ.copy()
        env.update(svc.env)
        log_path = self.log_dir / "train.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        log_file.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动训练 =====\n"
            f"     数据 {data_file}\n"
            f"     步数 {steps or '默认'}   成本 {cost if cost is not None else '默认'}   "
            f"{'从零开始' if from_scratch else '续训'}\n"
        )
        log_file.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(am),
                stdout=log_file, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env,
                creationflags=winproc.CHILD_FLAGS,
                startupinfo=winproc.child_startupinfo(),
                close_fds=True,
            )
        except OSError as exc:
            log_file.write(f"[启动失败] {exc}\n")
            log_file.close()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.job.assign(proc.pid)
        self._running["train"] = Running(proc, svc, time.time(), log_file)
        self._last_exit.pop("train", None)

        import re
        m = re.match(r"^([A-Za-z0-9]+)_(?:M\d+|H\d+|D\d+)\.parquet$", Path(data_file).name, re.I)
        self._train_meta = {
            "symbol": m.group(1).upper() if m else None,
            "target_steps": int(steps) if steps else None,
            "from_scratch": from_scratch,
            "cost": cost,
            "data_file": data_file,
        }
        return {"ok": True, "pid": proc.pid, "symbol": self._train_meta["symbol"]}

    def training_status(self) -> dict[str, Any]:
        run = self._running.get("train")
        if not run:
            return {"running": False, "managed": False, "external": False}
        alive = run.proc.poll() is None
        return {
            "running": alive,
            "managed": alive,
            "external": False,
            "pid": run.proc.pid,
            "uptime_sec": int(time.time() - run.started_at) if alive else 0,
            "exit_code": None if alive else run.proc.returncode,
            **self._train_meta,
        }
