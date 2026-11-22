"""服务定义与进程管理。

一个「服务」= 一条常驻命令。所有服务共用 venv 里的 python，区别只在
cwd / 命令行 / 环境变量。停止时先尝试 Ctrl+C 让对端自己收尾，超时才强杀。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import winproc

# 「本次追加步数」留空时的默认值。用「追加」而不是「总目标」的原因见
# start_training()：AlphaMaster 的 --steps 是总步数，续训时若总步数小于检查点
# 已有的步数，train_file 会打印「已完成全部 N 步，跳过训练」然后立刻退出。
DEFAULT_ADD_STEPS = 2000

# 进度行形如「[352/400] 新公式=…」
_PROGRESS_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
_TOTAL_RE = re.compile(r"训练步数[:：]\s*(\d+)")
_CKPT_RE = re.compile(r"_step_(\d+)\.pt$", re.I)
_START_MARK = "启动训练 ====="


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
        # 最近一次训练的结局。进程退出后 _running 会清空，界面还得知道
        # 「上一把是失败还是正常结束」，否则只能看到状态跳回空闲
        self._train_exit: int | None = None
        self._train_error: str | None = None
        self._train_exited_at: float | None = None
        self._train_state_file = log_dir / "train_state.json"
        self._load_train_state()

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
        if key == "train":
            # 手动停止不算失败
            self._set_train_outcome(run.proc.returncode, None)
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

    # ── 训练结局（落盘，重启控制台后仍知道上一把是跑成还是失败）───────
    def _load_train_state(self) -> None:
        try:
            raw = json.loads(self._train_state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        self._train_exit = raw.get("exit")
        self._train_error = raw.get("error")
        self._train_exited_at = raw.get("exited_at")
        meta = raw.get("meta")
        if isinstance(meta, dict):
            self._train_meta = meta

    def _save_train_state(self) -> None:
        try:
            self._train_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._train_state_file.write_text(json.dumps({
                "exit": self._train_exit,
                "error": self._train_error,
                "exited_at": self._train_exited_at,
                "meta": self._train_meta,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _set_train_outcome(self, code: int | None, err: str | None) -> None:
        self._train_exit = code
        self._train_error = err
        self._train_exited_at = time.time()
        self._save_train_state()

    def _fail(self, msg: str) -> dict[str, Any]:
        """参数/环境没通过时也要在界面上留下痕迹，不能只有 2 秒 toast。

        不动 _train_exited_at：那是「真进程退出」的时间，用来判断
        训练历史文件是不是我们自己刚写的。
        """
        self._train_error = msg
        self._save_train_state()
        return {"ok": False, "error": msg}

    # ── 训练（单独处理：命令行是动态的）──────────────────────────────
    def _log_segment(self, n: int = 400) -> list[str]:
        """只取「最后一次启动训练」之后的日志行。

        train.log 是追加写的，直接 tail 会把上一次训练的 [N/M] 混进来，
        新任务刚起来那几秒就会显示上一轮的步数。
        """
        lines = self.tail("train", n)
        cut = 0
        for i, ln in enumerate(lines):
            if _START_MARK in ln:
                cut = i
        return lines[cut:]

    def train_progress(self) -> dict[str, Any]:
        """从训练日志里抠当前步数，作为历史文件还没落盘时的兜底。"""
        seg = self._log_segment(600)
        step = total = None
        for ln in seg:
            m = _PROGRESS_RE.search(ln)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
        if total is None:
            for ln in seg:
                m = _TOTAL_RE.search(ln)
                if m:
                    total = int(m.group(1))
        if step is None and total is not None and any("开始 Alpha" in l for l in seg):
            step = 0
        return {"step": step, "total": total}

    @staticmethod
    def _symbol_of(data_file: str) -> str | None:
        m = re.match(r"^(.+?)_(?:M\d+|H\d+|D\d+)\.parquet$", Path(data_file).name, re.I)
        return m.group(1).upper() if m else None

    @staticmethod
    def _ckpt_step(am: Path, symbol: str | None) -> int:
        """检查点已有的步数。

        train_file.py 用 `start_step >= TRAIN_STEPS` 判断「已经练完了」，
        所以启动前必须先知道这个数，否则传 2000 而检查点已在 3200 时，
        训练会在 10 秒内打印「跳过训练」直接退出（退出码 0，界面看着像没启动）。
        """
        if not symbol:
            return 0
        best = 0
        ck_dir = am / "checkpoints"
        if not ck_dir.is_dir():
            return 0
        try:
            for p in ck_dir.glob(f"ckpt_{symbol}_step_*.pt"):
                m = _CKPT_RE.search(p.name)
                if m:
                    best = max(best, int(m.group(1)))
        except OSError:
            return best
        return best

    @staticmethod
    def _evaluate_exit(code: int | None, seg: list[str]) -> tuple[bool, str | None]:
        """判断进程退出是「正常收工」还是「没练起来」。

        返回 (是否正常, 给用户看的一句话)。
        """
        joined = "\n".join(seg)
        tail = [l.strip() for l in seg if l.strip()]

        # 先看最要命的一种：目标步数已经达成，进程走正常路径退出 0，
        # 界面上分不出「训练完了」和「根本没练」
        for ln in reversed(seg):
            if "跳过训练" in ln:
                return False, ("目标步数已达成，本次没有真正训练"
                               "（检查点步数超前，请加大步数或勾选从零开始）")
        for ln in reversed(seg):
            s = ln.strip()
            if not s:
                continue
            if s.startswith(("[错误]", "[中止]", "[启动失败]")):
                return False, s
        if "Traceback (most recent call last)" in joined:
            return False, "训练进程异常退出：" + (tail[-1] if tail else f"exit={code}")
        if code == 0:
            return True, None
        return False, f"训练进程退出（exit={code}）：" + (tail[-1] if tail else "无输出")

    def training_plan(self, data_file: str, add_steps: int | None = None,
                      from_scratch: bool = False) -> dict[str, Any]:
        """给界面算「从第几步练到第几步」，避免用户填了早已越过的总步数。"""
        am = Path(self.cfg["alphamaster"])
        symbol = self._symbol_of(data_file) if data_file else None
        try:
            add = int(add_steps) if add_steps else DEFAULT_ADD_STEPS
        except (TypeError, ValueError):
            add = DEFAULT_ADD_STEPS
        add = max(1, add)
        start = 0 if from_scratch else self._ckpt_step(am, symbol)
        return {
            "symbol": symbol,
            "start_step": start,
            "add_steps": add,
            "total_steps": start + add,
            "from_scratch": from_scratch,
            "exists": bool(data_file) and Path(data_file).exists(),
        }

    def start_training(self, data_file: str, steps: int | None,
                       cost: float | None, from_scratch: bool) -> dict[str, Any]:
        if self.is_running("train"):
            return self._fail("已有训练在跑，先停止")

        am = Path(self.cfg["alphamaster"])
        fg = Path(self.cfg["factor_governor"])
        py = str(self.cfg["python"])
        script = fg / "tools" / "train_with_cost.py"
        if not script.exists():
            return self._fail(f"找不到训练入口 {script}")
        if not str(data_file or "").strip():
            return self._fail("没有选择数据文件")
        data_path = Path(data_file)
        if not data_path.exists():
            return self._fail(f"数据文件不存在 {data_file}")

        # 步数按「本次追加」解释：录入 2000 就是在这个检查点之上再练 2000 步。
        # AlphaMaster 的 --steps 是总目标步数，所以这里换算成 start + add。
        try:
            add_steps = int(steps) if steps not in (None, "") else 0
        except (TypeError, ValueError):
            return self._fail(f"步数不是整数：{steps!r}")
        if add_steps < 0:
            return self._fail("步数不能为负")
        if add_steps == 0:
            add_steps = DEFAULT_ADD_STEPS
        if cost is not None:
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                return self._fail(f"成本率不是数字：{cost!r}")
            if cost < 0:
                return self._fail("成本率不能为负")

        symbol = self._symbol_of(data_file)
        start_step = 0 if from_scratch else self._ckpt_step(am, symbol)
        total_steps = start_step + add_steps

        cmd = [py, str(script), "--data-file", str(data_path),
               "--steps", str(total_steps)]
        if cost is not None:
            cmd += ["--cost", str(cost)]
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
            f"     品种 {symbol or '?'}   本次追加 {add_steps} 步   "
            f"{'从零开始' if from_scratch else f'从第 {start_step} 步续训'}   "
            f"目标总步数 {total_steps}   成本 {cost if cost is not None else '默认'}\n"
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
            self._set_train_outcome(None, f"无法创建训练进程：{type(exc).__name__}: {exc}")
            return {"ok": False, "error": self._train_error}

        self.job.assign(proc.pid)
        self._running["train"] = Running(proc, svc, time.time(), log_file)
        self._last_exit.pop("train", None)
        self._set_train_outcome(None, None)

        self._train_meta = {
            "symbol": symbol,
            "target_steps": total_steps,     # 总目标（与前端的「追加步数」区分）
            "add_steps": add_steps,
            "start_step": start_step,
            "from_scratch": from_scratch,
            "cost": cost,
            "data_file": str(data_path),
        }
        self._save_train_state()

        # 存活探测：真训练不可能 2 秒内结束。秒退说明参数/环境有问题，
        # 这时必须把原因当场回给界面，而不是让它「看起来启动了」。
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return {"ok": True, "pid": proc.pid, "symbol": symbol,
                    "start_step": start_step, "add_steps": add_steps,
                    "total_steps": total_steps}

        seg = self._log_segment(120)
        ok, err = self._evaluate_exit(proc.returncode, seg)
        self._running.pop("train", None)
        self._last_exit["train"] = proc.returncode
        self._set_train_outcome(proc.returncode, err)
        try:
            log_file.write(f"===== 训练启动后立即退出：{err or '正常结束'}"
                           f"（exit={proc.returncode}）=====\n")
            log_file.close()
        except OSError:
            pass
        return {
            "ok": False,
            "error": err or "训练进程启动后立即退出（exit=0），未进入训练状态",
            "exit_code": proc.returncode,
            "log_tail": [l for l in seg if l.strip()][-8:],
        }

    def training_status(self) -> dict[str, Any]:
        run = self._running.get("train")
        meta = dict(self._train_meta)
        if not run:
            return {
                "running": False,
                "managed": False,
                "external": False,
                "last_exit": self._train_exit,
                "last_error": self._train_error,
                "last_exit_age": (time.time() - self._train_exited_at)
                                 if self._train_exited_at else None,
                **meta,
            }
        alive = run.proc.poll() is None
        if not alive:
            code = run.proc.returncode
            if self._train_exit is None:
                seg = self._log_segment(120)
                ok, err = self._evaluate_exit(code, seg)
                self._set_train_outcome(code, err)
                try:
                    run.log_file.write(
                        f"===== 训练进程已退出：{err or '正常结束'}（exit={code}）=====\n")
                    run.log_file.close()
                except OSError:
                    pass
            self._running.pop("train", None)
            self._last_exit["train"] = code
            return {
                "running": False,
                "managed": False,
                "external": False,
                "pid": run.proc.pid,
                "exit_code": code,
                "last_exit": code,
                "last_error": self._train_error,
                "last_exit_age": (time.time() - self._train_exited_at)
                                 if self._train_exited_at else None,
                **meta,
            }
        return {
            "running": True,
            "managed": True,
            "external": False,
            "pid": run.proc.pid,
            "uptime_sec": int(time.time() - run.started_at),
            "phase": "starting" if time.time() - run.started_at < 20 else "running",
            "exit_code": None,
            "last_exit": None,
            "last_error": None,
            **meta,
        }
