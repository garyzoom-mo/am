"""Read-only aggregation for the monitoring dashboard.

The monitor page deliberately keeps all execution paths untouched:
it only reads the existing training / backtest / realtime managers and,
when available, the MT5 terminal account snapshot.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from web.backtest_manager import backtest_manager
from web.progress import get_symbol_progress
from web.realtime_manager import realtime_manager
from web.settings import load_settings
from web.training_manager import training_manager
from web.training_time import get_training_time_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_MT5_TTL = 5.0
_MT5_ERROR_TTL = 30.0
_mt5_lock = threading.Lock()
_mt5_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_from_ts(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return None


def _mt5_snapshot() -> dict[str, Any]:
    """Return a cached MT5 account/positions snapshot.

    ``mt5.initialize()`` can take a second or two when the terminal is not
    running, so failures are cached for longer than successful reads.
    """
    now = time.time()
    with _mt5_lock:
        cached = _mt5_cache.get("data")
        if cached is not None and now - float(_mt5_cache.get("ts") or 0) < (
            _MT5_TTL if cached.get("available") else _MT5_ERROR_TTL
        ):
            return cached

        data = _read_mt5()
        _mt5_cache["ts"] = now
        _mt5_cache["data"] = data
        return data


def _read_mt5() -> dict[str, Any]:
    try:
        import MetaTrader5 as mt5
    except Exception as exc:  # pragma: no cover - platform dependent
        return {
            "available": False,
            "connected": False,
            "error": f"MetaTrader5 未安装或不可用: {exc}",
            "account": None,
            "positions": [],
        }

    try:
        if not mt5.initialize():
            return {
                "available": True,
                "connected": False,
                "error": f"MT5 连接失败: {mt5.last_error()}",
                "account": None,
                "positions": [],
            }

        account = mt5.account_info()
        if account is None:
            return {
                "available": True,
                "connected": False,
                "error": f"MT5 账户信息读取失败: {mt5.last_error()}",
                "account": None,
                "positions": [],
            }

        raw_positions = mt5.positions_get() or []
        positions: list[dict[str, Any]] = []
        for p in raw_positions:
            row = p._asdict() if hasattr(p, "_asdict") else dict(p)
            pos_type = int(row.get("type", 0))
            positions.append({
                "ticket": row.get("ticket"),
                "symbol": row.get("symbol"),
                "type": "BUY" if pos_type == 0 else "SELL",
                "volume": _num(row.get("volume")),
                "price_open": _num(row.get("price_open")),
                "price_current": _num(row.get("price_current")),
                "sl": _num(row.get("sl")),
                "tp": _num(row.get("tp")),
                "profit": _num(row.get("profit")),
                "swap": _num(row.get("swap")),
                "magic": row.get("magic"),
                "comment": row.get("comment"),
                "time": row.get("time"),
                "time_str": _iso_from_ts(row.get("time")),
            })

        floating = sum(
            (p.get("profit") or 0.0) + (p.get("swap") or 0.0) for p in positions
        )
        return {
            "available": True,
            "connected": True,
            "error": None,
            "account": {
                "login": account.login,
                "name": account.name,
                "server": account.server,
                "company": account.company,
                "currency": account.currency,
                "balance": _num(account.balance),
                "equity": _num(account.equity),
                "margin": _num(account.margin),
                "margin_free": _num(account.margin_free),
                "margin_level": _num(account.margin_level),
                "profit": _num(account.profit),
                "leverage": account.leverage,
                "trade_allowed": bool(account.trade_allowed),
            },
            "positions": positions,
            "positions_count": len(positions),
            "floating_pnl": round(floating, 2),
        }
    except Exception as exc:  # pragma: no cover - terminal dependent
        return {
            "available": True,
            "connected": False,
            "error": f"MT5 读取异常: {exc}",
            "account": None,
            "positions": [],
        }


def _resolve_training_symbol(job: dict[str, Any] | None) -> str | None:
    if job and job.get("symbol"):
        return str(job["symbol"])
    data_file = load_settings().get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return str(inspect_parquet_file(data_file).get("symbol") or "")
    except Exception:
        return None


def _training_snapshot() -> dict[str, Any]:
    status = training_manager.status()
    job = status.get("job")
    active = bool(status.get("active"))
    symbol = _resolve_training_symbol(job)

    # 外部训练进程检测（2026-11-17）：
    # 训练可能由 CLI / 打包版 / 计划任务直接启动，这类进程不在 Web 的 job
    # 管理里。只看 training_manager 会误报「训练进程异常退出 (exit_code=...)」，
    # 而实际上因子曲线文件正在被高频改写。用 training_history JSON 的 mtime
    # 兜底判断活跃度（引擎每 5 步原子写入一次，90 秒窗口足够容忍慢步）。
    external = False
    if not active and symbol:
        try:
            _hist = PROJECT_ROOT / f"training_history_{symbol}.json"
            if _hist.exists() and (time.time() - _hist.stat().st_mtime) < 90:
                active = True
                external = True
        except OSError:
            pass

    current_step = 0
    train_steps = 1
    progress_pct = 0.0
    best_score = None
    val_score = None
    has_strategy = False
    has_checkpoint = False
    state = "running" if external else (job.get("state") if job else "idle")

    if symbol:
        try:
            p = get_symbol_progress(symbol)
            current_step = p.current_step
            train_steps = max(1, int(p.train_steps or 1))
            best_score = p.best_score
            has_strategy = bool(p.has_strategy)
            has_checkpoint = bool(p.checkpoint_path)
            history = p.history or {}
            vals = history.get("val_score") or []
            if vals:
                val_score = _num(vals[-1])
            if active and not external:
                live_step = training_manager.parse_step_from_log()
                if live_step is not None:
                    current_step = max(current_step, live_step)
            progress_pct = min(100.0, 100.0 * current_step / train_steps)
            if not active:
                state = p.status
                # SymbolProgress 的 in_progress 表示「有历史进度」，并不代表进程在跑
                if state == "in_progress":
                    state = "paused"
        except Exception as exc:
            state = "error"
            status["error"] = f"训练进度读取失败: {exc}"

    summary = None
    if symbol:
        try:
            summary = get_training_time_summary(symbol, job=job, active=active)
        except Exception:
            summary = None

    return {
        "active": active,
        "state": state or "idle",
        "symbol": symbol,
        "timeframe": job.get("timeframe") if job else None,
        "mode": job.get("mode") if job else None,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": best_score,
        "val_score": val_score,
        "has_strategy": has_strategy,
        "has_checkpoint": has_checkpoint,
        "started_at": job.get("started_at") if job else None,
        "finished_at": job.get("finished_at") if job else None,
        "session_seconds": getattr(summary, "session_seconds", None),
        "history_total_seconds": getattr(summary, "history_total_seconds", None),
        "pid": None if external else (job.get("pid") if job else None),
        "exit_code": None if external else (job.get("exit_code") if job else None),
        "error": None if external else ((job or {}).get("error") or status.get("error")),
        "external": external,
        "log_path": job.get("log_path") if job else None,
    }


def _backtest_snapshot() -> dict[str, Any]:
    status = backtest_manager.status()
    job = status.get("job") or {}
    phase_index = int(status.get("phase_index") or 0)
    phase_total = max(1, int(status.get("phase_total") or 1))
    active = bool(status.get("active"))
    state = job.get("state") or "idle"
    progress_pct = 0.0
    if active:
        progress_pct = min(100.0, 100.0 * (phase_index + 1) / phase_total)
    elif state == "completed":
        progress_pct = 100.0

    return {
        "active": active,
        "state": state,
        "phase": status.get("phase") or "init",
        "phase_label": status.get("phase_label") or "待机",
        "phase_index": phase_index,
        "phase_total": phase_total,
        "progress_pct": round(progress_pct, 1),
        "symbol": job.get("symbol"),
        "strategy_file": job.get("strategy_file"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "log_path": job.get("log_path"),
    }


def _realtime_snapshot() -> dict[str, Any]:
    status = realtime_manager.status()
    watches = status.get("watches") or []
    errors = [
        w for w in watches
        if w.get("state") in ("error", "insufficient") or w.get("warn")
    ]
    return {
        "running": bool(status.get("running")),
        "count": len(watches),
        "error_count": len(errors),
        "watches": watches,
        "server_time": status.get("server_time"),
        "nearest_seconds_to_next": status.get("nearest_seconds_to_next"),
    }


def _alerts(
    training: dict[str, Any],
    backtest: dict[str, Any],
    realtime: dict[str, Any],
    mt5: dict[str, Any],
    data_file_error: str | None,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    def add(level: str, source: str, title: str, detail: str | None = None) -> None:
        alerts.append({
            "level": level,
            "source": source,
            "title": title,
            "detail": detail or "",
        })

    if data_file_error:
        add("error", "数据文件", "训练数据不可用", data_file_error)

    t_state = training.get("state")
    if t_state == "failed":
        add("error", "训练任务", "训练进程异常结束", training.get("error") or "请查看训练日志")
    elif t_state == "stopped":
        add("warn", "训练任务", "训练已被停止", "可重新开始训练或从检查点续训")

    b_state = backtest.get("state")
    if b_state == "failed":
        add("error", "回测任务", "回测进程异常结束", backtest.get("error") or "请查看回测日志")
    elif b_state == "stopped":
        add("warn", "回测任务", "回测已被停止", "可重新运行回测")

    for w in realtime.get("watches") or []:
        label = f"{w.get('symbol')} {w.get('timeframe')}"
        if w.get("state") == "error":
            add("error", "实时监控", f"{label} 计算失败", w.get("message") or "")
        elif w.get("state") == "insufficient":
            add("warn", "实时监控", f"{label} 历史数据不足", w.get("message") or "")
        elif w.get("warn"):
            add("warn", "实时监控", f"{label} 数据源提示", w.get("warn") or "")

    if not mt5.get("connected"):
        add("warn", "MT5", "MT5 未连接", mt5.get("error") or "无法读取账户与持仓，模拟盘持仓区将为空")

    return alerts


def build_monitor_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file_error = None
    data_file = settings.get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if info.get("valid") is False:
                data_file_error = info.get("message") or data_file
        except Exception as exc:
            data_file_error = f"{data_file}: {exc}"

    training = _training_snapshot()
    backtest = _backtest_snapshot()
    realtime = _realtime_snapshot()
    mt5 = _mt5_snapshot()
    alerts = _alerts(training, backtest, realtime, mt5, data_file_error)

    running_tasks = int(bool(training.get("active"))) + int(bool(backtest.get("active")))
    account = mt5.get("account") or {}
    metrics = {
        "running_tasks": running_tasks,
        "training_progress": training.get("progress_pct") or 0.0,
        "realtime_count": realtime.get("count") or 0,
        "alert_count": len(alerts),
        "error_count": sum(1 for a in alerts if a.get("level") == "error"),
        "warning_count": sum(1 for a in alerts if a.get("level") == "warn"),
        "equity": account.get("equity"),
        "floating_pnl": mt5.get("floating_pnl"),
    }

    return {
        "server_time": time.time(),
        "metrics": metrics,
        "training": training,
        "backtest": backtest,
        "realtime": realtime,
        "mt5": mt5,
        "alerts": alerts,
        "data_file": data_file or None,
    }
