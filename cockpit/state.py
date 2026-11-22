"""状态聚合：把服务、训练、品种快照、可用数据文件汇成一份给前端。

品种快照直接复用 `factor_governor.am.build_payload()`（纯标准库实现），
避免两套读盘逻辑各说各话。它要读 sqlite，按 TTL 缓存。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

_DATA_RE = re.compile(r"^(?P<sym>[A-Za-z0-9]+)_(?P<tf>M\d+|H\d+|D\d+)\.parquet$", re.I)

_am_cache: dict[str, Any] = {"at": 0.0, "data": None}
_AM_TTL = 5.0


def _import_am(fg_root: Path):
    src = str(fg_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from factor_governor import am  # noqa: PLC0415
        return am
    except Exception:  # noqa: BLE001
        return None


def symbol_payload(fg_root: Path) -> dict[str, Any]:
    now = time.time()
    if _am_cache["data"] is not None and now - _am_cache["at"] < _AM_TTL:
        return _am_cache["data"]
    am = _import_am(fg_root)
    if am is None:
        return {"error": "无法加载 factor_governor.am", "symbols": [], "totals": {}}
    try:
        data = am.build_payload()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "symbols": [], "totals": {}}
    _am_cache["data"] = data
    _am_cache["at"] = now
    return data


def _size_text(n: int) -> str:
    if n >= 1048576:
        return f"{n / 1048576:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


def scan_data_files(dirs: list[str], limit: int = 60) -> list[dict[str, Any]]:
    """找出可用于训练的数据文件。文件名必须是 <品种>_<周期>.parquet。

    0 字节或几 KB 的文件是历史遗留的空壳，标为不可用并排到最后，
    避免误选后训练直接失败。
    """
    seen: dict[str, dict[str, Any]] = {}
    for raw in dirs:
        base = Path(raw)
        if not base.is_dir():
            continue
        for pattern in ("*.parquet", "*/*.parquet"):
            for path in base.glob(pattern):
                if not path.is_file():
                    continue
                m = _DATA_RE.match(path.name)
                if not m:
                    continue
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                seen[key] = {
                    "path": str(path),
                    "name": path.name,
                    "symbol": m.group("sym").upper(),
                    "timeframe": m.group("tf").upper(),
                    "size_bytes": st.st_size,
                    "size_text": _size_text(st.st_size),
                    "usable": st.st_size >= 65536,   # 小于 64KB 不可能是有效行情
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                }
    items = sorted(
        seen.values(),
        key=lambda x: (not x["usable"], x["symbol"], x["timeframe"]),
    )
    return items[:limit]


def build_state(supervisor, cfg: dict[str, Any]) -> dict[str, Any]:
    am_root = Path(cfg["alphamaster"])
    fg_root = Path(cfg["factor_governor"])
    payload = symbol_payload(fg_root)

    services = supervisor.status()
    training = supervisor.training_status()
    # 进度来自 am 读训练历史文件（它认得出「外部进程」跑的训练）
    symbols = payload.get("symbols") or []
    want = (training.get("symbol") or "").upper()

    detail: dict[str, Any] = {}
    for s in symbols:
        tr = s.get("training") or {}
        if not tr.get("running"):
            continue
        item = {
            "symbol": s.get("symbol"),
            "step": tr.get("step") if tr.get("step") is not None else tr.get("ckpt_step"),
            "best": tr.get("best_ever") if tr.get("best_ever") is not None else tr.get("best"),
            "updated_sec": tr.get("history_age_sec"),
        }
        # 优先取「本次训练的品种」，避免多品种时串台
        if want and s.get("symbol", "").upper() == want:
            detail = item
            break
        if not detail:
            detail = item

    # 三种情况要区分清楚：
    #   本控制台起的训练  → managed=True，可以停
    #   别处起的训练      → managed=False / external=True，界面只能提示
    #   没有训练
    train_job: dict[str, Any] = dict(training)
    # 自己刚跑完的训练，历史文件 mtime 还热着，am 会当成「有进程在写」。
    # 这时候如果标成外部训练，界面会把「开始训练」拦下来，用户点了没反应。
    own_recent = (training.get("last_exit_age") is not None
                  and training["last_exit_age"] < 180)
    if detail and not training.get("running") and not own_recent:
        train_job["external"] = True
        train_job["external_reason"] = (
            f"训练历史 {detail.get('updated_sec') or 0:.0f} 秒前仍在更新"
            if detail.get("updated_sec") is not None else "检测到别的进程在写训练历史"
        )
        train_job["symbol"] = train_job.get("symbol") or detail["symbol"]
    if detail and train_job.get("symbol") is None:
        train_job["symbol"] = detail["symbol"]
    if detail.get("step") is not None:
        train_job["step"] = detail["step"]
    if detail.get("best") is not None:
        train_job["best"] = detail["best"]
    if detail.get("updated_sec") is not None:
        train_job["updated_sec"] = detail["updated_sec"]

    # 历史文件最多 10 秒才写一次，刚启动时还没有它 —— 用训练日志里的
    # 「[N/M]」补上，否则界面一直是「? / 2000」，看着像没开始。
    prog: dict[str, Any] = {}
    if training.get("running"):
        try:
            prog = supervisor.train_progress()
        except Exception:  # noqa: BLE001
            prog = {}
        if prog.get("step") is not None:
            cur = train_job.get("step")
            if cur is None or prog["step"] >= int(cur):
                train_job["step"] = prog["step"]
        if prog.get("total"):
            train_job["steps"] = prog["total"]
        # 训练历史里最后一条是上一轮的步数（比续训起点还小），启动瞬间
        # 显示成「3199 / 3230」会让人以为快练完了，这里兜到起点
        base = train_job.get("start_step")
        cur = train_job.get("step")
        if base is not None and (cur is None or int(cur) < int(base)):
            train_job["step"] = int(base)

    train_job.setdefault("managed", bool(training.get("running")))
    train_job.setdefault("external", False)
    # 目标步数：本次启动时算出来的总目标优先
    if not train_job.get("steps"):
        train_job["steps"] = train_job.get("target_steps") or None

    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": {
            "title": "AlphaMaster 控制台",
            "alphamaster": str(am_root),
            "factor_governor": str(fg_root),
            "python": cfg.get("python"),
            "job_object": bool(getattr(supervisor.job, "ok", False)),
        },
        "services": services,
        "train_job": train_job,
        "symbols": symbols,
        "totals": payload.get("totals") or {},
        "timeline": (payload.get("timeline") or [])[:40],
        "data_files": scan_data_files(cfg.get("data_dirs") or [str(am_root)]),
    }
