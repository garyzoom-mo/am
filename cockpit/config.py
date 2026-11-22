"""配置解析 —— 定位 AlphaMaster、factor-governor、Python 解释器与端口。

exe 双击运行时没有「当前目录」概念，所以所有路径都必须从 **exe 自身位置**
推导，不能依赖 cwd。首次运行会在 exe 旁生成 cockpit.json，之后以它为准。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

APP_TITLE = "AlphaMaster 控制台"
DEFAULT_PORT = 18888
CONFIG_NAME = "cockpit.json"


def _exe_dir() -> Path:
    """exe 所在目录；源码运行时是项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _bundle_dir() -> Path:
    """只读资源目录（PyInstaller onefile 时是临时解包目录）。"""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


EXE_DIR = _exe_dir()
BUNDLE_DIR = _bundle_dir()
CONFIG_PATH = EXE_DIR / CONFIG_NAME


def _default_config() -> dict[str, Any]:
    """自动探测默认路径。找不到的项留空，由 UI 提示用户填。"""
    am = EXE_DIR
    # 源码直接运行时 __file__ 在 cockpit/ 下，parents[1] 才是项目根
    if not (am / "train_file.py").exists() and not (am / "AlphaMaster.exe").exists():
        for cand in (EXE_DIR.parent, EXE_DIR.parent.parent):
            if (cand / "train_file.py").exists():
                am = cand
                break

    py = am / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = am / ".venv" / "Scripts" / "pythonw.exe"

    fg = am.parent / "缠论助手" / "factor-governor"
    if not (fg / "src" / "factor_governor").is_dir():
        fg = Path(r"E:\工作\缠论助手\factor-governor")

    return {
        "alphamaster": str(am),
        "factor_governor": str(fg),
        "python": str(py),
        "port": DEFAULT_PORT,
        "data_dirs": [str(am), str(am.parent / "缠论助手")],
        # 训练台（AlphaMaster 自带 Web 控制台）
        "web_port": 18765,
        # 实盘参数，UI 里可改的只有 dry_run
        "live": {
            "xauusd": {
                "enabled": True,
                "symbol": "XAUUSD",
                "timeframe": "M30",
                "broker": "mt5",
                "db": str(fg / "runs" / "live.db"),
                "extra": ["--sl-pct", "1.5", "--daily-loss", "0.03"],
            },
            "btcusdt": {
                "enabled": True,
                "symbol": "BTCUSDT",
                "timeframe": "M15",
                "broker": "binance",
                "db": str(fg / "runs" / "live_btc.db"),
                "extra": ["--max-lot", "5", "--sl-pct", "1.5",
                          "--daily-loss", "0.05", "--max-spread-bp", "5"],
            },
        },
    }


def load_config(create: bool = True) -> dict[str, Any]:
    cfg = _default_config()
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except (OSError, json.JSONDecodeError):
            pass
    elif create:
        try:
            CONFIG_PATH.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    return cfg


def paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        "alphamaster": Path(cfg["alphamaster"]),
        "factor_governor": Path(cfg["factor_governor"]),
        "python": Path(cfg["python"]),
    }


def log_dir(am: Path) -> Path:
    d = am / "logs" / "cockpit"
    d.mkdir(parents=True, exist_ok=True)
    return d
