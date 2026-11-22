"""
download_mt5_klines.py — 从本机已登录的 MT5 终端下载指定品种/周期的 K 线到 Parquet。

与 download_full_history.py 的区别：
  - 可指定任意周期 M1/M5/M15/M30/H1/H4/D1/W1/MN1
  - 可指定输出目录（默认跟随 Config.KLINE_CACHE_DIR）
  - 不读 .env 凭证，直接使用当前 MT5 终端已登录的会话

用法示例:
  .venv\\Scripts\\python.exe download_mt5_klines.py --symbol XAUUSD --timeframe M30 ^
      --out "E:\\工作\\缠论助手" --bars 50000

输出: {out}\\{symbol}_{timeframe}.parquet
列: time(int64 Unix秒), open, high, low, close, tick_volume
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed")
    sys.exit(1)

from config import Config  # noqa: E402

_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]
MAX_BARS_PER_REQUEST = 50000


def download(symbol: str, timeframe: str, out_dir: Path, bars: int) -> Path | None:
    tf = Config.get_timeframe(timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{timeframe}.parquet"

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"MT5 不认识品种 {symbol}")
            return None
        if not info.visible and not mt5.symbol_select(symbol, True):
            logger.error(f"无法把 {symbol} 加入 Market Watch: {mt5.last_error()}")
            return None

        logger.info(f"{symbol} {timeframe}: 请求 {bars:,} 根…")
        rates = None
        for attempt in range(5):
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            if rates is not None and len(rates) > 0:
                break
            err = mt5.last_error()
            wait = 35 * (attempt + 1)
            logger.warning(f"  第 {attempt + 1}/5 次失败 {err}，{wait}s 后重试")
            if attempt < 4:
                time.sleep(wait)
        if rates is None or len(rates) == 0:
            logger.error(f"{symbol} {timeframe}: 5 次重试后仍无数据")
            return None

        df = pd.DataFrame(rates)[_COLUMNS].astype({
            "time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "tick_volume": "int64",
        })
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df.to_parquet(out_path, index=False)

        t0 = datetime.fromtimestamp(int(df["time"].iloc[0]), tz=timezone.utc).strftime("%Y-%m-%d")
        t1 = datetime.fromtimestamp(int(df["time"].iloc[-1]), tz=timezone.utc).strftime("%Y-%m-%d")
        logger.success(f"{out_path.name}: {len(df):,} bars  {t0} ~ {t1}  -> {out_path}")
        if len(df) < Config.MIN_BARS:
            logger.warning(f"仅 {len(df)} 根，低于训练最低要求 {Config.MIN_BARS}")
        return out_path
    finally:
        mt5.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="从 MT5 下载指定周期 K 线 -> Parquet")
    parser.add_argument("--symbol", required=True, help="品种，如 XAUUSD")
    parser.add_argument("--timeframe", default="M30",
                        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
    parser.add_argument("--out", default=Config.KLINE_CACHE_DIR, help="输出目录")
    parser.add_argument("--bars", type=int, default=MAX_BARS_PER_REQUEST,
                        help=f"最多下载根数，默认 {MAX_BARS_PER_REQUEST}")
    args = parser.parse_args()

    print("=" * 62)
    print(f"  品种: {args.symbol}")
    print(f"  周期: {args.timeframe}")
    print(f"  根数: 最多 {args.bars:,}")
    print(f"  输出: {args.out}")
    print("=" * 62 + "\n")
    download(args.symbol.upper(), args.timeframe, Path(args.out), args.bars)


if __name__ == "__main__":
    main()
