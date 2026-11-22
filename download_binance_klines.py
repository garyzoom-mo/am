"""
download_binance_klines.py — 从币安现货公开行情下载 K 线并写成 AlphaMaster 可训练的 Parquet。

币安合约接口在本机常被地域限制（HTTP 451），所以这里只用现货公开行情端点：
    https://data-api.binance.vision/api/v3/klines
该端点不需要 API Key。

输出文件名 / 列格式与 AlphaMaster 一致：
    {OUT_DIR}/{symbol}_{timeframe}.parquet     例如 BTCUSDT_H1.parquet
    列: time(int64, Unix秒), open, high, low, close, tick_volume

用法:
    .venv\\Scripts\\python.exe download_binance_klines.py --symbols BTCUSDT ETHUSDT
    .venv\\Scripts\\python.exe download_binance_klines.py --symbols BTCUSDT --intervals 1h --max-bars 1000
    .venv\\Scripts\\python.exe download_binance_klines.py --symbols BTCUSDT --resume

默认输出目录: 项目根目录下的 data/
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

BASE_URLS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
INTERVAL_TO_TF = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
                  "1h": "H1", "4h": "H4", "1d": "D1", "1w": "W1"}
_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]
_DEFAULT_OUT = ROOT / "data"


def _get(path: str, params: dict, retries: int = 4):
    """带 gzip、代理（跟随环境变量）和重试的公开 API GET。"""
    query = urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for base in BASE_URLS:
        url = f"{base}{path}?{query}"
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "AlphaMaster/1.0",
                        "Accept-Encoding": "gzip",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                        raw = gzip.decompress(raw)
                    return json.loads(raw.decode("utf-8"))
            except Exception as exc:
                last_err = exc
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        logger.debug(f"端点不可用 {base}: {last_err}")
    raise RuntimeError(f"币安行情请求失败: {last_err}")


def server_time_ms() -> int:
    return int(_get("/api/v3/time", {})["serverTime"])


def fetch_klines(symbol: str, interval: str, max_bars: int | None) -> pd.DataFrame:
    """向前分页拉取现货 K 线；返回按时间升序、已去重的 DataFrame。"""
    end_ms = server_time_ms()
    rows: list[list] = []
    page = 0
    while max_bars is None or len(rows) < max_bars:
        limit = 1000 if max_bars is None else min(1000, max_bars - len(rows))
        if limit <= 0:
            break
        batch = _get("/api/v3/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_ms,
        })
        if not batch:
            break
        rows.extend(batch)
        page += 1
        if page % 20 == 0:
            logger.info(f"  {symbol} {interval}: 已拉取 {len(rows):,} 根…")

        first_open = int(batch[0][0])
        if len(batch) < limit:
            break
        new_end = first_open - 1
        if new_end >= end_ms:
            break
        end_ms = new_end
        time.sleep(0.12)  # 轻量限速，别把公开端点打挂

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    now_ms = server_time_ms()
    records = []
    for item in rows:
        open_ms, o, h, l, c, vol, close_ms = (
            int(item[0]), item[1], item[2], item[3], item[4], item[5], int(item[6])
        )
        if close_ms > now_ms:  # 未收盘的 bar 不参与训练/回测
            continue
        records.append({
            "time": open_ms // 1000,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "tick_volume": int(float(vol)),
        })

    df = pd.DataFrame(records, columns=_COLUMNS)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if max_bars is not None and len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)
    return df.astype({
        "time": "int64",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "tick_volume": "int64",
    })


def run(out_dir: Path, symbols: list[str], intervals: list[str],
        max_bars: int | None, resume: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for interval in intervals:
            tf = INTERVAL_TO_TF[interval]
            path = out_dir / f"{symbol}_{tf}.parquet"
            if resume and path.exists():
                logger.info(f"跳过已存在: {path.name}")
                continue
            logger.info(f"下载 {symbol} {interval} -> {path.name}")
            try:
                df = fetch_klines(symbol, interval, max_bars)
                if df.empty:
                    logger.warning(f"  {path.name}: 无数据")
                    continue
                df.to_parquet(path, index=False)
                t0 = datetime.fromtimestamp(int(df['time'].iloc[0]), tz=timezone.utc).strftime("%Y-%m-%d")
                t1 = datetime.fromtimestamp(int(df['time'].iloc[-1]), tz=timezone.utc).strftime("%Y-%m-%d")
                logger.success(f"  {path.name}: {len(df):,} bars  {t0} ~ {t1}")
                if len(df) < Config.MIN_BARS:
                    logger.warning(f"  {path.name}: 仅 {len(df)} 根，低于训练最低要求 {Config.MIN_BARS}")
            except Exception as exc:
                logger.error(f"  {path.name}: 失败 — {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载币安现货 K 线 -> AlphaMaster Parquet")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="品种列表，默认 BTCUSDT（如 BTCUSDT ETHUSDT SOLUSDT）")
    parser.add_argument("--intervals", nargs="+", default=["1h"],
                        help="周期列表，默认 1h；可选 1m/5m/15m/30m/1h/4h/1d/1w")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help=f"输出目录，默认 {_DEFAULT_OUT}")
    parser.add_argument("--max-bars", type=int, default=None, help="每个文件最多保留多少根（默认全量）")
    parser.add_argument("--resume", action="store_true", help="跳过已存在的 Parquet")
    args = parser.parse_args()

    bad = [i for i in args.intervals if i not in INTERVAL_TO_TF]
    if bad:
        parser.error(f"不支持的周期: {bad}；可选 {list(INTERVAL_TO_TF)}")

    print("=" * 62)
    print("  币安现货 K 线下载（无需 API Key）")
    print(f"  品种: {' '.join(args.symbols)}")
    print(f"  周期: {' '.join(args.intervals)}")
    print(f"  输出: {args.out}")
    print(f"  根数: {'全量' if args.max_bars is None else args.max_bars}")
    print("=" * 62 + "\n")

    run(Path(args.out), [s.upper() for s in args.symbols], args.intervals,
        args.max_bars, args.resume)


if __name__ == "__main__":
    main()
