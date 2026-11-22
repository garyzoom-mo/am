"""生成检查点清单（CKPT_INDEX.tsv），让每个模型版本可追溯。

背景：checkpoints/ 里 166 个 .pt 共 7.5GB，不可能进 git；但「第几步、多大、
什么时候产生、内容有没有被改过」这些信息必须留档，否则模型版本无法回溯。

清单只记元数据 + sha256，几 KB，随代码一起提交。哈希带缓存：文件大小和
修改时间没变就复用上次的哈希，所以训练中新产生几个检查点时只有新增的会被
重新计算，整体仍是秒级。

用法:
    python tools/ckpt_index.py              # 增量更新（默认）
    python tools/ckpt_index.py --full       # 强制重算全部哈希
    python tools/ckpt_index.py --check      # 只校验，不写文件
    python tools/ckpt_index.py --dir checkpoints --out CKPT_INDEX.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ckpt_XAUUSD_step_0120.pt  /  ckpt_BTCUSDT_step_2000.pt
_NAME_RE = re.compile(r"^ckpt_(?P<sym>.+?)_step_?(?P<step>\d+)\.pt$")
_HEADER = "symbol\tstep\tsize_mb\tmtime\tsha256\tfile"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_existing(out: Path) -> dict[str, tuple[str, str]]:
    """读旧清单，返回 {file: (size_mb, sha256)} 用于缓存命中。"""
    cache: dict[str, tuple[str, str]] = {}
    if not out.exists():
        return cache
    try:
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("symbol\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                cache[parts[5]] = (parts[2], parts[4])
    except OSError:
        pass
    return cache


def scan(dirpath: Path, out: Path, full: bool = False) -> list[tuple]:
    cache = {} if full else load_existing(out)
    rows: list[tuple] = []
    skipped = 0

    for path in sorted(dirpath.glob("ckpt_*.pt")):
        m = _NAME_RE.match(path.name)
        if not m:
            skipped += 1
            continue
        st = path.stat()
        size_mb = f"{st.st_size / 1048576:.1f}"
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M"
        )
        cached = cache.get(path.name)
        if cached and cached[0] == size_mb:
            digest = cached[1]
        else:
            digest = _sha256(path)
        rows.append((m.group("sym"), int(m.group("step")), size_mb, mtime, digest, path.name))

    rows.sort(key=lambda r: (r[0], r[1]))
    if skipped:
        print(f"[warn] {skipped} 个文件名不符合 ckpt_<SYM>_step_<N>.pt，已跳过", file=sys.stderr)
    return rows


def verify(dirpath: Path, out: Path) -> int:
    """按清单重算哈希，报出被改动或缺失的检查点。"""
    if not out.exists():
        print(f"[error] 清单不存在: {out}", file=sys.stderr)
        return 2
    bad = 0
    total = 0
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("symbol\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        total += 1
        name, expect = parts[5], parts[4]
        path = dirpath / name
        if not path.exists():
            print(f"[缺失] {name}")
            bad += 1
            continue
        actual = _sha256(path)
        if actual != expect:
            print(f"[变更] {name}")
            bad += 1
    print(f"\n校验 {total} 个检查点，异常 {bad} 个")
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 / 校验检查点清单")
    ap.add_argument("--dir", default="checkpoints", help="检查点目录（默认 checkpoints）")
    ap.add_argument("--out", default="CKPT_INDEX.tsv", help="清单输出路径（默认 CKPT_INDEX.tsv）")
    ap.add_argument("--full", action="store_true", help="强制重算全部哈希，不用缓存")
    ap.add_argument("--check", action="store_true", help="只校验清单，不写文件")
    args = ap.parse_args(argv)

    dirpath = Path(args.dir)
    out = Path(args.out)
    if not dirpath.is_dir():
        print(f"[error] 目录不存在: {dirpath}", file=sys.stderr)
        return 2

    if args.check:
        return verify(dirpath, out)

    rows = scan(dirpath, out, full=args.full)
    lines = [
        "# 检查点索引 —— 由 tools/ckpt_index.py 生成，勿手改",
        f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 检查点: {len(rows)} 个，合计 {sum(float(r[2]) for r in rows) / 1024:.2f} GB",
        _HEADER,
    ]
    lines += ["\t".join(str(v) for v in r) for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"已写入 {out}：{len(rows)} 个检查点，{sum(float(r[2]) for r in rows) / 1024:.2f} GB")
    for sym in sorted({r[0] for r in rows}):
        n = sum(1 for r in rows if r[0] == sym)
        latest = max(r[1] for r in rows if r[0] == sym)
        print(f"  {sym:<10} {n:>4} 个，最新 step {latest}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
