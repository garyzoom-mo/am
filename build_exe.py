"""把控制台打包成单文件 exe。

    python build_exe.py            # 打包
    python build_exe.py --clean    # 先清 build/dist

产物 dist/AlphaMaster.exe 会被复制到项目根目录，双击即可启动。
exe 内嵌的是**打包时这个 venv 的 Python**，不包含 torch —— 真正的训练/实盘
仍然由本机 .venv 里的 python 执行，exe 只做进程编排与界面。

需要 PyInstaller：.venv/Scripts/python.exe -m pip install pyinstaller
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_EXE = ROOT / "dist" / "AlphaMaster.exe"
TARGET = ROOT / "AlphaMaster.exe"


def main() -> int:
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",          # 单文件，双击即用
        "--windowed",         # 不弹黑窗口
        "--name", "AlphaMaster",
        "--icon", str(ROOT / "assets" / "app.ico"),
        "--add-data", f"{ROOT / 'cockpit' / 'page.html'};cockpit",
        str(ROOT / "run_cockpit.py"),
    ]
    if "--clean" in sys.argv:
        args.insert(3, "--clean")

    print("打包中…")
    proc = subprocess.run(args, cwd=str(ROOT))
    if proc.returncode != 0:
        print("打包失败")
        return proc.returncode

    if not DIST_EXE.exists():
        print(f"未找到产物 {DIST_EXE}")
        return 1
    shutil.copy2(DIST_EXE, TARGET)
    print(f"完成：{TARGET}  ({TARGET.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
