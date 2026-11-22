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
    size = DIST_EXE.stat().st_size / 1048576
    try:
        shutil.copy2(DIST_EXE, TARGET)
    except PermissionError:
        # 控制台正在运行时 Windows 不允许覆盖 exe。同一卷内可以把运行中的
        # exe 改名（句柄还指着旧文件），腾出位置后新 exe 就能就位 —— 用户
        # 下次双击启动的就是新版，不用先手动关掉。
        prev = TARGET.with_name("AlphaMaster.prev.exe")
        try:
            if prev.exists():
                prev.unlink()
        except OSError:
            pass
        try:
            TARGET.rename(prev)
            shutil.copy2(DIST_EXE, TARGET)
            print(f"完成：{TARGET}  ({size:.1f} MB)")
            print("注意：旧版正在运行，已改名为 AlphaMaster.prev.exe；下次启动即为新版。")
            return 0
        except OSError as exc:
            alt = TARGET.with_name("AlphaMaster.new.exe")
            shutil.copy2(DIST_EXE, alt)
            print(f"控制台正在运行，无法替换（{exc}）")
            print(f"新版已放到 {alt}，关掉控制台后改名成 AlphaMaster.exe 即可。")
            return 0
    print(f"完成：{TARGET}  ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
