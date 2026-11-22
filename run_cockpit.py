"""PyInstaller 打包入口。

不直接打包 cockpit/app.py —— 它是包内模块，用相对导入；从脚本方式启动会
报 ImportError。这里从包外 import，PyInstaller 会顺着依赖收进 cockpit 整个包。
"""
from __future__ import annotations

import sys

from cockpit.app import main

if __name__ == "__main__":
    sys.exit(main())
