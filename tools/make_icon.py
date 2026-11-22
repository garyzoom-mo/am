"""生成应用图标（一次性工具）。"""
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角深色底 + 蓝色描边
d.rounded_rectangle([6, 6, S - 6, S - 6], radius=52, fill=(13, 17, 23, 255),
                    outline=(31, 111, 235, 255), width=6)

# 三根 K 线（x, 实体上沿, 实体下沿, 颜色）。红涨绿跌，符合 A 股/国内习惯
candles = [
    (62, 108, 152, (248, 81, 73)),
    (112, 126, 176, (63, 185, 80)),
    (162, 80, 128, (248, 81, 73)),
]
for x, top, bottom, color in candles:
    d.line([x, top - 24, x, bottom + 24], fill=color, width=7)
    d.rounded_rectangle([x - 16, top, x + 16, bottom], radius=5, fill=color)

# 向上趋势线
pts = [(40, 200), (86, 176), (132, 186), (178, 142), (218, 104)]
d.line(pts, fill=(88, 166, 255), width=9, joint="curve")
d.ellipse([206, 92, 230, 116], fill=(88, 166, 255))

img.save("assets/app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                  (64, 64), (128, 128), (256, 256)])
img.save("assets/app.png")
print("icon ok")
