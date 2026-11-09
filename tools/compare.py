"""Side-by-side before/after sheet: python3 tools/compare.py <uploads> <stories> <out.jpg> [names]"""
import os, sys
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(__file__))
from edits import RECIPES

src_dir, st_dir, out = sys.argv[1:4]
names = sys.argv[4:] or list(RECIPES)
H = 800
tiles = []
for n in names:
    a = ImageOps.exif_transpose(Image.open(os.path.join(src_dir, RECIPES[n][0]))).convert("RGB")
    b = Image.open(os.path.join(st_dir, n + ".jpg"))
    a = a.resize((int(a.width * H / a.height), H)); b = b.resize((int(b.width * H / b.height), H))
    t = Image.new("RGB", (a.width + b.width + 10, H), "white"); t.paste(a, (0, 0)); t.paste(b, (a.width + 10, 0))
    tiles.append(t)
W = max(t.width for t in tiles)
sheet = Image.new("RGB", (W, len(tiles) * (H + 10)), "white")
for i, t in enumerate(tiles): sheet.paste(t, (0, i * (H + 10)))
sheet.save(out, quality=88)
