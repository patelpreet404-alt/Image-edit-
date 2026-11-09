"""Objective checks on finished story JPEGs + 100% crop sheets for eyeballing.
python3 tools/qa.py <out_dir_for_crops> files...
"""
import os, sys
import numpy as np
from PIL import Image

def stats(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    l = a @ np.array([0.2126, 0.7152, 0.0722])
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / (mx + 1e-5)
    # noise: high-pass std in the flattest 15% of the image (sky)
    from PIL import ImageFilter
    L = Image.fromarray(l.astype(np.uint8))
    hp = l - np.asarray(L.filter(ImageFilter.GaussianBlur(2)), np.float32)
    loc = np.asarray(Image.fromarray(np.abs(hp).clip(0,255).astype(np.uint8)).filter(ImageFilter.BoxBlur(16)), np.float32)
    flat = loc <= np.percentile(loc, 15)
    return dict(
        mean=l.mean(), p1=np.percentile(l, 1), p99=np.percentile(l, 99),
        clip_hi=(mx >= 254).mean() * 100, clip_lo=(mx <= 3).mean() * 100,
        sat=sat.mean(), sat95=np.percentile(sat, 95),
        flat_noise=hp[flat].std(),
        size=Image.open(p).size)

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    print(f"{'file':28s} {'mean':>5s} {'p1':>4s} {'p99':>4s} {'clipHi%':>7s} {'clipLo%':>7s} {'sat':>5s} {'sat95':>5s} {'noise':>5s}")
    for p in sys.argv[2:]:
        s = stats(p)
        print(f"{os.path.basename(p):28s} {s['mean']:5.0f} {s['p1']:4.0f} {s['p99']:4.0f} {s['clip_hi']:7.2f} {s['clip_lo']:7.2f} {s['sat']:5.2f} {s['sat95']:5.2f} {s['flat_noise']:5.2f}")
        im = Image.open(p)
        # 3 x 100% crops: upper sky, middle (subject band), lower foreground
        w, h = im.size
        crops = [im.crop((w//2-200, int(h*f)-200, w//2+200, int(h*f)+200)) for f in (0.15, 0.55, 0.85)]
        sheet = Image.new("RGB", (1220, 400), "white")
        for i, c in enumerate(crops): sheet.paste(c, (i*410, 0))
        sheet.save(os.path.join(out, "crop_" + os.path.basename(p)), quality=90)
