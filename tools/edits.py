"""Per-photo develop recipes -> 1080x1920 Instagram story JPEGs.

Usage:  python3 tools/edits.py <uploads_dir> <out_dir> [name ...]
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import grade as g  # noqa: E402


# ------------------------------------------------------------------ recipes
def temple_sun(x):
    """Golden morning over the temple: warm, airy, haze cut in the valley."""
    yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1]]
    flare = (xx - 1165) ** 2 + (yy - 1819) ** 2 < 17 ** 2  # green lens-flare dot
    x = g.heal(x, flare)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.5), radius=8, hi=0.04)
    x = g.apply_mask(x, g.dehaze(x, 0.35), g.linear_mask(x.shape, 0.62, 0.45))
    # warm the land, keep the sky clean blue
    warm = g.white_balance(x, temp=0.25, tint=0.04)
    x = g.apply_mask(x, warm, 1 - g.linear_mask(x.shape, 0.4, 0.55))
    x = g.tone(x, shadows=0.4, highlights=-0.25, blacks=-0.1)
    x = g.contrast(x, 0.14)
    x = g.apply_mask(x, g.clarity(x, 0.22), 1 - g.linear_mask(x.shape, 0.45, 0.55))
    x = g.hsl(x, blue=(-6, 0.25, -0.12), aqua=(10, 0.15, 0), orange=(0, 0.18, 0.02),
              yellow=(-6, 0.05, 0), green=(-14, -0.22, -0.05), red=(0, 0.12, 0))
    x = g.split_tone(x, sh_hue=200, sh_amt=0.18, hl_hue=40, hl_amt=0.18)
    sky = g.protect_highlights(g.linear_mask(x.shape, 0.0, 0.45), x, 0.55, 0.85)
    x = g.apply_mask(x, g.exposure(x, -0.3), sky)
    x = g.bloom(x, threshold=0.9, radius=25, amount=0.25, tint=(1.0, 0.93, 0.8))
    # restore a white-hot sun core
    x = g.apply_mask(x, g.exposure(x, 0.6), g.radial_mask(x.shape, 0.072, 0.176, 0.085, feather=0.8))
    x = g.vignette(x, 0.2)
    return g.grain(x)


def misty_layers(x):
    """Pastel, dreamy morning mist with the tower hill on the third."""
    x = g.crop(x, 378, 0, 1512, 2016)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.5), radius=4)
    x = g.white_balance(x, temp=0.1, tint=0.1)
    x = g.apply_mask(x, g.dehaze(x, 0.3), g.linear_mask(x.shape, 0.75, 0.5))
    x = g.tone(x, shadows=0.25, highlights=-0.3, blacks=-0.15)
    x = g.contrast(x, 0.18)
    x = g.clarity(x, 0.22)
    x = g.hsl(x, blue=(4, -0.12, 0.08), green=(-20, -0.35, -0.04),
              red=(0, 0.25, 0), orange=(0, 0.15, 0))
    x = g.split_tone(x, sh_hue=212, sh_amt=0.22, hl_hue=28, hl_amt=0.22)
    # soft peach glow sitting on the horizon
    glow = g.white_balance(g.exposure(x, 0.15), temp=0.35, tint=0.15)
    x = g.apply_mask(x, glow, g.radial_mask(x.shape, 0.55, 0.52, 0.9, 0.18, feather=0.9) * 0.6)
    x = g.matte(x, lift=0.03, roll=0.975)
    x = g.vignette(x, 0.15)
    return g.grain(x)


def golden_balcony(x):
    """Sun flare through the balcony awning, rich golden hour."""
    x = g.crop(x, 322, 0, 1771, 2576)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.45), radius=4)
    x = g.white_balance(x, temp=0.28, tint=0.03)
    x = g.tone(x, shadows=0.45, highlights=-0.5, blacks=-0.12)
    x = g.contrast(x, 0.16)
    x = g.clarity(x, 0.15)
    x = g.hsl(x, blue=(-8, 0.08, -0.12), orange=(0, 0.2, 0.03), yellow=(-8, 0.1, 0),
              green=(-12, -0.2, 0))
    x = g.split_tone(x, sh_hue=205, sh_amt=0.25, hl_hue=38, hl_amt=0.3)
    x = g.apply_mask(x, g.exposure(x, -0.3), g.linear_mask(x.shape, 0.0, 0.35))
    x = g.bloom(x, threshold=0.82, radius=30, amount=0.32)
    x = g.vignette(x, 0.22)
    return g.grain(x)


def sun_horizon(x):
    """The sun kissing the horizon beside Savitri hill: cinematic teal/orange."""
    x = g.crop(x, 258, 0, 1707, 2576)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.45), radius=4)
    x = g.white_balance(x, temp=0.12)
    x = g.apply_mask(x, g.dehaze(x, 0.18), g.linear_mask(x.shape, 0.6, 0.45))
    x = g.tone(x, shadows=0.35, highlights=-0.35, blacks=-0.2)
    x = g.contrast(x, 0.2)
    x = g.clarity(x, 0.15)
    x = g.hsl(x, blue=(-10, 0.18, -0.15), orange=(0, 0.25, 0.05), yellow=(-6, 0.12, 0))
    x = g.split_tone(x, sh_hue=200, sh_amt=0.3, hl_hue=32, hl_amt=0.3)
    x = g.apply_mask(x, g.exposure(x, -0.3), g.linear_mask(x.shape, 0.0, 0.4))
    x = g.bloom(x, threshold=0.85, radius=28, amount=0.35)
    x = g.vignette(x, 0.25)
    return g.grain(x)


def _remove_streak(x):
    """Replace the awning light-leak band in the sky with a smooth sky fit."""
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    centre = 1030 - 0.36 * yy
    band = (np.abs(xx - centre) < 80) & (yy < 880)
    box = (yy > 250) & (yy < 1000) & (xx > 600) & (xx < 1300)
    ref = box & ~(np.abs(xx - centre) < 120)
    ys, xs = yy[ref] / h, xx[ref] / w
    A = np.stack([np.ones_like(ys), xs, ys, xs * ys, xs ** 2, ys ** 2, ys ** 3, xs ** 2 * ys], 1)
    tgt = band & box
    ty, tx = yy[tgt] / h, xx[tgt] / w
    B = np.stack([np.ones_like(ty), tx, ty, tx * ty, tx ** 2, ty ** 2, ty ** 3, tx ** 2 * ty], 1)
    fit = x.copy()
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, x[..., c][ref], rcond=None)
        fit[..., c][tgt] = B @ coef
    m = g.blur1(tgt.astype(np.float32), 18)
    rng = np.random.default_rng(1)
    fit += rng.normal(0, 0.004, fit.shape).astype(np.float32) * m[..., None]
    return g.apply_mask(x, fit, m)


def afterglow_hill(x):
    """Afterglow over the lake with the lit pilgrim path up the hill."""
    x = _remove_streak(x)
    x = g.crop(x, 39, 361, 1082, 2215)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.45), radius=4)
    x = g.white_balance(x, temp=0.08, tint=0.08)
    x = g.tone(x, shadows=0.35, highlights=-0.25, blacks=-0.15)
    x = g.contrast(x, 0.15)
    x = g.clarity(x, 0.15)
    x = g.hsl(x, blue=(-10, 0.2, -0.1), orange=(0, 0.2, 0.02), yellow=(-10, 0.1, 0))
    x = g.split_tone(x, sh_hue=215, sh_amt=0.25, hl_hue=30, hl_amt=0.22)
    x = g.apply_mask(x, g.exposure(x, -0.25), g.linear_mask(x.shape, 0.0, 0.4))
    x = g.bloom(x, threshold=0.72, radius=10, amount=0.4)
    x = g.vignette(x, 0.25)
    return g.grain(x)


def ghats_dusk(x):
    """Pastel dusk over the ghats: peach sky, lilac water, lifted steps."""
    x = g.crop(x, 0, 0, 1449, 2576)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.4), radius=4)
    x = g.white_balance(x, temp=0.12, tint=0.12)
    x = g.tone(x, shadows=0.55, highlights=-0.25, blacks=-0.15)
    x = g.contrast(x, 0.16)
    x = g.clarity(x, 0.25)
    x = g.vibrance(x, 0.3)
    x = g.hsl(x, orange=(0, 0.25, 0.04), yellow=(-8, 0.15, 0), blue=(12, 0.05, 0),
              red=(0, 0.15, 0))
    x = g.split_tone(x, sh_hue=225, sh_amt=0.28, hl_hue=28, hl_amt=0.3)
    sky = g.white_balance(g.exposure(x, -0.15), temp=0.15, tint=0.12)
    x = g.apply_mask(x, sky, g.linear_mask(x.shape, 0.0, 0.32))
    x = g.bloom(x, threshold=0.8, radius=10, amount=0.3)
    x = g.matte(x, lift=0.02, roll=0.98)
    x = g.vignette(x, 0.2)
    return g.grain(x)


def blue_hour_city(x):
    """Moody blue hour: cool sky, golden city lights, glowing lakefront."""
    x = g.crop(x, 378, 0, 1512, 2016)
    x = g.smooth_flat(x, g.linear_mask(x.shape, 0.0, 0.58), radius=5, hi=0.03)
    x = g.dehaze(x, 0.15)
    x = g.apply_mask(x, g.dehaze(x, 0.3), g.linear_mask(x.shape, 0.7, 0.55))
    x = g.white_balance(x, temp=-0.15)
    # push the grey-green sky into a deep twilight blue
    sky = g.white_balance(x, temp=-0.55, tint=0.25)
    sky = g.split_tone(sky, sh_hue=222, sh_amt=0.5, hl_hue=215, hl_amt=0.35)
    x = g.apply_mask(x, sky, g.linear_mask(x.shape, 0.0, 0.6))
    # soft lilac-peach twilight band above the hills, green removed
    glow = g.white_balance(x, temp=0.15, tint=0.35)
    band = g.linear_mask(x.shape, 0.62, 0.45) * g.linear_mask(x.shape, 0.58, 0.64)
    x = g.apply_mask(x, glow, band * 0.8)
    x = g.tone(x, shadows=0.3, highlights=-0.2, blacks=-0.25)
    x = g.contrast(x, 0.18)
    x = g.apply_mask(x, g.clarity(x, 0.3), g.linear_mask(x.shape, 0.7, 0.55))
    x = g.hsl(x, blue=(-6, 0.25, -0.1), aqua=(15, 0.2, 0), yellow=(-8, 0.25, 0.1),
              orange=(0, 0.2, 0.05), green=(-15, -0.3, -0.15), purple=(0, 0.15, 0))
    x = g.split_tone(x, sh_hue=215, sh_amt=0.3, hl_hue=40, hl_amt=0.2)
    sky = g.white_balance(g.exposure(x, -0.5), temp=-0.15)
    x = g.apply_mask(x, sky, g.linear_mask(x.shape, 0.0, 0.5))
    x = g.bloom(x, threshold=0.7, radius=8, amount=0.45)
    x = g.vignette(x, 0.3)
    return g.grain(x)


def night_arch(x):
    """Lake lights framed by the scalloped arch: deep blue night, neon glow."""
    x = g.crop(x, 300, 0, 1434, 2016)
    x = g.chroma_denoise(x, 3.0, luma_radius=1.2)
    x = g.smooth_flat(x, radius=3, lo=0.003, hi=0.012)
    x = g.exposure(x, 0.5)
    x = g.white_balance(x, temp=-0.2, tint=0.1)
    x = g.tone(x, shadows=0.3, highlights=-0.1, blacks=-0.2)
    x = g.clarity(x, 0.1)
    x = g.hsl(x, blue=(0, 0.2, 0), purple=(0, 0.2, 0), orange=(0, 0.2, 0))
    x = g.split_tone(x, sh_hue=225, sh_amt=0.35, hl_hue=35, hl_amt=0.2)
    x = g.bloom(x, threshold=0.55, radius=10, amount=0.5)
    x = g.vignette(x, 0.3)
    return g.grain(x)


RECIPES = {
    "01_temple_sun": ("a541bc7f-image.jpg", temple_sun),
    "02_misty_layers": ("d50b1214-image.jpg", misty_layers),
    "03_golden_balcony": ("a46bb2b3-image.jpg", golden_balcony),
    "04_sun_horizon": ("6797eaff-image.jpg", sun_horizon),
    "05_afterglow_hill": ("2816b823-image.jpg", afterglow_hill),
    "06_ghats_dusk": ("334dac8a-image.jpg", ghats_dusk),
    "07_blue_hour_city": ("ccad9d4a-image.jpg", blue_hour_city),
    "08_night_arch": ("2ec63c87-image.jpg", night_arch),
}


def run(args):
    name, src_dir, out_dir = args
    src, fn = RECIPES[name]
    out = fn(g.load(os.path.join(src_dir, src)))
    path = os.path.join(out_dir, name + ".jpg")
    g.save_story(out, path)
    return path


if __name__ == "__main__":
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(RECIPES)
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        for p in ex.map(run, [(n, src_dir, out_dir) for n in names]):
            print("wrote", p)
