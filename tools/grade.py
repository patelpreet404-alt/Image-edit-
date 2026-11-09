"""Small photo-grading toolkit (numpy + Pillow only).

Every function takes and returns a float32 RGB array in [0, 1] (sRGB encoded)
so the steps can be chained like a Lightroom develop stack.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

STORY_SIZE = (1080, 1920)  # Instagram story, 9:16


# --------------------------------------------------------------------------- io
def load(path: str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def to_img(x: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def save_story(x: np.ndarray, path: str, sharpen: float = 45) -> None:
    im = to_img(x)
    if im.size != STORY_SIZE:
        im = im.resize(STORY_SIZE, Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen), threshold=4))
    im.save(path, "JPEG", quality=95, subsampling=0, optimize=True)


# ---------------------------------------------------------------------- helpers
def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def lum(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def blur1(a: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a single channel float array."""
    # three box passes ~= gaussian (box width from sigma)
    if radius <= 0:
        return a.astype(np.float32)
    w = max(1, int(round(np.sqrt(12 * radius * radius / 3 + 1))))
    r = w // 2
    out = a.astype(np.float32)
    for axis in (0, 1):
        for _ in range(3):
            out = _box(out, r, axis)
    return out


def _box(a, r, axis):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur3(x: np.ndarray, radius: float) -> np.ndarray:
    return np.stack([blur1(x[..., c], radius) for c in range(3)], -1)


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _set_lum(x, new_l, old_l=None):
    old_l = lum(x) if old_l is None else old_l
    ratio = (new_l + 1e-4) / (old_l + 1e-4)
    out = x * ratio[..., None]
    # soft-compress anything pushed past white instead of hard clipping
    m = out.max(-1, keepdims=True)
    over = m > 1
    out = np.where(over, out / np.maximum(m, 1e-6) * (1 - np.exp(-m)) / (1 - np.exp(-1.0)) , out)
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ basic panel
def exposure(x, ev):
    return np.clip(_srgb(_lin(x) * (2.0 ** ev)), 0, 1)


def white_balance(x, temp=0.0, tint=0.0):
    """temp > 0 warmer, tint > 0 more magenta. Roughly -1..1 ranges."""
    l = _lin(x)
    gains = np.array([1 + 0.12 * temp, 1 - 0.06 * tint, 1 - 0.12 * temp], np.float32)
    before = lum(l)
    l = l * gains
    l *= ((before + 1e-5) / (lum(l) + 1e-5))[..., None]
    return np.clip(_srgb(l), 0, 1)


def tone(x, shadows=0.0, highlights=0.0, whites=0.0, blacks=0.0, local=12.0):
    """Lightroom-style tone sliders (-1..1). `local` mixes in a blurred guide
    so shadows/highlights recover regions rather than single pixels."""
    l = lum(x)
    g = 0.6 * l + 0.4 * blur1(l, local) if local else l
    nl = l.copy()
    if shadows:
        m = 1 - smoothstep(0.0, 0.55, g)
        nl = nl + shadows * 0.45 * m * (1 - nl) * np.sqrt(np.clip(nl, 0, 1) + 0.02)
    if highlights:
        m = smoothstep(0.45, 1.0, g)
        nl = nl + highlights * 0.45 * m * nl * (1 - 0.3 * nl)
    if whites:
        m = smoothstep(0.7, 1.0, l)
        nl = nl + whites * 0.18 * m
    if blacks:
        m = 1 - smoothstep(0.0, 0.25, l)
        nl = nl + blacks * 0.10 * m
    return _set_lum(x, np.clip(nl, 0, 1), l)


def contrast(x, amount=0.0, pivot=0.45):
    """S-curve on luminance around `pivot`."""
    if not amount:
        return x
    l = lum(x)
    k = 1 + amount * 4
    s = 1 / (1 + np.exp(-k * (l - pivot) * 4))
    s0 = 1 / (1 + np.exp(-k * (0 - pivot) * 4))
    s1 = 1 / (1 + np.exp(-k * (1 - pivot) * 4))
    sl = (s - s0) / (s1 - s0)
    nl = l + (sl - l) * min(abs(amount) * 1.5, 1) * np.sign(amount)
    return _set_lum(x, np.clip(nl, 0, 1), l)


def curve(x, pts, channel=None):
    """Tone curve from control points [(in, out), ...] in 0..1.
    channel=None applies to all channels (RGB curve), else 0/1/2."""
    pts = sorted(pts)
    xs = np.linspace(0, 1, 1024)
    lut = np.interp(xs, [p[0] for p in pts], [p[1] for p in pts])
    # smooth the polyline into a gentle curve
    k = np.hanning(61); k /= k.sum()
    pad = np.pad(lut, 30, mode="edge")
    lut = np.convolve(pad, k, mode="valid")
    out = x.copy()
    chans = range(3) if channel is None else [channel]
    for c in chans:
        out[..., c] = np.interp(x[..., c], xs, lut)
    return np.clip(out, 0, 1)


def matte(x, lift=0.04, roll=0.97):
    """Faded film blacks + soft highlight roll-off."""
    return curve(x, [(0, lift), (0.25, 0.25 + lift * 0.4), (0.75, 0.75), (1, roll)])


def clarity(x, amount=0.0, radius=40):
    if not amount:
        return x
    l = lum(x)
    detail = l - blur1(l, radius)
    m = np.clip(4 * l * (1 - l), 0, 1) ** 0.7
    return _set_lum(x, np.clip(l + amount * detail * m * 1.6, 0, 1), l)


def texture(x, amount=0.0, radius=3):
    if not amount:
        return x
    l = lum(x)
    detail = l - blur1(l, radius)
    return _set_lum(x, np.clip(l + amount * detail * 1.5, 0, 1), l)


def dehaze(x, amount=0.3, patch=25, keep=0.92):
    """Dark-channel-prior dehaze with a soft transmission map."""
    if not amount:
        return x
    dark = x.min(-1)
    dark = np.asarray(
        Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)),
        np.float32) / 255
    dark = blur1(dark, patch)
    flat = dark.ravel()
    idx = np.argpartition(flat, -max(1, flat.size // 1000))[-max(1, flat.size // 1000):]
    A = np.clip(x.reshape(-1, 3)[idx].mean(0), 0.5, 1.0)
    t = 1 - keep * amount * dark / A.max()
    t = np.clip(t, 0.25, 1)[..., None]
    j = (x - A) / t + A
    return np.clip(j, 0, 1)


def saturation(x, amount=0.0):
    l = lum(x)[..., None]
    return np.clip(l + (x - l) * (1 + amount), 0, 1)


def vibrance(x, amount=0.0):
    l = lum(x)[..., None]
    mx, mn = x.max(-1, keepdims=True), x.min(-1, keepdims=True)
    sat = (mx - mn) / (mx + 1e-5)
    s = 1 + amount * (1 - sat) ** 1.5
    return np.clip(l + (x - l) * s, 0, 1)


# ---------------------------------------------------------------------- HSL mix
def _rgb2hsv(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rc = np.where(m, (mx - r) / (d + 1e-9), 0)
    gc = np.where(m, (mx - g) / (d + 1e-9), 0)
    bc = np.where(m, (mx - b) / (d + 1e-9), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc))
    h = np.where(m, (h / 6) % 1, 0)
    s = np.where(mx > 1e-6, d / (mx + 1e-9), 0)
    return h, s, mx


def _hsv2rgb(h, s, v):
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


HUES = {"red": 0, "orange": 30, "yellow": 55, "green": 110, "aqua": 180,
        "blue": 220, "purple": 275, "magenta": 320}


def hsl(x, **bands):
    """hsl(x, blue=(hue_shift_deg, sat_mult_delta, lum_delta), orange=(...))"""
    h, s, v = _rgb2hsv(x)
    hd = h * 360
    dh = np.zeros_like(h); ds = np.zeros_like(h); dv = np.zeros_like(h)
    for name, (shift, sat, lig) in bands.items():
        c = HUES[name]
        dist = np.abs((hd - c + 180) % 360 - 180)
        w = np.clip(1 - dist / 40, 0, 1) ** 2 * smoothstep(0.03, 0.2, s)
        dh += w * shift; ds += w * sat; dv += w * lig
    h2 = ((hd + dh) % 360) / 360
    s2 = np.clip(s * (1 + ds), 0, 1)
    v2 = np.clip(v * (1 + dv * 0.5), 0, 1)
    return np.clip(_hsv2rgb(h2, s2, v2), 0, 1)


# ----------------------------------------------------------------- color grade
def _hue_rgb(deg):
    return _hsv2rgb(np.array(deg / 360.0), np.array(1.0), np.array(1.0))


def split_tone(x, sh_hue=200, sh_amt=0.0, hl_hue=40, hl_amt=0.0, mid_hue=None,
               mid_amt=0.0, balance=0.0):
    l = lum(x)
    pivot = 0.5 + balance * 0.25
    ms = (1 - smoothstep(0.0, pivot + 0.15, l))[..., None]
    mh = smoothstep(pivot - 0.15, 1.0, l)[..., None]
    out = x.copy()
    for hue, amt, m in ((sh_hue, sh_amt, ms), (hl_hue, hl_amt, mh)):
        if amt:
            c = _hue_rgb(hue); c = c - lum(c[None])[0]
            out = out + amt * 0.25 * m * c
    if mid_hue is not None and mid_amt:
        mm = (np.clip(4 * l * (1 - l), 0, 1))[..., None]
        c = _hue_rgb(mid_hue); c = c - lum(c[None])[0]
        out = out + mid_amt * 0.25 * mm * c
    return np.clip(out, 0, 1)


# -------------------------------------------------------------- local masks
def linear_mask(shape, y0, y1, x0=None, x1=None):
    """1 at y0 fading to 0 at y1 (fractions of height)."""
    h, w = shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    m = 1 - smoothstep(min(y0, y1), max(y0, y1), yy) if y0 < y1 else smoothstep(y1, y0, yy)
    m = np.broadcast_to(m, (h, w)).copy()
    return m


def radial_mask(shape, cx, cy, rx, ry=None, feather=1.0):
    h, w = shape[:2]
    ry = rx * w / h if ry is None else ry
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w - cx) / rx) ** 2 + ((yy / h - cy) / ry) ** 2)
    return 1 - smoothstep(1 - feather, 1, d)


def apply_mask(x, edited, mask):
    m = mask[..., None]
    return np.clip(x * (1 - m) + edited * m, 0, 1)


def vignette(x, amount=0.25, roundness=0.9, feather=0.75):
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
    ny *= roundness * h / w if roundness else 1
    d = np.sqrt(nx ** 2 + (ny * w / h) ** 2) / np.sqrt(2)
    m = smoothstep(1 - feather, 1.15, d)
    return np.clip(_srgb(_lin(x) * (1 - amount * m)[..., None]), 0, 1)


def bloom(x, threshold=0.75, radius=25, amount=0.35, tint=(1.0, 0.85, 0.6)):
    l = lum(x)
    m = smoothstep(threshold, 1.0, l)[..., None]
    hi = x * m * np.array(tint, np.float32)
    glow = blur3(hi, radius) * 0.6 + blur3(hi, radius * 3) * 0.4
    return np.clip(1 - (1 - x) * (1 - amount * glow), 0, 1)


def grain(x, amount=0.007, size=1.4, seed=7):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(x.shape[:2]).astype(np.float32)
    if size > 1:
        n = blur1(n, size - 1) * size
    l = lum(x)
    m = np.clip(4 * l * (1 - l), 0.15, 1)
    return np.clip(x + (amount * n * m)[..., None], 0, 1)


# ------------------------------------------------------------------ retouching
def heal(x, mask, iters=400):
    """Diffusion in-paint inside `mask` (bool). Good for flares, dust, stickers
    on smooth areas. Works on the mask bbox only, so it is cheap."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return x
    pad = 12
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, x.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, x.shape[1])
    sub = x[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1]
    known = ~m
    # start from the mean of the border so convergence is quick
    sub[m] = sub[known].mean(0)
    for _ in range(iters):
        avg = (np.roll(sub, 1, 0) + np.roll(sub, -1, 0) + np.roll(sub, 1, 1) + np.roll(sub, -1, 1)) / 4
        sub[m] = avg[m]
    # re-add a little matched noise so the patch doesn't look plastic
    rng = np.random.default_rng(3)
    ref = x[y0:y1, x0:x1][known]
    sigma = float(np.std(ref - ref.mean(0))) * 0.08
    sub[m] += rng.normal(0, sigma, sub[m].shape).astype(np.float32)
    out = x.copy()
    soft = blur1(m.astype(np.float32), 2.0)[..., None]
    out[y0:y1, x0:x1] = x[y0:y1, x0:x1] * (1 - soft) + sub * soft
    return np.clip(out, 0, 1)


def crop(x, x0, y0, x1, y1):
    return x[int(y0):int(y1), int(x0):int(x1)]


def crop_story(x, cx=0.5, cy=0.5, zoom=1.0):
    """Largest 9:16 crop centred on (cx, cy), optionally zoomed in."""
    h, w = x.shape[:2]
    tw, th = (w, w * 16 / 9) if w * 16 / 9 <= h else (h * 9 / 16, h)
    tw, th = tw / zoom, th / zoom
    x0 = np.clip(cx * w - tw / 2, 0, w - tw)
    y0 = np.clip(cy * h - th / 2, 0, h - th)
    return crop(x, x0, y0, x0 + tw, y0 + th)


def chroma_denoise(x, radius=3.0, luma_radius=0.0):
    """Blur colour noise while keeping luminance detail (night shots)."""
    l = lum(x)
    c = x - l[..., None]
    c = blur3(c, radius)
    if luma_radius:
        l = 0.6 * l + 0.4 * blur1(l, luma_radius)
    return np.clip(c + l[..., None], 0, 1)


def smooth_flat(x, mask=None, radius=3.0, lo=0.004, hi=0.02):
    """Edge-aware smoothing of flat areas (skies): removes sensor noise and
    JPEG blocking without touching edges or textured detail."""
    l = lum(x)
    detail = blur1(np.abs(l - blur1(l, 2.0)), 4.0)
    w = 1 - smoothstep(lo, hi, detail)
    if mask is not None:
        w = w * mask
    return apply_mask(x, blur3(x, radius), w)


def protect_highlights(mask, x, start=0.75, end=0.95):
    """Fade a local-adjustment mask out over bright areas (the sun)."""
    return mask * (1 - smoothstep(start, end, lum(x)))


def white_point(x, target=0.95, pct=99.7, max_gain=1.2):
    """Levels: map the `pct` luminance percentile to `target` (never darkens)."""
    l = lum(x)
    p = float(np.percentile(l, pct))
    # never push more than ~0.4% of pixels (any channel) into clipping
    mc = float(np.percentile(x.max(-1), 99.6))
    gain = float(np.clip(min(target / max(p, 1e-3), 0.995 / max(mc, 1e-3)), 1.0, max_gain))
    if gain <= 1.001:
        return x
    return _set_lum(x, np.clip(l * gain, 0, 1.2), l)


def sky_gradient(x, mask, stops, strength=0.85):
    """Re-colour a sky with a designed vertical gradient while keeping its own
    luminance (like a gradient map in a sky mask). stops: [(y_frac, hue_deg, sat)]."""
    h, w = x.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)
    ys = [s[0] for s in stops]
    # interpolate colour in RGB between stop colours
    cols = np.stack([_hsv2rgb(np.array(hd / 360.0), np.array(s), np.array(1.0)) for _, hd, s in stops])
    row = np.stack([np.interp(yy, ys, cols[:, c]) for c in range(3)], -1)  # (h,3)
    row = row / (lum(row[None])[0][:, None] + 1e-5)
    l = lum(x)
    tgt = np.clip(row[:, None, :] * l[..., None], 0, 1)
    return apply_mask(x, tgt, mask * strength)


def neutral_blacks(x, below=0.06, amount=0.5):
    """Pull colour out of the deepest shadows so blacks read as ink, not paint."""
    l = lum(x)
    m = (1 - smoothstep(0.0, below, l)) * amount
    return apply_mask(x, np.repeat(l[..., None], 3, -1), m)
