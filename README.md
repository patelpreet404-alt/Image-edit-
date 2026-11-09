# Pushkar & Ajmer: Instagram story edits

Eight photos picked from the shoot and edited as 1080×1920 (9:16) Instagram stories.
They are ordered as a sunrise-to-night story.

| # | File | Look |
|---|------|------|
| 1 | `stories/01_temple_sun.jpg` | Golden morning over the temple. Valley haze cut, land warmed, clean blue sky, green lens-flare dot removed. |
| 2 | `stories/02_misty_layers.jpg` | Pastel, dreamy mist. TV-tower hill on the right third, sky fading from cool blue at the top to warm peach at the horizon. |
| 3 | `stories/03_golden_balcony.jpg` | Sun flare through the balcony awning. Rich golden hour with teal shadows. |
| 4 | `stories/04_sun_horizon.jpg` | Sun on the horizon beside Savitri hill. Cinematic teal and orange, deep sky gradient. |
| 5 | `stories/05_afterglow_hill.jpg` | Afterglow with the lit pilgrim path. Awning cropped out, light streak removed from the sky, glowing lights. |
| 6 | `stories/06_ghats_dusk.jpg` | Pastel dusk over the ghats. Peach sky, lilac water, steps lifted out of shadow. |
| 7 | `stories/07_blue_hour_city.jpg` | Ajmer at blue hour. Deep azure sky fading to a rose glow over the hills, land slightly darkened so the golden city lights pop. |
| 8 | `stories/08_night_arch.jpg` | Lake lights framed by the scalloped arch. Tighter crop, night noise cleaned, navy tones with ink-black (not blue) shadows, neon reflections. |

`previews/stories_grid.jpg` shows the whole set. `previews/before_after.jpg` shows each original next to its edit.

Not used: near-duplicates (the second temple frame and two of the sunset frames) and the dark Hotel Sarovar shot.

## Review process
Each edit went through several rounds of checks:
- **Full-frame review:** each photo against its original, and the whole set side by side.
- **100% crops:** sky, subject and foreground, checked for noise, banding, halos and retouch marks.
- **Measurements** (`python3 tools/qa.py <crop_dir> stories/*.jpg`): clipped highlights and shadows,
  white point, saturation, and noise in flat areas.

The references used were common Lightroom practice for sunsets, golden hour and blue hour:
- Pull highlights down and lift shadows, then add a gentle S-curve.
- Shift yellows toward orange and darken the blue sky.
- At blue hour, use a deep azure sky and slightly underexpose so the city lights glow.
- Keep grain fine and keep one consistent look across the set.

Fixes these checks led to:
- Grain reduced and sharpening softened.
- Skies smoothed with an edge-aware filter, which removed noise and JPEG blocking.
- The sun core restored in photo 1.
- A white point added so highlights are crisp, capped at about 0.4% clipping.
- Photo 7's khaki sky regraded to azure and rose.
- Photo 8's blue-crushed blacks neutralised, the arch made visible, and the crop tightened.

## Re-running the edits
```
python3 tools/edits.py <folder-with-originals> stories
```
`tools/grade.py` is a small numpy/Pillow develop toolkit: exposure, white balance, tone sliders, curves,
dehaze, clarity, HSL, split-toning, graduated and radial masks, bloom, grain, retouching and story crops.
Each photo's settings are in `tools/edits.py`.
