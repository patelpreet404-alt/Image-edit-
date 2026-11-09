# Pushkar & Ajmer: Instagram story edits

Eight photos picked from the shoot and edited as 1080×1920 (9:16) Instagram stories.
They are ordered as a sunrise-to-night story.

| # | File | Look |
|---|------|------|
| 1 | `stories/01_temple_sun.jpg` | Golden morning over the temple. Valley haze cut, land warmed, clean blue sky, green lens-flare dot removed. |
| 2 | `stories/02_misty_layers.jpg` | Pastel, dreamy mist. TV-tower hill placed on the right third, peach glow on the horizon, faded film blacks. |
| 3 | `stories/03_golden_balcony.jpg` | Sun flare through the balcony awning. Rich golden hour with teal shadows. |
| 4 | `stories/04_sun_horizon.jpg` | Sun on the horizon beside Savitri hill. Cinematic teal and orange, deep sky gradient. |
| 5 | `stories/05_afterglow_hill.jpg` | Afterglow with the lit pilgrim path. Awning cropped out, light streak removed from the sky, glowing lights. |
| 6 | `stories/06_ghats_dusk.jpg` | Pastel dusk over the ghats. Peach sky, lilac water, steps lifted out of shadow. |
| 7 | `stories/07_blue_hour_city.jpg` | Ajmer at blue hour. Slate-blue sky in place of the grey-green, golden city lights, glowing lakefront. |
| 8 | `stories/08_night_arch.jpg` | Lake lights framed by the scalloped arch. Night noise cleaned, deep blue tones, neon glow on the reflections. |

`previews/stories_grid.jpg` shows the whole set. `previews/before_after.jpg` shows each original next to its edit.

Not used: near-duplicates (the second temple frame and two of the sunset frames) and the dark Hotel Sarovar shot.

## Re-running the edits
```
python3 tools/edits.py <folder-with-originals> stories
```
`tools/grade.py` is a small numpy/Pillow develop toolkit: exposure, white balance, tone sliders, curves,
dehaze, clarity, HSL, split-toning, graduated and radial masks, bloom, grain, retouching and story crops.
Each photo's settings are in `tools/edits.py`.
