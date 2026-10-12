"""camcheck montage: stack old vs new raw frames vertically with labels.

Usage: python camcheck_montage.py OLD_DIR NEW_DIR OUT_PREFIX STEP [STEP ...]
Defaults reproduce the wide20-vs-full-arm comparison at t=10/20/26 s (steps 600/1200/1560).
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

EXPT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXPT, "frames")
W, H, BAR = 1280, 720, 40
DEFAULT_STEPS = (600, 1200, 1560)
STEP_LABELS = {600: "t=10s droplet absorbed, arm approaching", 1200: "t=20s scooping in the bed",
               1560: "t=26s lifted shovel, wet sand sliding"}


def main():
    if len(sys.argv) > 3:
        old_dir, new_dir, prefix = sys.argv[1:4]
        steps = [int(s) for s in sys.argv[4:]] or list(DEFAULT_STEPS)
    else:
        old_dir = os.path.join(EXPT, "frames", "wide20_extract")
        new_dir = os.path.join(EXPT, "frames", "phase9_deepbed_full_raw")
        prefix = "phase9_deepbed_full_camcmp_v2"
        steps = list(DEFAULT_STEPS)
    old_label = "wide20  pos=(0.516,-0.576,0.504) lookat z=0.12 fov=40  -  "
    new_label = "full-arm  pos=(0.7752,-0.8352,0.744) lookat z=0.30 fov=45  -  "
    font = ImageFont.load_default(size=26)
    for s in steps:
        old = Image.open(os.path.join(old_dir, f"frame_{s:04d}.png")).convert("RGB")
        new = Image.open(os.path.join(new_dir, f"frame_{s:04d}.png")).convert("RGB")
        canvas = Image.new("RGB", (W, 2 * H + 2 * BAR), (24, 24, 24))
        canvas.paste(old, (0, BAR))
        canvas.paste(new, (0, H + 2 * BAR))
        d = ImageDraw.Draw(canvas)
        tag = STEP_LABELS.get(s, f"step {s}")
        d.text((16, 8), old_label + tag, font=font, fill=(255, 255, 255))
        d.text((16, H + BAR + 8), new_label + tag, font=font, fill=(255, 255, 255))
        out_path = os.path.join(OUT, f"{prefix}_{s:04d}.png")
        canvas.save(out_path)
        print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
