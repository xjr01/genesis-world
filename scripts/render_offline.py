"""Offline particle renderer (no genesis/pyrender involvement): rasterize the settled npz
state with numpy+PIL only, as an independent ground-truth view of the fluid arrangement.

Sprites are true discs at the physical particle radius with soft edges and a fixed radial
shading; draw order = back-to-front (top view: sort by z ascending). Concentration palette
matches _concentration_colors (white milk -> tan -> dark-brown coffee).

Usage: python multiflow/scripts/render_offline.py [--npz path] [--out prefix]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scene constants (mirror mf15_pour_overflow.py)
R_IN_MUG, R_OUT_MUG, Z_RIM_MUG = 0.15, 0.166, 0.316
R_IN_JUG, R_OUT_JUG, Z_RIM_JUG = 0.125, 0.141, 0.316
JUG_X = 0.42
PS = 0.008

PALETTE_LO = np.array([1.00, 1.00, 1.00])          # c=0 milk
PALETTE_MID = np.array([0.95, 0.78, 0.30])         # c=0.5 latte
PALETTE_HI = np.array([0.30, 0.14, 0.05])          # c=1 coffee


def conc_color(c):
    c = np.clip(c, 0.0, 1.0)
    lo = PALETTE_LO * (1 - c[:, None] * 2.0) + PALETTE_MID * (c[:, None] * 2.0)
    hi = PALETTE_MID * (2.0 - c[:, None] * 2.0) + PALETTE_HI * (c[:, None] * 2.0 - 1.0)
    return np.where(c[:, None] <= 0.5, lo, hi)


def make_sprite(radius_px, rgb, light_from=(0.35, -0.45)):
    """Soft-edged disc sprite with simple sphere shading, RGBA uint8, size (2r+1)^2."""
    r = radius_px
    xs = np.arange(-r, r + 1)
    X, Y = np.meshgrid(xs, xs)
    D = np.hypot(X, Y) / max(r, 1)
    alpha = np.clip((1.0 - D) / 0.28, 0.0, 1.0)          # soft rim ~28% of radius
    inside = D <= 1.0
    # fake sphere normal for shading
    Z = np.sqrt(np.clip(1.0 - np.minimum(D, 1.0) ** 2, 0.0, 1.0))
    lx, ly = light_from
    lum = 0.55 + 0.45 * np.clip((lx * -X + ly * -Y) / max(r, 1) / np.sqrt(lx * lx + ly * ly) * 0.7 + Z * 0.6, 0.0, 1.0)
    col = np.clip(rgb[None, None, :] * lum[..., None], 0.0, 1.0)
    rgba = np.zeros((2 * r + 1, 2 * r + 1, 4), dtype=np.uint8)
    rgba[..., :3] = (col * 255).astype(np.uint8)
    rgba[..., 3] = (alpha * inside * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def draw_scene(pos, c_val, path, view="top", scale=1400, ss=2):
    """Rasterize top view: table, mug/jug shells (as flat rings), particles back-to-front."""
    if view != "top":
        raise ValueError("v1: top view only")
    # window: x in [-0.45, 0.95], y in [-0.5, 0.5]
    x0, x1, y0, y1 = -0.45, 0.95, -0.50, 0.50
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    # table (light wood-ish gray)
    tbl = Image.new("RGBA", img.size, (168, 160, 150, 255))
    img = Image.alpha_composite(img, tbl)
    d = ImageDraw.Draw(img, "RGBA")

    def P(wx, wy):
        return ((wx - x0) * scale * ss, (y1 - wy) * scale * ss)

    def ring(cx, cy, r_in, r_out, fill, edge):
        d.ellipse([*P(cx - r_out, cy + r_out), *P(cx + r_out, cy - r_out)], fill=fill)
        d.ellipse([*P(cx - r_in, cy + r_in), *P(cx + r_in, cy - r_in)], fill=(60, 52, 46, 255))
        d.ellipse([*P(cx - r_in, cy + r_in), *P(cx + r_in, cy - r_in)], outline=edge, width=int(0.0012 * scale * ss))
    # mug + jug shells (outer wall color; interior darker to suggest depth)
    ring(0.0, 0.0, R_IN_MUG, R_OUT_MUG, (205, 200, 195, 255), (120, 116, 112, 255))
    ring(JUG_X, 0.0, R_IN_JUG, R_OUT_JUG, (205, 200, 195, 255), (120, 116, 112, 255))
    # mug handle: thick arc on -x side (approximate C-handle footprint)
    ha = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dh = ImageDraw.Draw(ha)
    hx0, hx1 = -0.221, -0.166
    dh.arc([*P(hx0, 0.10), *P(hx1, -0.10)], start=90, end=270, fill=(205, 200, 195, 255), width=int(0.026 * scale * ss))
    img = Image.alpha_composite(img, ha)

    # particles: sort by z ascending (draw low first; top-view occlusion)
    order = np.argsort(pos[:, 2])
    pos_s, c_s = pos[order], c_val[order]
    cols = conc_color(c_s)
    rpx = int(round(0.5 * PS * scale * ss))          # physical radius = ps/2
    # sprite cache per quantized color
    cache = {}
    step = max(len(pos_s) // 9000 or 1, 1)
    for k in range(0, len(pos_s)):
        x, y = pos_s[k, 0], pos_s[k, 1]
        key = tuple((cols[k] * 16).astype(int))
        if key not in cache:
            cache[key] = make_sprite(rpx, cols[k])
        img.alpha_composite(cache[key], (int((x - x0) * scale * ss) - rpx, int((y1 - y) * scale * ss) - rpx))
    img = img.resize((W, H), Image.LANCZOS).convert("RGB")
    img.save(path)
    print(f"saved {path} ({W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(WORKSPACE, "videos", "mf15_settled.npz"))
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "videos", "offline"))
    ap.add_argument("--scale", type=int, default=1400)
    args = ap.parse_args()
    d = np.load(args.npz)
    pos = d["pos"][:, 0, :]
    n_c = int(d["n_coffee"])
    c_val = np.concatenate([np.ones(n_c), np.zeros(len(pos) - n_c)])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    draw_scene(pos, c_val, args.out + "_top.png", scale=args.scale)
    # closeup of the mug rim zone (reuse draw with a tighter window)
    global JUG_X_REF
    draw_closeup(pos, c_val, args.out + "_mugcloseup.png", scale=args.scale * 3)


def draw_closeup(pos, c_val, path, scale=4200, ss=2):
    x0, x1, y0, y1 = -0.20, 0.20, -0.20, 0.20
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    img = Image.alpha_composite(img, Image.new("RGBA", img.size, (168, 160, 150, 255)))
    d = ImageDraw.Draw(img, "RGBA")

    def P(wx, wy):
        return ((wx - x0) * scale * ss, (y1 - wy) * scale * ss)

    d.ellipse([*P(-R_OUT_MUG, R_OUT_MUG), *P(R_OUT_MUG, -R_OUT_MUG)], fill=(205, 200, 195, 255))
    d.ellipse([*P(-R_IN_MUG, R_IN_MUG), *P(R_IN_MUG, -R_IN_MUG)], fill=(60, 52, 46, 255))
    order = np.argsort(pos[:, 2])
    cols = conc_color(c_val)
    rpx = int(round(0.5 * PS * scale * ss))
    cache = {}
    for k in order:
        x, y = pos[k, 0], pos[k, 1]
        if np.hypot(x, y) > 0.185:
            continue
        key = tuple((cols[k] * 16).astype(int))
        if key not in cache:
            cache[key] = make_sprite(rpx, cols[k])
        img.alpha_composite(cache[key], (int((x - x0) * scale * ss) - rpx, int((y1 - y) * scale * ss) - rpx))
    img = img.resize((W, H), Image.LANCZOS).convert("RGB")
    img.save(path)
    print(f"saved {path} ({W}x{H})")


if __name__ == "__main__":
    main()


# ---------------- dynamic dumps rendering ----------------

def draw_dump_top(pos, c_val, valid, pose, path, scale=1400, ss=2):
    """Top view with the jug drawn at its dump-time pose (rim circle projected true)."""
    x0, x1, y0, y1 = -0.50, 1.00, -0.55, 0.55
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    img = Image.alpha_composite(img, Image.new("RGBA", img.size, (166, 158, 148, 255)))
    d = ImageDraw.Draw(img, "RGBA")

    def P(wx, wy):
        return ((wx - x0) * scale * ss, (y1 - wy) * scale * ss)

    # table edge line
    d.rectangle([0, 0, W * ss, H * ss], outline=(140, 132, 124, 255), width=2 * ss)

    def circle3d(center, axis, radius, n=64):
        a = np.asarray(axis, float); a /= np.linalg.norm(a)
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        ph = np.linspace(0, 2 * np.pi, n)
        return center + radius * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)

    def draw_cyl(o, ax, r_in, r_out, L, t_b):
        a = np.asarray(ax, float); a /= np.linalg.norm(a)
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
        # outer silhouette = swept circle family projected: approximate with rim + bottom outer circles + tangent lines
        for zoff, rr, fill in ((L, r_out, (208, 203, 198, 255)), (-t_b, r_out, (150, 145, 140, 255))):
            c = o + zoff * a
            pts = circle3d(c, a, rr)
            d.polygon([tuple(P(p[0], p[1])) for p in pts], fill=fill)
        pts_o = circle3d(o + L * a, a, r_in)
        d.polygon([tuple(P(p[0], p[1])) for p in pts_o], fill=(58, 50, 45, 255))

    # mug upright + jug at pose
    draw_cyl(np.array([0, 0, 0.016]), np.array([0, 0, 1.0]), R_IN_MUG, R_OUT_MUG, 0.30, 0.016)
    if pose is not None:
        o, a = pose[:3], pose[3:6]
    else:
        o, a = np.array([JUG_X, 0, 0.016]), np.array([0, 0, 1.0])
    draw_cyl(o, a, R_IN_JUG, R_OUT_JUG, 0.30, 0.016)

    order = np.argsort(pos[:, 2])
    cols = conc_color(c_val)
    rpx = int(round(0.5 * PS * scale * ss))
    cache = {}
    for k in order:
        if valid is not None and not valid[k]:
            continue
        x, y = pos[k, 0], pos[k, 1]
        key = tuple((cols[k] * 16).astype(int))
        if key not in cache:
            cache[key] = make_sprite(rpx, cols[k])
        img.alpha_composite(cache[key], (int((x - x0) * scale * ss) - rpx, int((y1 - y) * scale * ss) - rpx))
    img.resize((W, H), Image.LANCZOS).convert("RGB").save(path)
    print(f"saved {path}")


def draw_dump_oblique(pos, c_val, valid, pose, path, azim=-55, elev=32, scale=1500, ss=2):
    """Orthographic oblique view, particles only (painter's algorithm), with table grid."""
    ca, sa, ce, se = np.cos(np.deg2rad(azim)), np.sin(np.deg2rad(azim)), np.cos(np.deg2rad(elev)), np.sin(np.deg2rad(elev))
    Rm = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1, 0, 0], [0, ce, se], [0, -se, ce]])
    M = Rx @ Rm  # world -> view (x right, y up, z toward viewer)

    def proj(P):
        V = P @ M.T
        return V[:, 0], V[:, 1], V[:, 2]

    # include a fake table plane z=0 and container outlines
    verts = []
    for wx in np.arange(-0.55, 1.15, 0.2):
        line = np.stack([np.full(20, wx), np.linspace(-0.75, 0.75, 20), np.zeros(20)], 1)
        verts.append(line)
    for wy in np.arange(-0.75, 0.8, 0.2):
        line = np.stack([np.linspace(-0.55, 1.15, 20), np.full(20, wy), np.zeros(20)], 1)
        verts.append(line)
    allpts = np.concatenate(verts + [pos if valid is None else pos[valid]])
    X, Y, Zv = proj(allpts)
    x0, x1 = X.min() - 0.05, X.max() + 0.05
    y0, y1 = Y.min() - 0.05, Y.max() + 0.15
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGB", (W * ss, H * ss), (24, 24, 28))
    d = ImageDraw.Draw(img, "RGBA")

    def PXY(x, y):
        return ((x - x0) * scale * ss, (y1 - y) * scale * ss)

    for line in verts:
        Xl, Yl, _ = proj(line)
        d.line([tuple(PXY(a, b)) for a, b in zip(Xl, Yl)], fill=(60, 58, 56, 255), width=ss)
    # particles back-to-front
    pm = pos if valid is None else pos[valid]
    cm = c_val if valid is None else c_val[valid]
    X, Y, Zv = proj(pm)
    order = np.argsort(Zv)  # far (small z toward viewer? define z toward viewer positive) draw far first
    cols = conc_color(cm)
    rpx = int(round(0.5 * PS * scale * ss))
    cache = {}
    for k in order:
        key = tuple((cols[k] * 16).astype(int))
        if key not in cache:
            cache[key] = make_sprite(rpx, cols[k])
        img.paste(cache[key], (int(PXY(X[k], Y[k])[0]) - rpx, int(PXY(X[k], Y[k])[1]) - rpx), cache[key])
    img.resize((W, H), Image.LANCZOS).save(path)
    print(f"saved {path}")


def render_dumps(dumps_path, out_prefix, times=None):
    d = np.load(dumps_path)
    ts = sorted({float(k) for k in d.files if k[0].isdigit()})
    if times:
        ts = [t for t in ts if any(abs(t - q) < 0.01 for q in times)]
    for t in ts:
        key = f"{t:g}"
        pos, c_val = d[key], d[f"c_{key}"]
        valid = d[f"valid_{key}"].astype(bool) if f"valid_{key}" in d.files else None
        pose = d[f"pose_{key}"] if f"pose_{key}" in d.files else None
        draw_dump_top(pos, c_val, valid, pose, f"{out_prefix}_t{key}_top.png")
        draw_dump_oblique(pos, c_val, valid, pose, f"{out_prefix}_t{key}_oblique.png")
