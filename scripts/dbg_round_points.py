"""dbg: circular alpha mask on GL_POINTS sprites (octagon-outline regression).

Renders ring point clouds through the vendored pyrender pipeline genesis uses for render_particle_as='points'
(jit.forward_pass drawing GL_POINTS shaded by mesh.frag), once with the ROUND_POINTS define stripped from every
program (bare square sprites, the pre-fix behaviour) and once as shipped (each sprite clipped to a disc).

Two clouds, both centred in an orthographic top-down camera:
  ring   - 200 points at radius 0.1 (200 px), point_size 32 px, so the ring is continuous: its outline envelope and
           area are measured over 16 azimuth bins (the metric requested for the octagon check).
  sparse - 12 points on the same circle, spaced ~105 px so every sprite stands alone: each sprite is then measured
           on its own as bbox fill ratio (square 1.0, disc pi/4 = 0.785) and as the spread of its radius envelope
           over 16 azimuths around the sprite centre (square ~6.6 px, disc ~1 px), which is what actually separates
           a square sprite from a round one. A dense union outline cannot: the corner of a sprite sitting on a
           circle of radius R sticks out radially by only h^2 / (2*(R+h)) ~= 0.6 px, so the outline of a dense ring
           is round with or without the fix - the facet the fix removes lives on the individual sprites.

The depth-only pass is rendered as well, since particle depth images and shadow maps go through mesh_depth.frag.

A third scene checks the point-cloud shading: a single-colour ring under one oblique directional light, rendered on
the flat program family point clouds are routed to (their colour is the palette colour) and, for comparison, on the
lit family they used before. Lighting responds to where each particle sits, so the lit path prints azimuthal
brightness bands on the ring while the flat path is uniform; the band strength is the azimuthal Fourier amplitude of
the annulus brightness, reported per angular order including the 8-fold one.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \\
    "$PY" multiflow/scripts/dbg_round_points.py
"""

import os
import sys
from contextlib import contextmanager, nullcontext

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402
from OpenGL.GL import GL_PROGRAM_POINT_SIZE, GL_RENDERER, glDisable, glGetString, glPointSize  # noqa: E402
from PIL import Image  # noqa: E402

import genesis.ext.pyrender as pyrender  # noqa: E402
from genesis.ext.pyrender.constants import GLTF, ProgramFlags, RenderFlags  # noqa: E402
from genesis.ext.pyrender.jit_render import JITRenderer  # noqa: E402
from genesis.ext.pyrender.platforms.pyglet_platform import PygletPlatform  # noqa: E402
from genesis.ext.pyrender.shader_program import ShaderProgramCache  # noqa: E402

IMG = 640  # square image, px
PX_PER_WORLD = 2000.0  # orthographic scale, px per world unit
RADIUS = 0.1  # ring radius, world units -> 200 px
N_RING = 200  # ring spacing ~6.3 px, far under the point size, so the ring stays continuous
N_SPARSE = 12  # ring spacing ~105 px, so every sprite stands alone
POINT_SIZE = 32.0  # sprite size, px
N_BINS = 16
OUT_DIR = os.path.join(WORKSPACE, "scripts", "_out_round_points")


def ring_positions(n_points):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.stack([RADIUS * np.cos(theta), RADIUS * np.sin(theta), np.zeros(n_points)], axis=1)


def build_cloud_scene(positions):
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.ones(3))
    scene.add(
        pyrender.Mesh.from_points(positions, colors=np.ones((len(positions), 4), np.float32)), name="cloud"
    )
    camera = pyrender.OrthographicCamera(xmag=(IMG / 2) / PX_PER_WORLD, ymag=(IMG / 2) / PX_PER_WORLD)
    pose = np.eye(4)
    pose[2, 3] = 1.0  # a camera looks down its own -Z, so +Z placement frames the ring head on
    scene.add(camera, pose=pose)
    return scene


@contextmanager
def round_points_stripped():
    """Drop the ROUND_POINTS define before the program key is built, reproducing the pre-fix square sprites."""
    base_get_program = ShaderProgramCache.get_program

    def get_program_without_round_points(cache, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if defines is not None:
            defines.pop("ROUND_POINTS", None)
        return base_get_program(cache, vertex_shader, fragment_shader, geometry_shader, defines)

    ShaderProgramCache.get_program = get_program_without_round_points
    try:
        yield
    finally:
        ShaderProgramCache.get_program = base_get_program


def render_arms(scene):
    """Render the scene with and without the round-point gate, colour and depth only.

    Each arm draws through its own shader cache, the way the normal channel swaps in the normal-program cache: the
    plain cache is fed programs with ROUND_POINTS stripped, the round cache the shipped ones. Dropping the program id
    map per arm makes the jit re-resolve programs against the active cache; deleting and rebuilding programs inside
    one cache instead leaves draws on recycled program ids.
    """
    caches = {"plain": ShaderProgramCache(), "round": ShaderProgramCache()}
    jit = JITRenderer(scene, [], [])
    renderer = pyrender.Renderer(IMG, IMG, jit, point_size=POINT_SIZE)

    # Same GL state the genesis rasterizer sets before rendering: GL points take their size from the context.
    glDisable(GL_PROGRAM_POINT_SIZE)
    glPointSize(POINT_SIZE)

    images = {}
    for arm in ("plain", "round"):
        renderer._program_cache = caches[arm]
        with round_points_stripped() if arm == "plain" else nullcontext():
            jit.program_id.clear()
            color = renderer.render(scene, RenderFlags.OFFSCREEN)[0]
            depth = renderer.render(scene, RenderFlags.OFFSCREEN | RenderFlags.DEPTH_ONLY | RenderFlags.RET_DEPTH)[0]
        images[arm] = (color, depth)
    return images, renderer


def save_png(name, img):
    path = os.path.join(OUT_DIR, name + ".png")
    Image.fromarray(img).save(path)
    return path


class _LitPointsShim:
    """Presents a points primitive as a triangle primitive, so the lit program family is selected for it."""

    def __init__(self, primitive):
        self.primitive = primitive

    @property
    def mode(self):
        return GLTF.TRIANGLES

    @property
    def double_sided(self):
        return self.primitive.double_sided

    @property
    def buf_flags(self):
        return self.primitive.buf_flags

    @property
    def material(self):
        return self.primitive.material


@contextmanager
def points_forced_lit(renderer, jit):
    """Reproduce the pre-change behaviour: point clouds on the lit program family, fed the lighting uniforms."""
    base_get_program = renderer._get_primitive_program

    def lit_program(primitive, flags, program_flags):
        if primitive.mode == GLTF.POINTS and program_flags & ProgramFlags.USE_MATERIAL:
            primitive = _LitPointsShim(primitive)
        return base_get_program(primitive, flags, program_flags)

    renderer._get_primitive_program = lit_program
    unlit = jit.render_flags[:, 9].copy()
    jit.render_flags[:, 9] = 0
    try:
        yield
    finally:
        renderer._get_primitive_program = base_get_program
        jit.render_flags[:, 9] = unlit


def build_shading_scene():
    """A single-colour ring under one grazing directional light: the lit shade paints azimuthal brightness bands on
    it, the flat shade cannot. With the light 75 degrees off the view axis a particle's light response swings by
    tan(75 deg) * sprite_radius / ring_radius ~= 38 % over the ring, which is the band strength the lit path can
    produce on curved particle surfaces."""
    scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.full(3, 0.02))
    scene.add(pyrender.Mesh.from_points(ring_positions(N_RING), colors=np.ones((N_RING, 4), np.float32)), name="ring")

    camera = pyrender.OrthographicCamera(xmag=(IMG / 2) / PX_PER_WORLD, ymag=(IMG / 2) / PX_PER_WORLD)
    pose = np.eye(4)
    pose[2, 3] = 1.0
    scene.add(camera, pose=pose)

    elevation = np.deg2rad(75.0)  # angle off the view axis; the higher it is, the stronger the bands
    direction = np.array([np.sin(elevation), 0.0, -np.cos(elevation)])
    z_axis = -direction
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    light_pose = np.eye(4)
    light_pose[:3, 0], light_pose[:3, 1], light_pose[:3, 2] = x_axis, y_axis, z_axis
    scene.add(pyrender.DirectionalLight(intensity=2.0, color=np.ones(3)), pose=light_pose)
    return scene


def render_shading_arms(scene):
    """Render the lit scene on the flat (shipped) and lit (pre-change) point-cloud paths."""
    caches = {"flat": ShaderProgramCache(), "lit": ShaderProgramCache()}
    jit = JITRenderer(scene, [], [])
    renderer = pyrender.Renderer(IMG, IMG, jit, point_size=POINT_SIZE)
    glDisable(GL_PROGRAM_POINT_SIZE)
    glPointSize(POINT_SIZE)

    images = {}
    for arm in ("flat", "lit"):
        renderer._program_cache = caches[arm]
        jit.program_id.clear()
        with points_forced_lit(renderer, jit) if arm == "lit" else nullcontext():
            images[arm] = renderer.render(scene, RenderFlags.OFFSCREEN)[0]
    return images


def modulation_stats(img):
    """Azimuthal brightness spectrum of the ring annulus: Fourier amplitude per angular order, relative to the mean
    brightness. Lighting responds to every particle's position, so it modulates the ring; a flat shade cannot."""
    intensity = img.astype(np.float64).mean(axis=2)
    mask = intensity > 10
    ys, xs = np.nonzero(mask)
    center = (xs.mean(), ys.mean())
    radius = np.hypot(xs - center[0], ys - center[1])

    r_in = RADIUS * PX_PER_WORLD - POINT_SIZE / 2.0
    r_out = RADIUS * PX_PER_WORLD + POINT_SIZE / 2.0
    annulus = (radius >= r_in) & (radius <= r_out)
    angle = np.arctan2(ys[annulus] - center[1], xs[annulus] - center[0])
    brightness = intensity[ys[annulus], xs[annulus]]

    n_bins = 360
    index = np.clip(((angle + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    profile = np.bincount(index, weights=brightness, minlength=n_bins) / np.maximum(counts, 1)
    hit = counts > 0
    filled = np.full(n_bins, profile[hit].mean())
    filled[hit] = profile[hit]

    spectrum = np.abs(np.fft.rfft(filled - filled.mean())) / max(hit.sum(), 1) * 2.0
    orders = np.arange(1, 13)
    return {
        "n_pixels": int(annulus.sum()),
        "mean": float(filled.mean()),
        "amplitudes": {int(k): float(spectrum[k] / filled.mean()) for k in orders},
        "a8": float(spectrum[8] / filled.mean()),
        "a_max": float(spectrum[1:13].max() / filled.mean()),
        "peak_order": int(orders[np.argmax(spectrum[1:13])]),
    }


def report_modulation(label, stats):
    print(f"\n=== {label} ===")
    print(f"annulus pixels: {stats['n_pixels']}   mean brightness: {stats['mean']:.1f} / 255")
    print("azimuthal Fourier amplitude by angular order (% of mean brightness):")
    print("  " + "  ".join(f"k{k}={100.0 * a:5.2f}" for k, a in stats["amplitudes"].items()))
    print(f"8-fold (k=8) modulation: {100.0 * stats['a8']:.2f} %   "
          f"max over k=1..12: {100.0 * stats['a_max']:.2f} % at k={stats['peak_order']}")


def outline_stats(img):
    """Radius statistics of a rendered ring: overall spread plus the per-azimuth silhouette envelope."""
    mask = img.astype(np.int32).sum(axis=2) > 30  # object pixels against the black background
    ys, xs = np.nonzero(mask)
    center = (xs.mean(), ys.mean())
    radius = np.hypot(xs - center[0], ys - center[1])
    angle = np.arctan2(ys - center[1], xs - center[0])

    bins = []
    for b in range(N_BINS):
        a0 = 2.0 * np.pi * b / N_BINS - np.pi
        sel = (angle >= a0) & (angle < a0 + 2.0 * np.pi / N_BINS)
        bins.append((np.degrees(a0 + np.pi / N_BINS) % 360.0, radius[sel].min(), radius[sel].max()))

    return {
        "n_pixels": int(mask.sum()),
        "p5": float(np.percentile(radius, 5)),
        "p50": float(np.percentile(radius, 50)),
        "p95": float(np.percentile(radius, 95)),
        "bins": bins,
        "inner_spread": max(b[1] for b in bins) - min(b[1] for b in bins),
        "outer_spread": max(b[2] for b in bins) - min(b[2] for b in bins),
    }


def sprite_stats(img, positions):
    """Shape of every isolated sprite: bbox fill ratio and how far the silhouette reaches past its flat sides.

    A square sprite reaches h*sqrt(2) along its diagonal against h on its sides, a spread of h*(sqrt(2)-1) ~ 6.6 px;
    a disc reaches h in every direction, so the spread collapses to the rasterization residue.
    """
    mask = img.astype(np.int32).sum(axis=2) > 30
    half = int(0.75 * POINT_SIZE)  # window wide enough for the square's diagonal
    mid = (IMG - 1) / 2.0
    fills, spreads = [], []
    for x, y in positions[:, :2]:
        col = int(round(mid + PX_PER_WORLD * x))
        row = int(round(mid + PX_PER_WORLD * y))
        if mask[max(row - half, 0):row + half + 1, col - half:col + half + 1].mean() < 0.2:
            row = int(round(2 * mid - row))  # readback flips the image vertically
        window = mask[row - half:row + half + 1, col - half:col + half + 1]
        ys, xs = np.nonzero(window)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        fills.append(window.sum() / float((x1 - x0 + 1) * (y1 - y0 + 1)))
        spreads.append(np.hypot(xs - xs.mean(), ys - ys.mean()).max() - 0.5 * min(x1 - x0 + 1, y1 - y0 + 1))
    return {"fill_max": max(fills), "fill_mean": float(np.mean(fills)),
            "spread_max": float(np.max(spreads)), "spread_mean": float(np.mean(spreads))}


def depth_object_pixels(depth):
    return int((depth < 5.0).sum())  # the cloud sits at linear depth ~1, the far plane reads orders of magnitude higher


def report(label, stats):
    print(f"\n=== {label} ===")
    print(f"area: {stats['n_pixels']} px   ring radius p5/p50/p95: "
          f"{stats['p5']:.1f} / {stats['p50']:.1f} / {stats['p95']:.1f} px")
    print(f"{'azimuth':>8} {'r_in':>8} {'r_out':>8}  (px)")
    for azimuth, r_in, r_out in stats["bins"]:
        print(f"{azimuth:8.1f} {r_in:8.1f} {r_out:8.1f}")
    print(f"16-bin outline envelope spread: inner {stats['inner_spread']:.2f} px, "
          f"outer {stats['outer_spread']:.2f} px")


def report_sprites(label, stats):
    print(f"sprite bbox fill ratio (square 1.0, disc 0.785): mean {stats['fill_mean']:.3f}, "
          f"max {stats['fill_max']:.3f}")
    print(f"sprite radius-envelope spread over 16 azimuths (square ~6.6 px, disc ~1 px): "
          f"mean {stats['spread_mean']:.2f} px, max {stats['spread_max']:.2f} px")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    platform = PygletPlatform(64, 64)
    platform.init_context()
    platform.make_current()
    print(f"GL device: {glGetString(GL_RENDERER).decode()}")

    results = {}
    for name, positions in (("ring", ring_positions(N_RING)), ("sparse", ring_positions(N_SPARSE))):
        images, renderer = render_arms(build_cloud_scene(positions))
        results[name] = {}
        for arm, (color, depth) in images.items():
            # Analyse the PNG on disk, so the numbers describe the saved artifact rather than the in-memory buffer.
            path = save_png(f"{name}_{arm}_sprites", color)
            png = np.array(Image.open(path))
            results[name][arm] = {
                "png": path,
                "outline": outline_stats(png) if name == "ring" else None,
                "sprites": sprite_stats(png, positions) if name == "sparse" else None,
                "depth_px": depth_object_pixels(depth),
            }
            print(f"{name:6s} {arm:5s}: color area {int((png.astype(np.int32).sum(axis=2) > 30).sum())} px, "
                  f"depth area {results[name][arm]['depth_px']} px -> {path}")
        print(f"point sprite state supported = {renderer._point_sprite_supported}\n")

    plain_sprites = results["sparse"]["plain"]["sprites"]
    round_sprites = results["sparse"]["round"]["sprites"]
    ok_square = plain_sprites["fill_max"] > 0.95 and plain_sprites["spread_max"] > 4.0
    ok_round = round_sprites["fill_max"] < 0.85 and round_sprites["spread_max"] <= 2.0

    for arm in ("plain", "round"):
        report(f"{arm.upper()} arm - dense ring outline", results["ring"][arm]["outline"])
        report_sprites(f"{arm.upper()} arm - isolated sprite shape", results["sparse"][arm]["sprites"])
        print(f"depth-only silhouette area: {results['sparse'][arm]['depth_px']} px on the sparse ring, "
              f"{results['ring'][arm]['depth_px']} px on the dense ring")

    print("=" * 70)
    print(f"pre-fix arm (ROUND_POINTS stripped) shows square sprites   : {'YES' if ok_square else 'NO'} "
          f"(fill {plain_sprites['fill_max']:.3f}, envelope spread {plain_sprites['spread_max']:.2f} px)")
    print(f"post-fix arm (ROUND_POINTS gate) shows round sprites       : {'YES' if ok_round else 'NO'} "
          f"(fill {round_sprites['fill_max']:.3f}, envelope spread {round_sprites['spread_max']:.2f} px)")
    area_ratio = results["sparse"]["round"]["depth_px"] / results["sparse"]["plain"]["depth_px"]
    print(f"depth-only silhouette area round/plain on the sparse ring  : {area_ratio:.3f} (disc/square = 0.785)")

    shading = render_shading_arms(build_shading_scene())
    lit_stats = modulation_stats(shading["lit"])
    flat_stats = modulation_stats(shading["flat"])
    for arm in ("lit", "flat"):
        save_png(f"shading_{arm}", shading[arm])
    report_modulation("SHADING pre-change - point clouds on the lit program family", lit_stats)
    report_modulation("SHADING shipped - point clouds on the flat program family", flat_stats)

    print("=" * 70)
    ok_lit_bands = lit_stats["a_max"] > 0.05
    ok_flat_uniform = flat_stats["a_max"] <= 0.01
    print(f"pre-change lit path paints azimuthal bands (>5%)            : {'YES' if ok_lit_bands else 'NO'} "
          f"(max {100.0 * lit_stats['a_max']:.2f} % at k={lit_stats['peak_order']}, "
          f"8-fold {100.0 * lit_stats['a8']:.2f} %)")
    print(f"shipped flat path is azimuthally uniform (<=1% at every k)  : {'YES' if ok_flat_uniform else 'NO'} "
          f"(max {100.0 * flat_stats['a_max']:.2f} %, 8-fold {100.0 * flat_stats['a8']:.2f} %)")
    if not (ok_square and ok_round and ok_lit_bands and ok_flat_uniform):
        sys.exit(1)


if __name__ == "__main__":
    main()
