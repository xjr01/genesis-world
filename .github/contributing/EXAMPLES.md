# Examples Reference

Location: `examples/teapot/`. This branch contains the PBSTF, PBD, DFSPH, and WCSPH teapot comparison.

## Running Examples

```bash
uv run examples/teapot/pbstf_teapot.py --vis --preview particles
uv run examples/teapot/sph_teapot.py --pressure-solver WCSPH --vis --preview render
uv run examples/teapot/render_teapot_comparison.py --jobs 2
```

## Command-line convention

Every example parses its arguments the same way, so learning one script's CLI teaches all of them.

**Short flags are a closed, reserved set.** Each letter means exactly one thing across the whole tree. A flag
that is not in this table gets no short form.

| Short | Long | Type | Default |
|-------|------|------|---------|
| `-v` | `--vis` | `store_true` | off |
| `-g` | `--gpu` | `store_true` | off (CPU) |
| `-b` | `--num-envs` | `int` | per script |
| `-s` | `--steps` | `int` | per script |
| `-t` | `--seconds` | `float` | per script |
| `-e` | `--exp-name` | `str` | per script |
| `-r` | `--record` | `store_true` | off |
| `-o` | `--output-dir` | `str` | per script |
| `-d` | `--debug` | `store_true` | off |

Rules:

1. Short flag first, then long: `parser.add_argument("-v", "--vis", ...)`.
2. Long names use dashes between words (`--num-envs`); argparse exposes them as the underscored
   `args.num_envs`, so no `dest=` is needed.
3. Single-dash flags are one character.
4. Booleans are `store_true` and default off, which makes `default=False` redundant. Where the negative form
   reads better, a single `store_true` negative flag covers it (`--no-ipc`, `--no-force`).
5. Use `--steps` when the script's natural unit is iterations, `-t/--seconds` when it reasons in simulated
   time (that is, it also exposes `--dt` or compares against `scene.t`).
6. `--dt` is the only spelling for the timestep; it matches `SimOptions(dt=...)`.
7. Every flag carries a `help=`. Keyword order is `action=` / `type=`, then `default=`, then `help=`.
8. The parser is named `parser`, built as the first statements of `main()`, with
   `if __name__ == "__main__": main()` at the bottom.

## Backend

The teapot examples use CUDA for the particle simulation and the native Genesis ray tracer.

Long-running scripts cut their horizon under `"PYTEST_VERSION" in os.environ` so the examples CI stays cheap.

## Output files

Everything an example writes goes under `out/`, resolved against the directory the script is launched from, so
a single ignore rule covers the whole tree and nothing lands next to the sources. Name the file after the
example rather than something generic, since every script shares the one directory. Where a script exposes
`-o/--output-dir`, default it to a subdirectory of `out/`.

## Style

Examples are read as reference code, so they are held to `CODING_GUIDELINES.md` like the rest of the tree.
Beyond it, two things matter most here:

- **Keep a script linear.** A helper function is warranted when it is called more than once, or is passed as
  a callback or thread target. A one-shot linear script reads top to bottom inside `main()`.
- **Model the sanctioned data access.** Reach for the public entity and solver getters first, and
  `qd_to_numpy` / `qd_to_torch` from `genesis.utils.misc` when no getter exists.

## Shared Scene

- `fluid_helper.py` defines the geometry, particle sampling, bounds, camera, and teapot motion.
- `teapot_render.py` constructs the shared scene and selects particle or ray-traced water previews.
- `teapot_checkpoint.py` saves and restores physical frame state for rendering replay through entity APIs.
- The three method entry points own their solver and material parameters.
- `render_teapot_comparison.py` records selected methods and verifies identical initialization before composition.
