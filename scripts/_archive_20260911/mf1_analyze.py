"""MF-1 validation analysis: check the 4 acceptance criteria against the mf1_*.csv files."""

import os
import sys

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    path = os.path.join(WORKSPACE, "videos", name)
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def main():
    epsd0 = load("mf1_epsd0.csv")
    baseline = load("mf1_baseline.csv")
    epsd02 = load("mf1_epsd0.2.csv")

    print("=== 1. zero-regression: multiflow epsd=0 vs main-repo baseline, 5 s, KE per-frame ===")
    n = min(len(epsd0), len(baseline))
    ke_a, ke_b = epsd0["ke"][:n], baseline["ke"][:n]
    denom = np.maximum(np.abs(ke_b), 1e-12)
    rel = np.abs(ke_a - ke_b) / denom
    print(f"frames compared: {n}, max |dKE|/|KE| = {rel.max():.3e}  (threshold 1e-4) -> {'PASS' if rel.max() < 1e-4 else 'FAIL'}")
    print(f"  KE final: multiflow={ke_a[-1]:.8g} baseline={ke_b[-1]:.8g}")

    print("=== 2. conservation: epsd=0.2, 10 s, |sum_c(t)-sum_c(0)|/sum_c(0) < 1% ===")
    s0 = epsd02["sum_c"][0]
    relc = np.abs(epsd02["sum_c"] - s0) / s0
    print(f"sum_c(0)={s0:.6f}, max rel drift = {relc.max():.3e}  -> {'PASS' if relc.max() < 0.01 else 'FAIL'}")

    print("=== 3. monotonic smoothing of std(c) ===")
    std0_a, std1_a = epsd02["std_c"][0], epsd02["std_c"][-1]
    std0_b, std1_b = epsd0["std_c"][0], epsd0["std_c"][-1]
    print(f"epsd=0.2: std_c {std0_a:.6f} -> {std1_a:.6f}  -> {'PASS' if std1_a < 0.8 * std0_a else 'FAIL'}")
    drift_b = abs(std1_b - std0_b) / std0_b
    print(f"epsd=0  : std_c {std0_b:.6f} -> {std1_b:.6f} (drift {drift_b:.3e})  -> {'PASS' if drift_b < 0.05 else 'FAIL'}")

    print("=== 4. stability: NaN counts ===")
    for name, d in [("epsd0", epsd0), ("baseline", baseline), ("epsd0.2", epsd02)]:
        mx = int(np.nanmax(d["nan_count"]))
        print(f"{name}: max nan_count = {mx}  -> {'PASS' if mx == 0 else 'FAIL'}")

    print("=== extra: c_zbar trajectory (epsd=0.2) ===")
    idx = np.linspace(0, len(epsd02) - 1, 6).astype(int)
    for i in idx:
        print(f"  t={epsd02['t'][i]:6.2f}s  sum_c={epsd02['sum_c'][i]:10.4f}  std_c={epsd02['std_c'][i]:.5f}  c_zbar={epsd02['c_zbar'][i]:.4f}  ke={epsd02['ke'][i]:.4g}")


if __name__ == "__main__":
    main()
