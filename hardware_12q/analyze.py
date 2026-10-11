"""
Score counts_*.json against the ideal distribution (XEB, P_better) in both bit
orders, plus a 100-shot bootstrap view.  Usage: python analyze.py
"""
import glob
import json

from common import score

rows = []
for f in sorted(glob.glob("counts_*.json")):
    d = json.load(open(f))
    if "counts" not in d:
        continue
    counts = d["counts"]
    for be in (False, True):
        s = score(counts, big_endian=be)
        rows.append((d.get("device", f) + (" [bits reversed]" if be else ""), s))
print(f"{'device':40s} {'shots':>6s} {'XEB fidelity':>14s} {'P_better':>9s} {'kept':>12s} {'top50':>7s}")
for name, s in rows:
    print(f"{name:40s} {s['shots']:6d} {s['xeb_fidelity']:7.2f} ±{s['xeb_err']:5.2f} {s['p_better']:9.3f} "
          f"{100 * s['signal_kept']:6.0f}% ±{100 * s['signal_kept_err']:3.0f} {s['mass_on_ideal_top50']:7.3f}")
first = rows[0][1] if rows else None
if first:
    print(f"\nreference: ideal P_better {first['ideal']:.3f}, random floor {first['floor']:.3f}; "
          f"ideal top-50 mass {first['ideal_top50']:.3f}, uniform {first['uniform_top50']:.3f}")

# equal-shots view: 100-shot bootstrap, 200 draws
import numpy as np
print("\nEqual-shots view: XEB fidelity at 100 shots (bootstrap mean and spread)")
rng = np.random.default_rng(0)
for f in sorted(glob.glob("counts_*.json")):
    d = json.load(open(f))
    if "counts" not in d:
        continue
    keys = list(d["counts"]); w = np.array([d["counts"][k] for k in keys], float); w /= w.sum()
    for be in (False, True):
        vals = []
        for _ in range(200):
            draw = rng.choice(len(keys), 100, p=w)
            c = {}
            for i in draw:
                c[keys[i]] = c.get(keys[i], 0) + 1
            vals.append(score(c, big_endian=be)["xeb_fidelity"])
        print(f"{d.get('device', f) + (' [bits reversed]' if be else ''):40s} {np.mean(vals):6.2f} ± {np.std(vals):.2f}")
