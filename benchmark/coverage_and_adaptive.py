"""
Coverage of the 95% prediction intervals, and adaptive vs fixed vs random
choice of measurement points.
"""
import json

import numpy as np
from scipy.special import ndtr

import build_benchmark as B
import run_benchmark as RB

cfg = B.cfg
meas, fields = np.load("measurements.npz"), np.load("warpage_fields.npz")
NOISES = cfg["tasks"]["noise_fraction_of_warpage_range"]
START = np.array([i * 25 + j for i in (8, 16) for j in (8, 16)])
N_MAX = 64
GRID64 = np.array([int(i) * 25 + int(j) for i in np.linspace(0, 24, 8).round() for j in np.linspace(0, 24, 8).round()])


def post_known_sigma(S, A, w0, y, sig):
    E = np.sum((S @ A.T + w0 - y) ** 2, 1) / (2 * sig ** 2)
    p = np.exp(-(E - E.min()))
    return p / p.sum()


def score(p, S, Af, w0f, wf, s_true):
    kt = int(sum(int(v) << i for i, v in enumerate(s_true)))
    pred = p @ (S @ Af.T + w0f)
    return dict(map_exact=bool((S[np.argmax(p)] == s_true).all()), p_true=float(p[kt]),
                rms_um=float(np.sqrt(np.mean((pred - wf) ** 2))))


def main():
    rng = np.random.default_rng(3)
    cover, adaptive = [], []
    for rg in cfg["tasks"]["region_grids"]:
        key = f"{rg[0]}x{rg[1]}"
        R = rg[0] * rg[1]
        M = 2 ** R
        S = ((np.arange(M)[:, None] >> np.arange(R)) & 1).astype(float)
        w0f, Af = RB.surrogate(rg)
        P = S @ Af.T + w0f
        wg, sg = fields[f"{key}_fields_um"][0], fields[f"{key}_maps"][0]
        sig_golden = float(np.sqrt(np.mean((P[int(sum(int(v) << i for i, v in enumerate(sg)))] - wg) ** 2)))
        print(key, "model discrepancy from reference board (um):", round(sig_golden, 4), flush=True)
        for k, (wf, s_true) in enumerate(zip(fields[f"{key}_fields_um"], fields[f"{key}_maps"])):
            if k == 0:
                continue
            for nf in NOISES:
                sig_noise = nf * np.ptp(wf)
                for mname, idx in [("full_field", np.arange(625)), ("sparse_16", B.SPARSE)]:
                    y = meas[f"{key}__map{k}__{mname}__noise{nf:.2f}"]
                    _, p, sig_fit = RB.posterior(S, Af[idx], w0f[idx], y)
                    sm_naive = np.sqrt(max(sig_fit ** 2 - sig_noise ** 2, 0.0))
                    sm = sig_golden
                    keep = p > 1e-8
                    pk, Pk = p[keep] / p[keep].sum(), P[keep]
                    cov = {}
                    for lab_, s_ in (("naive", sm_naive), ("reference_board", sm)):
                        F = pk @ ndtr((wf[None, :] - Pk) / max(s_, 1e-9))
                        cov[lab_] = float(np.mean((F > 0.025) & (F < 0.975)))
                    cover.append(dict(grid=key, map_id=k, measurement=mname, noise=nf,
                                      coverage95_naive=cov["naive"], coverage95=cov["reference_board"],
                                      sigma_model_um=float(sm)))
                yfull = meas[f"{key}__map{k}__full_field__noise{nf:.2f}"]
                chosen = [int(v) for v in START]
                curve = []
                while True:
                    p = post_known_sigma(S, Af[chosen], w0f[chosen], yfull[chosen], sig_noise)
                    curve.append(dict(n=len(chosen), **score(p, S, Af, w0f, wf, s_true)))
                    if len(chosen) >= N_MAX:
                        break
                    mean = p @ P
                    var = p @ (P ** 2) - mean ** 2
                    var[chosen] = -1
                    chosen.append(int(np.argmax(var)))
                fixed = score(post_known_sigma(S, Af[B.SPARSE], w0f[B.SPARSE], yfull[B.SPARSE], sig_noise),
                              S, Af, w0f, wf, s_true)
                fixed64 = score(post_known_sigma(S, Af[GRID64], w0f[GRID64], yfull[GRID64], sig_noise),
                                S, Af, w0f, wf, s_true)
                rand = []
                for _ in range(10):
                    idx = rng.choice(625, 16, replace=False)
                    rand.append(score(post_known_sigma(S, Af[idx], w0f[idx], yfull[idx], sig_noise), S, Af, w0f, wf, s_true))
                full = score(post_known_sigma(S, Af, w0f, yfull, sig_noise), S, Af, w0f, wf, s_true)
                adaptive.append(dict(grid=key, map_id=k, noise=nf, adaptive_curve=curve, fixed_grid_16=fixed,
                                     random_16_mean=dict(map_exact=float(np.mean([r["map_exact"] for r in rand])),
                                                         p_true=float(np.mean([r["p_true"] for r in rand])),
                                                         rms_um=float(np.mean([r["rms_um"] for r in rand]))),
                                     full_field_625=full, fixed_grid_64=fixed64, chosen_points=chosen))
        print(key, "done", flush=True)

    json.dump(dict(coverage=cover, adaptive=adaptive), open("coverage_adaptive.json", "w"), indent=1)

    import pandas as pd
    c = pd.DataFrame(cover)
    print("\n95% interval coverage of the true warpage field (median over 7 maps; target 0.95)")
    print("reference-board discrepancy:")
    print(c.groupby(["grid", "measurement", "noise"]).coverage95.median().unstack().round(3).to_string())
    print("naive (fit-minus-noise) discrepancy:")
    print(c.groupby(["grid", "measurement", "noise"]).coverage95_naive.median().unstack().round(3).to_string())
    print("\nAdaptive vs fixed vs random vs full field (7 maps per row; map 0 is the reference board)")
    rows = []
    for a in adaptive:
        a16 = next(x for x in a["adaptive_curve"] if x["n"] == 16)
        a8 = next(x for x in a["adaptive_curve"] if x["n"] == 8)
        a32 = next(x for x in a["adaptive_curve"] if x["n"] == 32)
        a64 = a["adaptive_curve"][-1]
        rows.append(dict(grid=a["grid"], noise=a["noise"],
                         exact_adapt8=a8["map_exact"], exact_adapt16=a16["map_exact"],
                         exact_fixed16=a["fixed_grid_16"]["map_exact"], exact_rand16=a["random_16_mean"]["map_exact"],
                         exact_full625=a["full_field_625"]["map_exact"], exact_adapt32=a32["map_exact"],
                         exact_adapt64=a64["map_exact"], exact_fixed64=a["fixed_grid_64"]["map_exact"],
                         rms_adapt64=a64["rms_um"], rms_fixed64=a["fixed_grid_64"]["rms_um"],
                         rms_adapt16=a16["rms_um"], rms_fixed16=a["fixed_grid_16"]["rms_um"],
                         rms_rand16=a["random_16_mean"]["rms_um"], rms_full625=a["full_field_625"]["rms_um"]))
    d = pd.DataFrame(rows)
    agg = d.groupby(["grid", "noise"]).agg(
        exact_adapt8=("exact_adapt8", "sum"), exact_adapt16=("exact_adapt16", "sum"),
        exact_fixed16=("exact_fixed16", "sum"), exact_rand16=("exact_rand16", "sum"),
        exact_full625=("exact_full625", "sum"), exact_adapt32=("exact_adapt32", "sum"),
        exact_adapt64=("exact_adapt64", "sum"), exact_fixed64=("exact_fixed64", "sum"),
        rms_adapt64=("rms_adapt64", "median"), rms_fixed64=("rms_fixed64", "median"),
        rms_adapt16=("rms_adapt16", "median"), rms_fixed16=("rms_fixed16", "median"),
        rms_rand16=("rms_rand16", "median"), rms_full625=("rms_full625", "median"))
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string())
    agg.round(4).to_csv("adaptive_summary.csv")
    c.groupby(["grid", "measurement", "noise"]).coverage95.median().round(4).to_csv("coverage_summary.csv")


if __name__ == "__main__":
    main()
