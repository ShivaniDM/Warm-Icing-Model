"""
Calibration when reality differs from the model: S0 baseline, S1 materials off,
S2 6x6 true map, S3 x1.10 tool bias, S4 3 um noise + bias. Map 0 is the reference
board used to size model error.
"""
import json

import numpy as np
from scipy.special import ndtr

import build_benchmark as B
import fea3d as F
import run_benchmark as RB

rng = np.random.default_rng(42)
R = 9
S = ((np.arange(512)[:, None] >> np.arange(R)) & 1).astype(float)
w0f, Af = RB.surrogate((3, 3))
P = S @ Af.T + w0f
maps9 = np.load("warpage_fields.npz")["3x3_maps"]


def reality(regions, maps, materials=None):
    B.configure(regions)
    if materials:
        F.CORE, F.BU_NOMINAL, F.BU_CU_RICH = materials
    sub = F.Substrate()
    sub.precompute()
    return np.array([sub.solve(np.asarray(s, float))[0] * 1e3 for s in maps])


def materials_off():
    m = B.cfg["materials"]
    core = dict(E=m["core"]["E_MPa"], nu=m["core"]["nu"], a=15e-6)
    nom = dict(E=7500, nu=0.30, a=25e-6)
    f, cu = 0.25, m["copper"]
    E = (1 - f) * nom["E"] + f * cu["E_MPa"]
    rich = dict(E=E, nu=(1 - f) * 0.30 + f * cu["nu"], a=((1 - f) * nom["E"] * nom["a"] + f * cu["E_MPa"] * cu["cte_per_K"]) / E)
    return core, nom, rich


def calibrate(y, sig_noise, sig_ref):
    E, p, _ = RB.posterior(S, Af, w0f, y)
    sig = np.sqrt(sig_noise ** 2 + sig_ref ** 2)
    E = np.sum((P - y) ** 2, 1) / (2 * sig ** 2)
    p = np.exp(-(E - E.min()))
    return p / p.sum()


def coverage(p, w_true, s):
    keep = p > 1e-8
    pk, Pk = p[keep] / p[keep].sum(), P[keep]
    Fc = pk @ ndtr((w_true[None, :] - Pk) / max(s, 1e-9))
    return float(np.mean((Fc > 0.025) & (Fc < 0.975)))


def smooth_rms(resid, sig_noise, k=5):
    g = resid.reshape(25, 25)
    pad = np.pad(g, k // 2, mode="edge")
    sm = np.mean([pad[i:i + 25, j:j + 25] for i in range(k) for j in range(k)], axis=0)
    return float(np.sqrt(max(np.mean(sm ** 2) - sig_noise ** 2 / k ** 2, 0)))


def run(name, W_true, truth9, noise_frac=None, noise_um=None, scale=1.0, recal=False):
    global P
    P_base = S @ Af.T + w0f
    P = P_base
    rows = []
    sig_n = lambda w: noise_um if noise_um else noise_frac * np.ptp(w)
    w_ref = W_true[0]
    y_ref = scale * w_ref + sig_n(w_ref) * rng.standard_normal(625)
    k_ref = int(sum(int(v) << i for i, v in enumerate(truth9[0]))) if truth9 is not None else None
    if k_ref is None:
        k_ref = int(np.argmin(np.mean((P - y_ref) ** 2, 1)))
    if recal:
        X = np.column_stack([w0f, Af @ S[k_ref]])
        a, b = np.linalg.lstsq(X, y_ref, rcond=None)[0]
        P = a * w0f + b * (S @ Af.T)
    sig_ref = smooth_rms(y_ref - P[k_ref], sig_n(w_ref))
    for k in range(1, len(W_true)):
        w = W_true[k]
        y = scale * w + sig_n(w) * rng.standard_normal(625)
        p = calibrate(y, sig_n(w), sig_ref)
        pred = p @ P / scale if recal else p @ P
        r = dict(rms_nominal=float(np.sqrt(np.mean((w0f - w) ** 2))), rms_cal=float(np.sqrt(np.mean((pred - w) ** 2))),
                 coverage95=coverage(p, scale * w if recal else w, sig_ref))
        if truth9 is not None:
            kmap = int(np.argmax(p))
            r.update(map_exact=bool((S[kmap] == truth9[k]).all()), regions_correct=int((S[kmap] == truth9[k]).sum()))
        rows.append(r)
    agg = {key: float(np.median([r[key] for r in rows])) for key in rows[0] if key != "map_exact"}
    if truth9 is not None:
        agg["maps_exact"] = f"{sum(r['map_exact'] for r in rows)}/{len(rows)}"
    agg["model_error_from_reference_um"] = sig_ref
    if recal:
        agg["recal_factors"] = [round(float(a), 3), round(float(b), 3)]
    print(f"{name:34s}", {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in agg.items()}, flush=True)
    return agg


def main():
    out = {}
    W0 = np.load("warpage_fields.npz")["3x3_fields_um"]
    W1 = reality((3, 3), maps9, materials_off())
    maps36 = rng.integers(0, 2, (8, 36))
    W2 = reality((6, 6), maps36)
    np.savez("stress_fields.npz", W1=W1, W2=W2, maps36=maps36)
    for rc in (False, True):
        tag = " + reference-board recal" if rc else ""
        for nm, args in [("S0 baseline (3% noise)", (W0, maps9, dict(noise_frac=0.03))),
                         ("S1 materials 10-20% off", (W1, maps9, dict(noise_frac=0.03))),
                         ("S2 finer 6x6 truth", (W2, None, dict(noise_frac=0.03))),
                         ("S3 tool bias x1.10", (W0, maps9, dict(noise_frac=0.03, scale=1.10))),
                         ("S4 iNEMI noise 3um + bias", (W0, maps9, dict(noise_um=3.0, scale=1.10)))]:
            out[nm + tag] = run(nm + tag, args[0], args[1], recal=rc, **args[2])
    json.dump(out, open("stress_tests.json", "w"), indent=1)


if __name__ == "__main__":
    main()
