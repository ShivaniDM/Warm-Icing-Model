"""
Calibrate every benchmark task and compute exact Metropolis spectral gaps for
classical and quantum proposals. Writes results.csv.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_benchmark as B
import fea3d as F
import warpage_ising_check as W

# Proposal fidelity f in Q_f = f*Q + (1-f)*Uniform, taken from the 9-qubit hardware runs (signal kept).
FIDELITY = {"ionq": 0.84, "ibm": 0.30, "rigetti": 0.10}
QUANTUM_3X4_SLICE = ("sparse_16", 0.03)


def surrogate(regions):
    B.configure(regions)
    k0, D0 = F.clt_curvature(F.BU_NOMINAL, F.CORE, F.T_BU, F.T_CORE, F.DT)
    k1, D1 = F.clt_curvature(F.BU_CU_RICH, F.CORE, F.T_BU, F.T_CORE, F.DT)
    dD = D1 / D0 - 1
    prm = dict(m0=k0, dm=k1 * (1 + dD) - k0, D0=1.0, dD=dD)
    plate = W.Plate(25, tuple(regions))
    h = F.L / 24
    R = regions[0] * regions[1]
    f = lambda s: plate.solve(np.asarray(s, float), **prm) * h ** 2 * 1e3
    w0 = f(np.zeros(R))
    A = np.column_stack([f(np.eye(R)[i]) - w0 for i in range(R)])
    return w0, A


def posterior(S, A, w0, y):
    sse = np.sum((S @ A.T + w0 - y) ** 2, 1)
    best = None
    for sig in np.geomspace(0.05, 50, 200):
        E = sse / (2 * sig ** 2)
        ll = -E.min() + np.log(np.exp(-(E - E.min())).sum()) - len(y) * np.log(sig)
        if best is None or ll > best[0]:
            best = (ll, sig)
    E = sse / (2 * best[1] ** 2)
    E -= E.min()
    p = np.exp(-E)
    return E, p / p.sum(), best[1]


def neighbours(M, R):
    return np.arange(M)[:, None] ^ (1 << np.arange(R))[None, :]


def annealing(E, R, rng, runs=200, flips=2000):
    M = len(E)
    nb = neighbours(M, R)
    x = rng.integers(M, size=runs)
    best = x.copy()
    for t in range(flips):
        T = 5.0 * (0.01 / 5.0) ** (t / flips)
        y = nb[x, rng.integers(R, size=runs)]
        acc = rng.random(runs) < np.exp(np.minimum(0.0, (E[x] - E[y]) / T))
        x = np.where(acc, y, x)
        best = np.where(E[x] < E[best], x, best)
    return float(np.mean(best == np.argmin(E)))


def quantum_Q(E, R, rng, n_gamma=4, n_t=2):
    M = len(E)
    S = ((np.arange(M)[:, None] >> np.arange(R)) & 1).astype(float)
    Z = 2 * S - 1
    X = np.column_stack([np.ones(M)] + [Z[:, i] for i in range(R)] +
                        [Z[:, i] * Z[:, j] for i in range(R) for j in range(i + 1, R)])
    c = np.linalg.lstsq(X, E, rcond=None)[0]
    alpha = np.sqrt(R) / np.sqrt(np.sum(c[1:] ** 2))
    Hmix = np.zeros((M, M))
    for k in range(R):
        Hmix[np.arange(M), np.arange(M) ^ (1 << k)] = 1.0
    Q = np.zeros((M, M))
    for _ in range(n_gamma):
        g = rng.uniform(0.25, 0.6)
        vals, vecs = np.linalg.eigh(np.diag((1 - g) * alpha * (E - E.mean())) + g * Hmix)
        for _ in range(n_t):
            t = rng.uniform(2, 20)
            U = (vecs * np.exp(-1j * vals * t)) @ vecs.conj().T
            Q += np.abs(U) ** 2
    Q /= n_gamma * n_t
    return 0.5 * (Q + Q.T)


def local_Q(R):
    M = 2 ** R
    Q = np.zeros((M, M))
    for k in range(R):
        Q[np.arange(M), np.arange(M) ^ (1 << k)] = 1.0 / R
    return Q


def main():
    os.chdir(HERE)
    cfg = B.cfg
    meas = np.load("measurements.npz")
    fields = np.load("warpage_fields.npz")
    rng = np.random.default_rng(11)
    rows = []
    for rg in cfg["tasks"]["region_grids"]:
        key = f"{rg[0]}x{rg[1]}"
        R = rg[0] * rg[1]
        M = 2 ** R
        w0_full, A_full = surrogate(rg)
        S = ((np.arange(M)[:, None] >> np.arange(R)) & 1).astype(float)
        Uq = np.full((M, M), 1.0 / M)
        Lq = local_Q(R)
        for k, (wf, s_true) in enumerate(zip(fields[f"{key}_fields_um"], fields[f"{key}_maps"])):
            kt = int(sum(int(v) << i for i, v in enumerate(s_true)))
            for mname, idx in [("full_field", np.arange(625)), ("sparse_16", B.SPARSE)]:
                for nf in cfg["tasks"]["noise_fraction_of_warpage_range"]:
                    t0 = time.time()
                    y = meas[f"{key}__map{k}__{mname}__noise{nf:.2f}"]
                    A, w0 = A_full[idx], w0_full[idx]
                    E, p, sig = posterior(S, A, w0, y)
                    kmap = int(np.argmax(p))
                    pred = p @ (S @ A_full.T + w0_full)
                    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
                    s_ls = np.round(lsq_linear(A, y - w0, bounds=(0, 1)).x)
                    nmin = int(np.sum([np.all(E[np.arange(M)[i] ^ (1 << np.arange(R))] >= E[i]) for i in range(M)]))
                    row = dict(grid=key, map_id=k, measurement=mname, noise_fraction=nf,
                               noise_sigma_um=round(nf * np.ptp(wf), 4), fitted_sigma_um=round(float(sig), 4),
                               map_regions_correct=int((S[kmap] == s_true).sum()), regions=R,
                               p_true_map=round(float(p[kt]), 4), rank_true_map=int((p > p[kt]).sum()) + 1,
                               maps_for_90pct=int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.9) + 1),
                               local_optima=nmin,
                               nominal_rms_um=round(rms(w0_full - wf), 3), calibrated_rms_um=round(rms(pred - wf), 3),
                               lsq_round_regions_correct=int((s_ls == s_true).sum()),
                               sa_success=annealing(E, R, rng))
                    if R == 9 or (mname, nf) == QUANTUM_3X4_SLICE:
                        Qq = quantum_Q(E, R, rng)
                        g_loc, g_uni = W.spectral_gap(E, Lq), W.spectral_gap(E, Uq)
                        best = max(g_loc, g_uni)
                        row.update(gap_single_flip_vs_best=g_loc / best, gap_uniform_vs_best=g_uni / best,
                                   gap_quantum_ideal_vs_best=W.spectral_gap(E, Qq) / best)
                        for name, f in FIDELITY.items():
                            row[f"gap_quantum_{name}_vs_best"] = W.spectral_gap(E, f * Qq + (1 - f) * Uq) / best
                    rows.append(row)
                    print(key, k, mname, nf, row["map_regions_correct"], row["calibrated_rms_um"],
                          round(row.get("gap_quantum_ionq_vs_best", float("nan")), 2), f"{time.time() - t0:.1f}s", flush=True)
    cols = sorted({c for r in rows for c in r}, key=lambda c: list(rows[-1]).index(c) if c in rows[-1] else 99)
    cols = list(dict.fromkeys(list(rows[0]) + [c for r in rows for c in r]))
    with open("results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
