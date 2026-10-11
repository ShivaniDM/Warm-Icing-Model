"""
Exact Metropolis chain distribution after t steps from a false optimum
(9 regions, full field, 3% noise). Writes accuracy_budget.json.
"""
import json
import numpy as np
import build_benchmark as B
import run_benchmark as RB

R, M = 9, 512
S = ((np.arange(M)[:, None] >> np.arange(R)) & 1).astype(float)
meas, fields = np.load("measurements.npz"), np.load("warpage_fields.npz")
w0f, Af = RB.surrogate((3, 3))
rng = np.random.default_rng(5)
U, L = np.full((M, M), 1 / M), RB.local_Q(R)
budgets = [10, 30, 100, 300, 1000]
out = {}
for k in range(8):
    y = meas[f"3x3__map{k}__full_field__noise0.03"]
    E, p, _ = RB.posterior(S, Af, w0f, y)
    Qq = RB.quantum_Q(E, R, rng)
    mins = [i for i in range(M) if np.all(E[i ^ (1 << np.arange(R))] >= E[i]) and i != np.argmin(E)]
    start = min(mins, key=lambda i: E[i]) if mins else int(np.argmax(E))
    props = {"single-flip": L, "uniform": U, "quantum ideal": Qq,
             "quantum @ IonQ": .84 * Qq + .16 * U, "quantum @ IBM": .30 * Qq + .70 * U, "quantum @ Rigetti": .10 * Qq + .90 * U}
    for name, Q in props.items():
        acc = np.minimum(1.0, np.exp(-(E[None, :] - E[:, None])))
        P = Q * acc
        np.fill_diagonal(P, 0)
        np.fill_diagonal(P, 1 - P.sum(1))
        d = np.zeros(M); d[start] = 1
        res, t = {}, 0
        for b in budgets:
            while t < b:
                d = d @ P; t += 1
            res[b] = dict(tvd=0.5 * np.abs(d - p).sum(), region_err=np.abs(d @ S - p @ S).max())
        out.setdefault(name, []).append(res)
print("median over 8 posteriors (9 regions, full field, 3% noise), chain started in the best-fitting false optimum")
print(f"{'proposal':20s}" + "".join(f"  TVD@{b:<5d}" for b in budgets) + "   region-prob error @100 / @1000")
summary = {}
for name, rs in out.items():
    tv = [np.median([r[b]["tvd"] for r in rs]) for b in budgets]
    re100 = np.median([r[100]["region_err"] for r in rs]); re1000 = np.median([r[1000]["region_err"] for r in rs])
    summary[name] = dict(tvd=dict(zip(budgets, tv)), region_err_100=re100, region_err_1000=re1000)
    print(f"{name:20s}" + "".join(f"  {v:9.3f}" for v in tv) + f"   {re100:.3f} / {re1000:.3f}")
json.dump(summary, open("accuracy_budget.json", "w"), indent=1, default=float)
