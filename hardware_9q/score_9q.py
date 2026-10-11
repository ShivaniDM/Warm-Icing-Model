"""
Score every 9-qubit run in counts/ (signal kept and XEB), plus 100-shot subsamples.
Runs are never pooled. qBraid results for Rigetti list qubit 0 first and are reversed.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(HERE, "experiment_spec.json")))
R, E = spec["R"], np.array(spec["energies"])
Zi = {tuple(z): k for k, z in enumerate(spec["Z"])}
c = [x for x in spec["circuits"] if x["name"] == "B_start357"][0]
ideal, k0 = np.array(c["ideal_distribution"]), c["start_index"]
lower = E < E[k0]
pi, pu = c["ideal_p_down"], c["uniform_p_down"]
D = 2 ** R
den = (ideal ** 2).sum() - 1 / D


def idx(bits):
    b = bits.replace(" ", "")[-R:]
    return Zi[tuple(+1 if b[R - 1 - i] == "0" else -1 for i in range(R))]


def metrics(xs):
    return (lower[xs].mean() - pu) / (pi - pu), (ideal[xs].mean() - 1 / D) / den


def score_file(path, rng):
    d = json.load(open(path))
    cnt = d["counts"].get("B_start357")
    if not cnt:
        return None
    rev = "rigetti" in str(d.get("backend", ""))
    xs = np.repeat([idx(b[::-1] if rev else b) for b in cnt], list(cnt.values()))
    n = len(xs)
    kept, xeb = metrics(xs)
    pb = lower[xs].mean()
    out = dict(file=os.path.basename(path), backend=d.get("backend"), run=d.get("run", ""), shots=n,
               kept=kept, kept_err=np.sqrt(pb * (1 - pb) / n) / (pi - pu),
               xeb=xeb, xeb_err=ideal[xs].std() / np.sqrt(n) / den)
    sub = [metrics(rng.choice(xs, 100, replace=False)) for _ in range(500)] if n > 100 else [metrics(xs)]
    out.update(kept100=np.mean([s[0] for s in sub]), kept100_sd=np.std([s[0] for s in sub]),
               xeb100=np.mean([s[1] for s in sub]), xeb100_sd=np.std([s[1] for s in sub]))
    return out


def score_all():
    rng = np.random.default_rng(0)
    rows = [r for f in sorted(glob.glob(os.path.join(HERE, "counts", "counts_*.json")))
            if (r := score_file(f, rng))]
    return rows


if __name__ == "__main__":
    print(f"B_start357: ideal P_better {pi:.3f}, random floor {pu:.3f}")
    print(f"{'file':44s}{'shots':>6s}{'signal kept':>15s}{'XEB':>15s}{'kept @100':>13s}{'XEB @100':>13s}")
    for r in score_all():
        print(f"{r['file']:44s}{r['shots']:6d}{100*r['kept']:8.0f}% ±{100*r['kept_err']:3.0f}"
              f"{r['xeb']:9.2f} ±{r['xeb_err']:4.2f}{100*r['kept100']:7.0f}% ±{100*r['kept100_sd']:2.0f}"
              f"{r['xeb100']:8.2f} ±{r['xeb100_sd']:4.2f}")
