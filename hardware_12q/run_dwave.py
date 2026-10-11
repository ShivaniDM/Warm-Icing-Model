"""
The 12-region posterior on a D-Wave annealer (needs DWAVE_API_TOKEN).
Usage: python run_dwave.py 1000
"""
import json
import os
import sys

import numpy as np

from common import HERE, R, SPEC

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def ising_from_spec():
    J = np.array(SPEC["J"])
    h = np.array(SPEC["h"])
    return {i: float(h[i]) for i in range(R)}, {(i, j): float(J[i, j]) for i in range(R) for j in range(i + 1, R)}


def to_index(sample):
    return int(sum(((1 - int(sample[i])) // 2) << i for i in range(R)))


def analyse(idx_counts):
    E = np.array(SPEC["E"])
    post = np.array(SPEC["p_posterior"])
    total = sum(idx_counts.values())
    q = np.zeros(2 ** R)
    for k, n in idx_counts.items():
        q[int(k)] += n
    q /= total
    best = int(np.argmax(post))
    top = np.argsort(-post)
    m90 = int(np.searchsorted(np.cumsum(post[top]), 0.9) + 1)
    betas = np.geomspace(0.05, 20, 200)
    tv = [0.5 * np.abs(q - np.exp(-b * (E - E.min())) / np.exp(-b * (E - E.min())).sum()).sum() for b in betas]
    return dict(reads=total, tvd_vs_posterior=float(0.5 * np.abs(q - post).sum()),
                p_best_map=float(q[best]), posterior_p_best_map=float(post[best]),
                mass_on_posterior_90pct_set=float(q[top[:m90]].sum()), maps_in_90pct_set=m90,
                best_effective_beta=float(betas[int(np.argmin(tv))]), tvd_at_best_beta=float(min(tv)))


if __name__ == "__main__":
    reads = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    from dwave.system import DWaveSampler, EmbeddingComposite
    h, J = ising_from_spec()
    scale = max(max(abs(v) for v in h.values()) / 4.0, max(abs(v) for v in J.values()) / 1.0, 1e-12)
    hs, Js = {k: v / scale for k, v in h.items()}, {k: v / scale for k, v in J.items()}
    sampler = EmbeddingComposite(DWaveSampler())
    ss = sampler.sample_ising(hs, Js, num_reads=reads, label="warpcal_12q")
    idx = {}
    for rec in ss.data(["sample", "num_occurrences"]):
        k = to_index(rec.sample)
        idx[k] = idx.get(k, 0) + int(rec.num_occurrences)
    info = dict(energy_scale_divisor=scale, solver=ss.info.get("problem_id", ""), timing=ss.info.get("timing", {}))
    json.dump(dict(device="dwave", counts_by_map_index=idx, **info), open(os.path.join(HERE, "counts_dwave.json"), "w"), default=str)
    print(analyse(idx))
