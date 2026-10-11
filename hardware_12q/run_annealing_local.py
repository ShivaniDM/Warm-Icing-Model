"""
Classical simulated annealing (dwave-samplers) on the 12-region posterior.
Usage: python run_annealing_local.py 1000
"""
import json
import os
import sys

from dwave.samplers import SimulatedAnnealingSampler

from common import HERE
from run_dwave import analyse, ising_from_spec, to_index

reads = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
h, J = ising_from_spec()
results = {}
for label, beta_range in [("anneal to posterior temperature (beta 0.1 -> 1)", (0.1, 1.0)),
                          ("default schedule (anneals cold, finds best maps)", None)]:
    kw = dict(num_reads=reads, seed=7)
    if beta_range:
        kw["beta_range"] = beta_range
    ss = SimulatedAnnealingSampler().sample_ising(h, J, **kw)
    idx = {}
    for rec in ss.data(["sample", "num_occurrences"]):
        k = to_index(rec.sample)
        idx[k] = idx.get(k, 0) + int(rec.num_occurrences)
    res = analyse(idx)
    results[label] = dict(res, counts_by_map_index=idx)
    print(f"\n{label}")
    for key, v in res.items():
        print(f"  {key:30s} {v:.4f}" if isinstance(v, float) else f"  {key:30s} {v}")

json.dump(results, open(os.path.join(HERE, "counts_local_annealing.json"), "w"))
print("\nsaved counts_local_annealing.json")
print("Reading: tvd_vs_posterior = distance from the exact posterior (0 = perfect, 1 = disjoint);")
print("mass_on_posterior_90pct_set = share of samples on the maps that hold 90% of the true probability.")
