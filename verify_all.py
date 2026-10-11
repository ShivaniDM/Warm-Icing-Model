"""
Recompute every reported hardware number from the raw counts and print the benchmark
numbers from the saved outputs.  Usage: python verify_all.py [--specs]
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
H9, H12, BM = (os.path.join(ROOT, d) for d in ("hardware_9q", "hardware_12q", "benchmark"))
fails = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        fails.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:58s} recomputed {got:8.3f}   quoted {want:8.3f}")


print("\n[A] 9-qubit circuit B_start357 (hardware_9q/counts): signal kept and XEB, full shots")
sys.path.insert(0, H9)
import score_9q

rows = {r["file"]: r for r in score_9q.score_all()}
quoted_9q = {
    "counts_ionq_forte_B_start357.json": (98, 0.96),
    "counts_ionq_forte_B_start357_run2.json": (71, 0.65),
    "counts_ionq_forte1_run3_qbraid_oq_100.json": (81, 0.79),
    "counts_ionq_forte_ent1_B_start357_100.json": (60, 0.46),
    "counts_ibm_fez.json": (38, 0.12),
    "counts_ibm_kingston.json": (30, 0.20),
    "counts_ibm_kingston_run2.json": (11, 0.10),
    "counts_ibm_marrakesh.json": (13, 0.11),
    "counts_rigetti_cepheus.json": (9, 0.03),
    "counts_ionqsim_5000.json": (51, 0.46),
    "counts_fake_kingston_1791639956.json": (59, 0.50),
}
for f, (k, x) in quoted_9q.items():
    r = rows[f]
    check(f"{f} signal kept (%)", 100 * r["kept"], k, 0.5)
    check(f"{f} XEB", r["xeb"], x, 0.005)

print("\n[B] 12-qubit circuit (hardware_12q): XEB fidelity, full shots, bit order as given")
os.chdir(H12)
sys.path.insert(0, H12)
from common import score

quoted_12q = {
    "counts_ionq_forte_ent1_real_100.json": 0.49,
    "counts_ibm_kingston_real_2000.json": 0.43,
    "counts_rigetti_cepheus_12q_1000.json": 0.02,
    "counts_aws_dm1_sim_1000.json": 0.97,
    "counts_ionq_forte_ent_noisemodel_300.json": 0.38,
    "counts_ionq_forte_ent_noisemodel_300_run2.json": 0.32,
    "counts_ionq_forte_ent_noisemodel_1000.json": 0.31,
    "counts_sim_fake_kingston_300.json": 0.61,
    "counts_sim_fake_marrakesh_300.json": 0.53,
    "counts_sim_fake_fez_300.json": 0.41,
    "counts_sim_ideal_300.json": 1.17,
}
for f, x in quoted_12q.items():
    s = score(json.load(open(f))["counts"])
    check(f"{f} XEB", s["xeb_fidelity"], x, 0.005)
s = score(json.load(open("counts_ionq_forte_ent1_real_100.json"))["counts"], big_endian=True)
print(f"        (IonQ 12q run with bits reversed: XEB {s['xeb_fidelity']:.2f} -> 'as given' is the right order)")

print("\n[C] 12-qubit depth ladder (hardware_12q/ladder): XEB at 66 / 132 / 198 ZZ")
sys.path.insert(0, os.path.join(H12, "ladder"))
import ladder as L

for dev, want in [("ibm_kingston_real", (0.24, 0.28, 0.43)), ("sim_fake_kingston", (0.55, 0.70, 0.54)),
                  ("sim_fake_marrakesh", (0.48, 0.57, 0.51)), ("sim_fake_fez", (0.39, 0.51, 0.36))]:
    d = json.load(open(os.path.join(H12, "ladder", f"ladder_counts_{dev}.json")))
    for n, w in zip(L.NAMES, want):
        check(f"{dev} {n} XEB", L.xeb(d["counts"][n], n)[0], w, 0.005)

print("\n[D] Local simulated-annealing baseline (hardware_12q/counts_local_annealing.json)")
from run_dwave import analyse

ann = json.load(open("counts_local_annealing.json"))
for label, want_tvd in [("anneal to posterior temperature (beta 0.1 -> 1)", 0.10),
                        ("default schedule (anneals cold, finds best maps)", 0.29)]:
    a = analyse({int(k): v for k, v in ann[label]["counts_by_map_index"].items()})
    check(f"annealing, {label[:34]}... TVD", a["tvd_vs_posterior"], want_tvd, 0.005)

if "--specs" in sys.argv:
    print("\n[E] Rebuild 12-qubit specs from main.py (expects max differences of 0)")
    subprocess.run([sys.executable, os.path.join(H12, "make_specs_12q.py")], check=True, cwd=H12)

print("\n[F] Benchmark numbers, read from saved outputs (regenerate: see RESULTS_MAP.txt)")
fv = json.load(open(os.path.join(BM, "fea_verification.json")))
rel = max(v["relative_difference"] for g in fv.values() for v in g.values())
print(f"  FEA vs lamination theory, worst relative curvature difference: {100 * rel:.3f}%   (quoted: about 0.03%)")
print("  Tsai et al. 2021 validation (benchmark/validation_tsai2021_flipchip.txt):")
for line in open(os.path.join(BM, "validation_tsai2021_flipchip.txt")):
    if line.startswith("  25") or line.startswith("RMS"):
        print("     " + line.rstrip())
import pandas as pd

rs = pd.read_csv(os.path.join(BM, "results.csv"))
red = (rs.nominal_rms_um / rs.calibrated_rms_um).groupby([rs.grid, rs.measurement, rs.noise_fraction]).median()
print(f"  calibration error reduction (median per condition): {red.min():.1f}x to {red.max():.1f}x   (quoted: 3-15x)")
for line in open(os.path.join(BM, "summary.txt")):
    if line.strip().startswith("3x3  full_field            0.03") or line.strip().startswith("3x4  full_field            0.03"):
        print("     summary.txt: " + line.rstrip())
acc = json.load(open(os.path.join(BM, "accuracy_budget.json")))
print("  accuracy after 1,000 steps (region error):  " +
      ", ".join(f"{k} {v['region_err_1000']:.3f}" for k, v in acc.items()))
cov = pd.read_csv(os.path.join(BM, "coverage_summary.csv"))
print(f"  95% interval coverage across conditions: {cov.coverage95.min():.3f} to {cov.coverage95.max():.3f}")
ad = pd.read_csv(os.path.join(BM, "adaptive_summary.csv"))
for _, r in ad[ad.noise == 0.01].iterrows():
    print(f"  adaptive 16 points, {r.grid} at 1% noise: exact maps adaptive {r.exact_adapt16}/7, "
          f"fixed {r.exact_fixed16}/7, random {r.exact_rand16}/7, full field {r.exact_full625}/7")
print("  stress tests: benchmark/stress_tests_results.txt")

print("\n" + ("ALL HARDWARE CHECKS PASS" if not fails else f"{len(fails)} CHECK(S) FAILED: " + "; ".join(fails)))
