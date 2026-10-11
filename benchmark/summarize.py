"""Aggregate results.csv into summary.json, summary.txt and figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results.csv")
df["exact_map"] = df.map_regions_correct == df.regions
df["lsq_exact"] = df.lsq_round_regions_correct == df.regions
df["uncertain"] = df.maps_for_90pct > 1
df["error_reduction"] = df.nominal_rms_um / df.calibrated_rms_um

g = df.groupby(["grid", "measurement", "noise_fraction"])
summary = g.agg(tasks=("map_id", "size"),
                map_exact=("exact_map", "mean"), lsq_exact=("lsq_exact", "mean"),
                uncertain=("uncertain", "mean"), true_map_rank_median=("rank_true_map", "median"),
                local_optima_median=("local_optima", "median"),
                nominal_rms_um=("nominal_rms_um", "median"), calibrated_rms_um=("calibrated_rms_um", "median"),
                error_reduction_median=("error_reduction", "median"),
                sa_success_median=("sa_success", "median")).reset_index()
gapcols = [c for c in df.columns if c.startswith("gap_")]
gaps = df.dropna(subset=["gap_quantum_ideal_vs_best"]).groupby(["grid", "measurement", "noise_fraction"])[gapcols].median().reset_index()
pd.set_option("display.width", 220)
print(summary.round(3).to_string(index=False))
print()
print(gaps.round(3).to_string(index=False))
overall = df.dropna(subset=["gap_quantum_ideal_vs_best"]).groupby("grid")[gapcols].median()
print()
print(overall.round(4).to_string())
json.dump(dict(per_task_group=summary.round(4).to_dict("records"), sampler_speed=gaps.round(4).to_dict("records"),
               sampler_speed_overall_median=overall.round(4).to_dict("index")), open("summary.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for (grid, meas), sub in summary.groupby(["grid", "measurement"]):
    axs[0].plot(sub.noise_fraction * 100, sub.calibrated_rms_um, marker="o", label=f"{grid}, {meas}")
axs[0].axhline(summary.nominal_rms_um.median(), color="grey", ls="--")
axs[0].text(5.5, summary.nominal_rms_um.median() * 0.95, "nominal design model (median)", fontsize=8, ha="center", va="top")
axs[0].set_xlabel("measurement noise, % of warpage range")
axs[0].set_ylabel("full-field prediction error, RMS µm (median)")
axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title("Calibration accuracy", fontsize=10)
lab = {"gap_quantum_ideal_vs_best": "quantum ideal", "gap_quantum_ionq_vs_best": "quantum @ IonQ f=0.84",
       "gap_quantum_ibm_vs_best": "quantum @ IBM f=0.30", "gap_quantum_rigetti_vs_best": "quantum @ Rigetti f=0.10"}
cols = {"gap_quantum_ideal_vs_best": "#1d3f87", "gap_quantum_ionq_vs_best": "#2f6fdf",
        "gap_quantum_ibm_vs_best": "#8fb0ec", "gap_quantum_rigetti_vs_best": "#c5d4f2"}
sub = gaps[gaps.grid == "3x3"].groupby("noise_fraction")[list(lab)].median()
for c in lab:
    axs[1].plot(sub.index * 100, sub[c], marker="o", color=cols[c], label=lab[c])
axs[1].axhline(1, color="grey", ls=":")
axs[1].set_xlabel("measurement noise, % of warpage range")
axs[1].set_ylabel("mixing speed vs best classical (median)")
axs[1].set_title("Sampler speed, 9 regions (both measurement types)", fontsize=10)
axs[1].legend(frameon=False, fontsize=8)
fig.savefig("benchmark_summary.png", dpi=150, bbox_inches="tight")
