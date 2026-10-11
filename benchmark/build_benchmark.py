"""
Generate the benchmark data from benchmark_config.json: hidden maps,
3D FEA warpage fields and noisy measurements.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fea3d as F

cfg = json.load(open(os.path.join(HERE, "benchmark_config.json")))
SPARSE = np.array([i * 25 + j for i in (0, 8, 16, 24) for j in (0, 8, 16, 24)])


def materials():
    m = cfg["materials"]
    core = dict(E=m["core"]["E_MPa"], nu=m["core"]["nu"], a=m["core"]["cte_per_K"])
    nom = dict(E=m["buildup_nominal"]["E_MPa"], nu=m["buildup_nominal"]["nu"], a=m["buildup_nominal"]["cte_per_K"])
    cu = m["copper"]
    f = m["buildup_cu_rich"]["copper_fraction"]
    E = (1 - f) * nom["E"] + f * cu["E_MPa"]
    a = ((1 - f) * nom["E"] * nom["a"] + f * cu["E_MPa"] * cu["cte_per_K"]) / E
    nu = (1 - f) * nom["nu"] + f * cu["nu"]
    return core, nom, dict(E=E, nu=nu, a=a)


def configure(regions):
    g = cfg["geometry_mm"]
    F.L, F.T_CORE, F.T_BU = g["side"], g["core_thickness"], g["buildup_thickness"]
    F.DT = cfg["thermal"]["measurement_temperature_C"] - cfg["thermal"]["zero_warpage_temperature_C"]
    F.CORE, F.BU_NOMINAL, F.BU_CU_RICH = materials()
    F.REGIONS = tuple(regions)


def main():
    os.chdir(HERE)
    t = cfg["tasks"]
    rng_maps = np.random.default_rng(t["map_seed"])
    fields, maps_all, verification = {}, {}, {}
    for rg in t["region_grids"]:
        configure(rg)
        key = f"{rg[0]}x{rg[1]}"
        R = rg[0] * rg[1]
        maps = rng_maps.integers(0, 2, (t["hidden_maps_per_grid"], R))
        sub = F.Substrate()
        sub.precompute()
        W = np.array([sub.solve(s.astype(float))[0] * 1e3 for s in maps])  # um
        fields[key], maps_all[key] = W, maps
        if key not in verification:
            gx = np.linspace(0, F.L, 25)
            X, Y = np.meshgrid(gx, gx, indexing="ij")
            m = ((np.abs(X - F.L / 2) < F.L * 0.3) & (np.abs(Y - F.L / 2) < F.L * 0.3)).ravel()
            x, y = X.ravel()[m] - F.L / 2, Y.ravel()[m] - F.L / 2
            M = np.column_stack([np.ones_like(x), x, y, (x ** 2 + y ** 2) / 2, x * y, x ** 2 - y ** 2])
            v = {}
            for name, s, top in [("all_nominal", np.zeros(R), F.BU_NOMINAL), ("all_cu_rich", np.ones(R), F.BU_CU_RICH)]:
                w = sub.solve(s)[0]
                k_fea = np.linalg.lstsq(M, w[m], rcond=None)[0][3]
                k_clt = F.clt_curvature(top, F.CORE, F.T_BU, F.T_CORE, F.DT)[0]
                v[name] = dict(curvature_fea_per_mm=float(k_fea), curvature_clt_per_mm=float(k_clt),
                               relative_difference=float(abs(abs(k_fea) / abs(k_clt) - 1)),
                               warpage_peak_to_valley_um=float(np.ptp(w) * 1e3))
            verification[key] = v
        print(key, "warpage p-p um:", np.round(np.ptp(W, axis=1), 1), flush=True)

    rng_noise = np.random.default_rng(t["noise_seed"])
    meas = {}
    rows = []
    for key in fields:
        for k, w in enumerate(fields[key]):
            rngw = np.ptp(w)
            for mname, idx in [("full_field", np.arange(625)), ("sparse_16", SPARSE)]:
                for nf in t["noise_fraction_of_warpage_range"]:
                    sig = nf * rngw
                    y = w[idx] + sig * rng_noise.standard_normal(len(idx))
                    tag = f"{key}|map{k}|{mname}|noise{nf:.2f}"
                    meas[tag] = y
                    for p, v in zip(idx, y):
                        rows.append([key, k, mname, nf, round(sig, 4), int(p), int(p // 25), int(p % 25), round(float(v), 4)])

    with open("hidden_maps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["grid", "map_id", "region_bits (region index = row*cols + col; row along x)"])
        for key, maps in maps_all.items():
            for k, s in enumerate(maps):
                wr.writerow([key, k, "".join(str(int(b)) for b in s)])
    np.savez_compressed("warpage_fields.npz", **{f"{k}_fields_um": v for k, v in fields.items()},
                        **{f"{k}_maps": v for k, v in maps_all.items()})
    np.savez_compressed("measurements.npz", **{k.replace("|", "__"): v for k, v in meas.items()},
                        sparse_16_indices=SPARSE)
    with open("measurements_long.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["grid", "map_id", "measurement", "noise_fraction", "noise_sigma_um", "point_index", "ix", "iy", "warpage_um"])
        wr.writerows(rows)
    json.dump(verification, open("fea_verification.json", "w"), indent=1)
    print(json.dumps(verification, indent=1))


if __name__ == "__main__":
    main()
