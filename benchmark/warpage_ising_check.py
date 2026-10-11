"""
Plate surrogate, Ising form of the warpage posterior, and exact Metropolis
spectral gaps (Layden et al., Nature 619, 282 (2023)).
"""
import itertools
import json
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RNG = np.random.default_rng(7)


class Plate:
    def __init__(self, n_nodes=25, region_shape=(3, 3)):
        self.N = n_nodes
        self.rx, self.ry = region_shape
        self.R = self.rx * self.ry
        N = self.N
        idx = lambda i, j: i * N + j

        def op(rows):
            data, ri, ci = [], [], []
            for r, entries in enumerate(rows):
                for c, v in entries:
                    ri.append(r)
                    ci.append(c)
                    data.append(v)
            return sp.csr_matrix((data, (ri, ci)), shape=(len(rows), N * N))

        xx_rows, xx_pts = [], []
        for i in range(1, N - 1):
            for j in range(N):
                xx_rows.append([(idx(i - 1, j), 1), (idx(i, j), -2), (idx(i + 1, j), 1)])
                xx_pts.append((i, j))
        yy_rows, yy_pts = [], []
        for i in range(N):
            for j in range(1, N - 1):
                yy_rows.append([(idx(i, j - 1), 1), (idx(i, j), -2), (idx(i, j + 1), 1)])
                yy_pts.append((i, j))
        xy_rows, xy_pts = [], []
        for i in range(N - 1):
            for j in range(N - 1):
                xy_rows.append([(idx(i + 1, j + 1), 1), (idx(i + 1, j), -1),
                                (idx(i, j + 1), -1), (idx(i, j), 1)])
                xy_pts.append((i + 0.5, j + 0.5))
        self.Dxx, self.Dyy, self.Dxy = op(xx_rows), op(yy_rows), op(xy_rows)
        self.reg_xx = np.array([self._region(p) for p in xx_pts])
        self.reg_yy = np.array([self._region(p) for p in yy_pts])
        self.reg_xy = np.array([self._region(p) for p in xy_pts])

        pinned = {idx(0, 0), idx(0, N - 1), idx(N - 1, 0)}
        self.free = np.array([k for k in range(N * N) if k not in pinned])
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        B = np.column_stack([np.ones(N * N), ii.ravel(), jj.ravel()])
        self.P_plane = np.eye(N * N) - B @ np.linalg.pinv(B)
        self.centroids = np.array(
            [((a + 0.5) / self.rx, (b + 0.5) / self.ry)
             for a in range(self.rx) for b in range(self.ry)])

    def _region(self, p):
        i, j = p
        a = min(int(i / (self.N - 1) * self.rx), self.rx - 1)
        b = min(int(j / (self.N - 1) * self.ry), self.ry - 1)
        return a * self.ry + b

    def solve(self, s, m0=1.0, dm=0.5, D0=1.0, dD=0.0):
        s = np.asarray(s, dtype=float)
        mt = lambda reg: m0 + dm * s[reg]
        Df = lambda reg: D0 * (1.0 + dD * s[reg])
        K = (self.Dxx.T @ sp.diags(Df(self.reg_xx)) @ self.Dxx
             + self.Dyy.T @ sp.diags(Df(self.reg_yy)) @ self.Dyy
             + 2 * self.Dxy.T @ sp.diags(Df(self.reg_xy)) @ self.Dxy)
        f = self.Dxx.T @ mt(self.reg_xx) + self.Dyy.T @ mt(self.reg_yy)
        Kf = K[self.free][:, self.free].tocsc()
        w = np.zeros(self.N * self.N)
        w[self.free] = spla.spsolve(Kf, -f[self.free])
        return self.P_plane @ w


def all_configs(R):
    return np.array(list(itertools.product([0, 1], repeat=R)), dtype=float)


def linearity_report(plate, dD_values, obs_idx):
    R = plate.R
    S = all_configs(R)
    out = {}
    for dD in dD_values:
        W = np.array([plate.solve(s, dD=dD)[obs_idx] for s in S])
        X1 = np.column_stack([np.ones(len(S)), S])
        coef1, *_ = np.linalg.lstsq(X1, W, rcond=None)
        res1 = W - X1 @ coef1
        pairs = [S[:, i] * S[:, j] for i in range(R) for j in range(i + 1, R)]
        X2 = np.column_stack([X1] + pairs)
        coef2, *_ = np.linalg.lstsq(X2, W, rcond=None)
        res2 = W - X2 @ coef2
        scale = np.sqrt(np.mean((W - W.mean(0)) ** 2))
        out[f"dD={dD}"] = {
            "affine_rel_rms_error": float(np.sqrt(np.mean(res1 ** 2)) / scale),
            "affine_max_abs_error_over_range": float(np.max(np.abs(res1)) / np.ptp(W)),
            "affine+pairwise_rel_rms_error": float(np.sqrt(np.mean(res2 ** 2)) / scale),
        }
    return out


def build_posterior(plate, obs_idx, s_true, sigma_frac, dD=0.0):
    R = plate.R
    w0 = plate.solve(np.zeros(R), dD=dD)[obs_idx]
    A = np.column_stack([plate.solve(np.eye(R)[i], dD=dD)[obs_idx] - w0 for i in range(R)])
    y_clean = plate.solve(s_true, dD=dD)[obs_idx]
    sigma = sigma_frac * np.ptp(y_clean)
    y = y_clean + sigma * RNG.standard_normal(len(obs_idx))
    c = y - w0
    d = c - A.sum(1) / 2
    G = A.T @ A
    J = G / (4 * sigma ** 2)
    h = A.T @ d / (2 * sigma ** 2)
    return dict(A=A, w0=w0, y=y, sigma=sigma, J=J, h=h)


def ising_energies(J, h, R):
    Z = 2 * all_configs(R) - 1
    Joff = J - np.diag(np.diag(J))
    return 0.5 * np.einsum("ki,ij,kj->k", Z, Joff, Z) - Z @ h, Z


def exact_energies_direct(post, R):
    S = all_configs(R)
    r = S @ post["A"].T - (post["y"] - post["w0"])
    return np.sum(r ** 2, 1) / (2 * post["sigma"] ** 2)


def coupling_stats(J, centroids):
    R = J.shape[0]
    off = [(abs(J[i, j]), J[i, j], np.linalg.norm(centroids[i] - centroids[j]))
           for i in range(R) for j in range(i + 1, R)]
    mags = np.array([o[0] for o in off])
    signs = np.array([o[1] for o in off])
    dist = np.array([o[2] for o in off])
    near = mags[dist <= dist.min() + 1e-9]
    far = mags[dist >= dist.max() - 1e-9]
    return {
        "n_pairs": len(off),
        "frac_pairs_above_1pct_of_max": float(np.mean(mags > 0.01 * mags.max())),
        "frac_pairs_above_10pct_of_max": float(np.mean(mags > 0.10 * mags.max())),
        "frac_positive_couplings(antiferro in z)": float(np.mean(signs > 0)),
        "mean_|J|_nearest / mean_|J|_farthest": float(near.mean() / far.mean()),
        "corr(|J|, distance)": float(np.corrcoef(mags, dist)[0, 1]),
    }


def spectral_gap(E, Q):
    dE = np.abs(E[:, None] - E[None, :])
    S = Q * np.exp(-0.5 * dE)
    np.fill_diagonal(S, 0.0)
    acc = np.minimum(1.0, np.exp(-(E[None, :] - E[:, None])))
    Poff = Q * acc
    np.fill_diagonal(Poff, 0.0)
    np.fill_diagonal(S, 1.0 - Poff.sum(1))
    ev = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]
    return float(1.0 - ev[1])


def local_proposal(Z):
    n = Z.shape[1]
    Ham = (n - (Z @ Z.T)) / 2
    return (np.isclose(Ham, 1)).astype(float) / n


def uniform_proposal(M):
    return np.full((M, M), 1.0 / M)


def quantum_proposal(E, R, J, h, n_gamma=8, n_t=3):
    M = 2 ** R
    Joff = J - np.diag(np.diag(J))
    alpha = np.sqrt(R) / np.sqrt(np.sum(np.triu(Joff, 1) ** 2) + np.sum(h ** 2))
    Hmix = np.zeros((M, M))
    for k in range(R):
        flip = np.arange(M) ^ (1 << (R - 1 - k))
        Hmix[np.arange(M), flip] = 1.0
    Hp = np.diag(E - E.mean())
    Q = np.zeros((M, M))
    count = 0
    for _ in range(n_gamma):
        g = RNG.uniform(0.25, 0.6)
        vals, vecs = np.linalg.eigh((1 - g) * alpha * Hp + g * Hmix)
        for _ in range(n_t):
            t = RNG.uniform(2, 20)
            U = (vecs * np.exp(-1j * vals * t)) @ vecs.conj().T
            Q += np.abs(U) ** 2
            count += 1
    Q /= count
    return 0.5 * (Q + Q.T)


def main():
    t0 = time.time()
    report = {}
    plate = Plate(n_nodes=25, region_shape=(3, 3))
    R = plate.R
    N = plate.N
    dense_obs = np.arange(N * N)
    sparse_obs = np.array([i * N + j for i in range(2, N, 6) for j in range(2, N, 6)])

    report["linearity_full_field"] = linearity_report(plate, [0.0, 0.05, 0.15, 0.30], dense_obs)

    s_true = RNG.integers(0, 2, R).astype(float)
    report["s_true"] = s_true.astype(int).tolist()
    cases = []
    for obs_name, obs in [("full_field", dense_obs), ("sparse_16pts", sparse_obs)]:
        for sigma_frac in [0.01, 0.05, 0.2, 0.5]:
            post = build_posterior(plate, obs, s_true, sigma_frac)
            E_ising, Z = ising_energies(post["J"], post["h"], R)
            E_direct = exact_energies_direct(post, R)
            shift = E_direct - E_ising
            mapping_err = float(np.ptp(shift) / max(np.ptp(E_direct), 1e-300))
            E = E_direct - E_direct.min()
            pi = np.exp(-E)
            pi /= pi.sum()
            S = all_configs(R)
            truth_k = int(np.argmin(np.abs(S - s_true).sum(1)))
            map_k = int(np.argmin(E))
            gaps = {
                "local": spectral_gap(E, local_proposal(Z)),
                "uniform": spectral_gap(E, uniform_proposal(len(E))),
                "quantum": spectral_gap(E, quantum_proposal(E, R, post["J"], post["h"])),
            }
            cases.append({
                "obs": obs_name,
                "sigma_frac": sigma_frac,
                "ising_mapping_rel_error": mapping_err,
                "MAP_equals_truth": map_k == truth_k,
                "posterior_mass_on_truth": float(pi[truth_k]),
                "posterior_entropy_bits": float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum()),
                "spectral_gaps": gaps,
                "coupling_stats": coupling_stats(post["J"], plate.centroids),
                "cond(A^T A)": float(np.linalg.cond(post["A"].T @ post["A"])),
            })
            print(obs_name, sigma_frac, gaps, flush=True)
    report["posterior_cases"] = cases

    # reference: a random dense spin glass of the same size
    Jr = RNG.standard_normal((R, R))
    Jr = np.triu(Jr, 1)
    Jr = Jr + Jr.T
    hr = RNG.standard_normal(R)
    ref = []
    for T in [1.0, 0.5, 0.25]:
        Er, Zr = ising_energies(Jr / T, hr / T, R)
        Er = Er - Er.min()
        ref.append({"T": T, "gaps": {
            "local": spectral_gap(Er, local_proposal(Zr)),
            "uniform": spectral_gap(Er, uniform_proposal(len(Er))),
            "quantum": spectral_gap(Er, quantum_proposal(Er, R, Jr / T, hr / T)),
        }})
    report["reference_random_spin_glass"] = ref
    report["runtime_s"] = time.time() - t0
    with open(sys.argv[1] if len(sys.argv) > 1 else "warpage_ising_results.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
