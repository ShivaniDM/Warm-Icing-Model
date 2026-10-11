"""Choose the 9-qubit hardware test circuits (settings A/B and start states)."""
import itertools
import json
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import warpage_ising_check as W


def instance(seed=3, sigma_frac=0.05, region_shape=(3, 3)):
    W.RNG = np.random.default_rng(seed)
    plate = W.Plate(n_nodes=25, region_shape=region_shape)
    R = plate.R
    s_true = W.RNG.integers(0, 2, R).astype(float)
    post = W.build_posterior(plate, np.arange(plate.N ** 2), s_true, sigma_frac)
    Joff = post["J"] - np.diag(np.diag(post["J"]))
    E, Z = W.ising_energies(post["J"], post["h"], R)
    return dict(R=R, J=Joff, h=post["h"], E=E - E.min(), Z=Z, s_true=s_true)


def local_minima(E, Z):
    R = Z.shape[1]
    idx = {tuple(z): k for k, z in enumerate(Z)}
    mins = []
    for k, z in enumerate(Z):
        ok = True
        for i in range(R):
            z2 = z.copy()
            z2[i] *= -1
            if E[idx[tuple(z2)]] < E[k]:
                ok = False
                break
        if ok:
            mins.append(k)
    return mins


def alpha_of(J, h, R):
    return np.sqrt(R) / np.sqrt(np.sum(np.triu(J, 1) ** 2) + np.sum(h ** 2))


def proposal_circuit(z, J, h, gamma, t, steps, measure=True):
    R = len(z)
    a = alpha_of(J, h, R)
    dt = t / steps
    qc = QuantumCircuit(R, R if measure else 0)
    for i in range(R):
        if z[i] < 0:
            qc.x(i)
    for _ in range(steps):
        for i in range(R):
            for j in range(i + 1, R):
                qc.rzz(2 * dt * (1 - gamma) * a * J[i, j], i, j)
        for i in range(R):
            qc.rz(-2 * dt * (1 - gamma) * a * h[i], i)
        for i in range(R):
            qc.rx(2 * dt * gamma, i)
    if measure:
        qc.measure(range(R), range(R))
    return qc


def bitstring_to_index(bits_qiskit, Zrows_index):
    R = len(bits_qiskit)
    z = tuple(+1 if bits_qiskit[R - 1 - i] == "0" else -1 for i in range(R))
    return Zrows_index[z]


def ideal_distribution(z, J, h, gamma, t, steps, Zindex, M):
    qc = proposal_circuit(z, J, h, gamma, t, steps, measure=False)
    probs = Statevector(qc).probabilities_dict()
    p = np.zeros(M)
    for bits, v in probs.items():
        p[bitstring_to_index(bits, Zindex)] += v
    return p


def exact_row(z_k, inst, gamma, t):
    R, E = inst["R"], inst["E"]
    M = 2 ** R
    a = alpha_of(inst["J"], inst["h"], R)
    Hmix = np.zeros((M, M))
    for k in range(R):
        flip = np.arange(M) ^ (1 << (R - 1 - k))
        Hmix[np.arange(M), flip] = 1.0
    H = (1 - gamma) * a * np.diag(E - E.mean()) + gamma * Hmix
    vals, vecs = np.linalg.eigh(H)
    U = (vecs * np.exp(-1j * vals * t)) @ vecs.conj().T
    return np.abs(U[:, z_k]) ** 2


def p_down(row, E, k):
    return float(row[E < E[k]].sum())


def main():
    inst = instance()
    R, E, Z = inst["R"], inst["E"], inst["Z"]
    M = 2 ** R
    Zindex = {tuple(int(v) for v in z): k for k, z in enumerate(Z)}
    mins = local_minima(E, Z)
    gmin = int(np.argmin(E))
    false_mins = sorted([k for k in mins if k != gmin], key=lambda k: E[k])
    print("local minima:", len(mins), "false:", len(false_mins))
    starts = sorted(false_mins, key=lambda k: -np.sum(E < E[k]))[:3]
    print("starts:", starts, "energies:", [round(E[k], 1) for k in starts],
          "#lower:", [int(np.sum(E < E[k])) for k in starts])

    rows = []
    for gamma in [0.3, 0.45, 0.6]:
        for t in [1.0, 2.0, 3.0, 4.0]:
            for steps in [1, 2, 3]:
                pd_tr, pd_ex, tvd = [], [], []
                for k in starts:
                    z = tuple(int(v) for v in Z[k])
                    p_tr = ideal_distribution(z, inst["J"], inst["h"], gamma, t, steps, Zindex, M)
                    p_ex = exact_row(k, inst, gamma, t)
                    pd_tr.append(p_down(p_tr, E, k))
                    pd_ex.append(p_down(p_ex, E, k))
                    tvd.append(0.5 * np.abs(p_tr - p_ex).sum())
                unif = [float(np.sum(E < E[k]) / M) for k in starts]
                rows.append(dict(gamma=gamma, t=t, steps=steps,
                                 p_down_trotter=float(np.mean(pd_tr)),
                                 p_down_exact=float(np.mean(pd_ex)),
                                 p_down_uniform=float(np.mean(unif)),
                                 tvd_trotter_vs_exact=float(np.mean(tvd))))
    rows.sort(key=lambda r: -r["p_down_trotter"])
    for r in rows[:12]:
        print(r)
    json.dump(dict(starts=starts, rows=rows), open(sys.argv[1] if len(sys.argv) > 1 else "hw_design.json", "w"), indent=1)


if __name__ == "__main__":
    main()
