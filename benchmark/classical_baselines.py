"""
Quantum proposals vs single-flip, uniform, block-move and parallel-tempering
Metropolis at an equal number of posterior evaluations (circuit cost not counted).
Writes classical_baselines.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB

R, M = 9, 512
S = ((np.arange(M)[:, None] >> np.arange(R)) & 1).astype(float)
CHAINS = 400
BUDGETS = [100, 300, 1000, 3000]
PT_LADDERS = {"parallel tempering (4 replicas, beta 1-0.2)": np.array([1.0, 0.6, 0.35, 0.2]),
              "parallel tempering (8 replicas, beta 1-0.005)": np.geomspace(1.0, 0.005, 8)}


def metropolis(E, propose, budget, rng):
    x = rng.integers(0, M, CHAINS)
    acc_sum, n = np.zeros((CHAINS, R)), 0
    burn = budget // 10
    for step in range(budget):
        y = propose(x, rng)
        ok = rng.random(CHAINS) < np.exp(np.minimum(0.0, E[x] - E[y]))
        x = np.where(ok, y, x)
        if step >= burn:
            acc_sum += S[x]
            n += 1
    return acc_sum / n


def parallel_tempering(E, budget, rng, BETAS):
    K = len(BETAS)
    x = rng.integers(0, M, (CHAINS, K))
    acc_sum, n = np.zeros((CHAINS, R)), 0
    sweeps = max(budget // K, 1)
    burn = sweeps // 10
    rows = np.arange(CHAINS)
    for step in range(sweeps):
        y = x ^ (1 << rng.integers(0, R, (CHAINS, K)))
        ok = rng.random((CHAINS, K)) < np.exp(np.minimum(0.0, BETAS[None, :] * (E[x] - E[y])))
        x = np.where(ok, y, x)
        i = rng.integers(0, K - 1, CHAINS)
        a, b = x[rows, i], x[rows, i + 1]
        sw = rng.random(CHAINS) < np.exp(np.minimum(0.0, (BETAS[i] - BETAS[i + 1]) * (E[a] - E[b])))
        x[rows[sw], i[sw]], x[rows[sw], i[sw] + 1] = b[sw], a[sw]
        if step >= burn:
            acc_sum += S[x[:, 0]]
            n += 1
    return acc_sum / n


def from_matrix(Q):
    cum = np.cumsum(Q, axis=1)
    cum[:, -1] = 1.0

    def propose(x, rng):
        u = rng.random(len(x))
        return np.minimum((cum[x] < u[:, None]).sum(1), M - 1)
    return propose


def main():
    os.chdir(HERE)
    meas = np.load("measurements.npz")
    fields = np.load("warpage_fields.npz")
    w0, A = RB.surrogate((3, 3))
    rng = np.random.default_rng(5)
    results = {}
    for nf in (0.01, 0.03):
        per_task = []
        for k in range(len(fields["3x3_maps"])):
            y = meas[f"3x3__map{k}__full_field__noise{nf:.2f}"]
            E, p, _ = RB.posterior(S, A, w0, y)
            truth = p @ S
            Qq = RB.quantum_Q(E, R, np.random.default_rng(100 + k))
            U = np.full((M, M), 1.0 / M)
            samplers = {
                "single-flip": lambda x, r: x ^ (1 << r.integers(0, R, len(x))),
                "uniform": lambda x, r: r.integers(0, M, len(x)),
                "block (1-3 flips)": lambda x, r: x ^ np.bitwise_or.reduce(
                    [(r.random(len(x)) < (1.0 if j == 0 else 0.5)) * (1 << r.integers(0, R, len(x))) for j in range(3)], axis=0),
                "quantum ideal": from_matrix(Qq),
                "quantum f=0.84": from_matrix(0.84 * Qq + 0.16 * U),
                "quantum f=0.30": from_matrix(0.30 * Qq + 0.70 * U),
            }
            row = {}
            for B in BUDGETS:
                for name, prop in samplers.items():
                    est = metropolis(E, prop, B, rng)
                    row.setdefault(name, []).append(float(np.sqrt(np.mean((est - truth) ** 2))))
                for name, betas in PT_LADDERS.items():
                    est = parallel_tempering(E, B, rng, betas)
                    row.setdefault(name, []).append(float(np.sqrt(np.mean((est - truth) ** 2))))
            per_task.append(row)
            print(f"noise {nf:.2f} task {k} done", flush=True)
        names = per_task[0].keys()
        results[f"{nf:.2f}"] = {n: [float(np.median([t[n][b] for t in per_task])) for b in range(len(BUDGETS))] for n in names}
    out = dict(budgets_energy_evaluations=BUDGETS, metric="median over 8 tasks of RMS error of region copper probabilities",
               chains=CHAINS, exact_enumeration="512 evaluations, error 0", results=results)
    json.dump(out, open("classical_baselines.json", "w"), indent=1)
    for nf, res in results.items():
        print(f"\nnoise {nf}: RMS error of region probabilities vs budget (energy evaluations)")
        print(f"{'sampler':34s}" + "".join(f"{b:>9d}" for b in BUDGETS))
        for n, v in sorted(res.items(), key=lambda kv: kv[1][-1]):
            print(f"{n:34s}" + "".join(f"{e:9.3f}" for e in v))
    print("\nexact enumeration: 512 evaluations, error 0")


if __name__ == "__main__":
    main()
