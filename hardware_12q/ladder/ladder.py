"""
The 12-qubit circuit at 1, 2 and 3 Trotter steps, scored by XEB.
Usage: python ladder.py sims 1000 | ibm <backend> <shots> | add <device> <circuit|all> <file> | analyze
"""
import glob
import json
import os
import sys

import numpy as np
from qiskit import qasm2, transpile

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "ladder_spec.json")))
NAMES = sorted(SPEC["circuits"], key=lambda n: SPEC["circuits"][n]["steps"])
R, D = 12, 4096


def circuits():
    out = []
    for n in NAMES:
        qc = qasm2.load(os.path.join(HERE, f"{n}.qasm"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        qc.name = n
        out.append(qc)
    return out


def xeb(counts, name, big_endian=False):
    ideal = np.array(SPEC["circuits"][name]["ideal"])
    den = SPEC["circuits"][name]["xeb_den"]
    idx = []
    for b, k in counts.items():
        s = b.replace(" ", "")[-R:]
        idx += [int(s[::-1] if big_endian else s, 2)] * int(k)
    v = ideal[np.array(idx)]
    return float((v.mean() - 1 / D) / den), float(v.std() / np.sqrt(len(v)) / den), len(v)


def record(device, counts_by_circuit, note=""):
    path = os.path.join(HERE, f"ladder_counts_{device}.json")
    old = json.load(open(path)) if os.path.exists(path) else dict(device=device, counts={}, note=note)
    old["counts"].update(counts_by_circuit)
    json.dump(old, open(path, "w"))
    print("saved", path)


def sims(shots):
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston, FakeMarrakesh
    qcs = circuits()
    ideal = AerSimulator(seed_simulator=1)
    res = ideal.run(transpile(qcs, ideal), shots=shots).result()
    record("sim_ideal", {q.name: res.get_counts(i) for i, q in enumerate(qcs)})
    for Fake in (FakeKingston, FakeFez, FakeMarrakesh):
        be = Fake()
        sim = AerSimulator.from_backend(be, seed_simulator=1)
        t = transpile(qcs, be, optimization_level=3, seed_transpiler=7)
        res = sim.run(t, shots=shots).result()
        record(f"sim_{be.name}", {q.name: res.get_counts(i) for i, q in enumerate(qcs)}, "IBM noise model")


def ibm(backend_name, shots):
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, "..", ".env"))
    except ImportError:
        pass
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"],
                               instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    be = svc.backend(backend_name)
    qcs = transpile(circuits(), be, optimization_level=3, seed_transpiler=7)
    for q in qcs:
        print(q.name, "two-qubit gates:", {k: v for k, v in q.count_ops().items() if k in ("cz", "ecr", "cx")})
    sampler = SamplerV2(mode=be)
    sampler.options.dynamical_decoupling.enable = True
    job = sampler.run(qcs, shots=shots)
    print("job id:", job.job_id())
    res = job.result()
    out = {}
    for q, r in zip(qcs, res):
        out[q.name] = r.data[list(r.data.keys())[0]].get_counts()
    record(backend_name, out, f"IBM hardware, job {job.job_id()}")


def add(device, circuit, path):
    text = open(path).read()
    if "RESULT_JSON_START" in text:
        d = json.loads(text.split("RESULT_JSON_START")[1].split("RESULT_JSON_END")[0])
    else:
        d = json.loads(text)
    if circuit == "all":
        record(device, d["counts"])
        return
    counts = (d.get("resultData", {}).get("measurementCounts") or d.get("measurementCounts") or d.get("counts") or d)
    if circuit in counts:
        counts = counts[circuit]
    record(device, {circuit: counts})


def analyze():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "ladder_counts_*.json"))):
        d = json.load(open(f))
        for n in NAMES:
            if n not in d["counts"]:
                continue
            a, b = xeb(d["counts"][n], n), xeb(d["counts"][n], n, big_endian=True)
            best, order = (a, "") if a[0] >= b[0] else (b, " (bits reversed)")
            rows.append((d["device"] + order, SPEC["circuits"][n]["zz"], *best))
    print(f"{'device':45s} {'ZZ':>4s} {'XEB fidelity':>15s} {'shots':>6s}")
    for r in rows:
        print(f"{r[0]:45s} {r[1]:4d} {r[2]:8.2f} ± {r[3]:.2f} {r[4]:6d}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for dev in dict.fromkeys(r[0] for r in rows):
        pts = sorted((r[1], r[2], r[3]) for r in rows if r[0] == dev)
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=dev)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("ZZ interactions in the circuit (1, 2, 3 Trotter steps)")
    ax.set_ylabel("XEB fidelity vs exact ideal")
    ax.set_title("Depth ladder: XEB fidelity vs circuit depth")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(HERE, "ladder_results.png"), dpi=150, bbox_inches="tight")
    print("figure: ladder_results.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sims":
        sims(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    elif cmd == "ibm":
        ibm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif cmd == "add":
        add(sys.argv[2], sys.argv[3], sys.argv[4])
    analyze() if cmd in ("analyze", "sims", "add") else None
