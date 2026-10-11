"""
9-qubit hardware test of the proposal circuit.
Usage: python warpage_hw.py prepare | ibm --backend <name> --shots N | fake | analyze <files>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

SPEC = "experiment_spec.json"
SETTINGS = [
    ("A", 0.45, 2.0, 2),
    ("B", 0.45, 3.0, 3),
]


def build_circuit(name, z, J, h, alpha, gamma, t, steps):
    R = len(z)
    dt = t / steps
    qc = QuantumCircuit(R, R, name=name)
    for i in range(R):
        if z[i] < 0:
            qc.x(i)
    for _ in range(steps):
        for i in range(R):
            for j in range(i + 1, R):
                qc.rzz(2 * dt * (1 - gamma) * alpha * J[i][j], i, j)
        for i in range(R):
            qc.rz(-2 * dt * (1 - gamma) * alpha * h[i], i)
        for i in range(R):
            qc.rx(2 * dt * gamma, i)
    qc.measure(range(R), range(R))
    return qc


def load_circuits(spec):
    out = []
    for c in spec["circuits"]:
        out.append(build_circuit(c["name"], c["z"], spec["J"], spec["h"], spec["alpha"],
                                 c["gamma"], c["t"], c["steps"]))
    return out


def prepare():
    import hw_design as H
    from qiskit.quantum_info import Statevector

    inst = H.instance()
    R, E, Z = inst["R"], inst["E"], inst["Z"]
    M = 2 ** R
    Zindex = {tuple(int(v) for v in z): k for k, z in enumerate(Z)}
    mins = H.local_minima(E, Z)
    gmin = int(np.argmin(E))
    false_mins = [k for k in mins if k != gmin]
    starts = sorted(false_mins, key=lambda k: -np.sum(E < E[k]))[:3]
    alpha = float(H.alpha_of(inst["J"], inst["h"], R))
    circuits = []
    os.makedirs("qasm", exist_ok=True)
    for label, g, t, st in SETTINGS:
        for k in starts:
            z = [int(v) for v in Z[k]]
            name = f"{label}_start{k}"
            qc = build_circuit(name, z, inst["J"].tolist(), inst["h"].tolist(), alpha, g, t, st)
            p = np.zeros(M)
            for bits, v in Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict().items():
                p[H.bitstring_to_index(bits, Zindex)] += v
            circuits.append(dict(name=name, setting=label, start_index=int(k), z=z,
                                 gamma=g, t=t, steps=st,
                                 ideal_p_down=float(p[E < E[k]].sum()),
                                 uniform_p_down=float(np.sum(E < E[k]) / M),
                                 ideal_distribution=p.tolist()))
            with open(f"qasm/{name}.qasm", "w") as fh:
                fh.write(qasm2.dumps(qc))
    spec = dict(R=R, J=inst["J"].tolist(), h=inst["h"].tolist(), alpha=alpha,
                energies=E.tolist(), Z=Z.astype(int).tolist(),
                ground_index=gmin, n_local_minima=len(mins), circuits=circuits)
    json.dump(spec, open(SPEC, "w"))
    print(f"wrote {SPEC} and {len(circuits)} qasm files")


def save_counts(backend_name, counts, extra=None):
    fn = f"counts_{backend_name.replace('.', '-')}_{int(time.time())}.json"
    json.dump(dict(backend=backend_name, counts=counts, **(extra or {})), open(fn, "w"), indent=1)
    print("saved", fn)
    return fn


def run_ibm(backend_name, shots):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    else:
        service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)
    spec = json.load(open(SPEC))
    circs = transpile(load_circuits(spec), backend=backend, optimization_level=3, seed_transpiler=7)
    for c in circs:
        print(c.name, "two-qubit gates:", sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")),
              "depth:", c.depth())
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = True
    job = sampler.run(circs, shots=shots)
    print("job id:", job.job_id(), "- waiting for results")
    res = job.result()
    counts = {c.name: r.data.c.get_counts() for c, r in zip(circs, res)}
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    save_counts(backend_name, counts, {"job_id": job.job_id(), "usage_seconds": usage, "shots": shots})


def retrieve_ibm(job_id):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    else:
        service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    print("status:", job.status())
    res = job.result()
    spec = json.load(open(SPEC))
    names = [c["name"] for c in spec["circuits"]]
    counts = {n: r.data.c.get_counts() for n, r in zip(names, res)}
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    save_counts(job.backend().name, counts, {"job_id": job_id, "usage_seconds": usage})


FORTE_1Q, FORTE_2Q = 0.0001645, 0.001121
FORTE_MIN_DEBIAS, FORTE_MIN_NO_DEBIAS = 168.195, 25.7899


def select(circs, settings):
    keep = set(settings.split(",")) if settings else None
    return [c for c in circs if keep is None or c.name.split("_")[0] in keep or c.name in keep]


def gate_counts(qc):
    ops = qc.count_ops()
    two = sum(v for k, v in ops.items() if k in ("rzz", "cx", "cz", "zz", "ms"))
    one = sum(v for k, v in ops.items() if k not in ("rzz", "cx", "cz", "zz", "ms", "measure", "barrier"))
    return one, two


def offline_cost(settings, shots, debias):
    spec = json.load(open(SPEC))
    total = 0.0
    print(f"{'circuit':<14}{'1q':>5}{'2q':>5}{'usage $':>10}{'billed $':>10}")
    for c in select(load_circuits(spec), settings):
        one, two = gate_counts(c)
        usage = shots * (one * FORTE_1Q + two * FORTE_2Q)
        billed = max(usage, FORTE_MIN_DEBIAS if debias else FORTE_MIN_NO_DEBIAS)
        total += billed
        print(f"{c.name:<14}{one:>5}{two:>5}{usage:>10.2f}{billed:>10.2f}")
    print(f"total (list price, {shots} shots, debiasing {'on' if debias else 'off'}): ${total:,.2f}")


def _ionq_backend(backend_name, noise):
    from qiskit_ionq import IonQProvider

    backend = IonQProvider(os.environ.get("IONQ_API_KEY")).get_backend(backend_name)
    if noise:
        backend.set_options(noise_model=noise)
    return backend


def _mitigation(debias):
    from qiskit_ionq import ErrorMitigation
    return ErrorMitigation.DEBIASING if debias else ErrorMitigation.NO_DEBIASING


def ionq_estimate(backend_name, shots, settings, debias):
    import requests

    spec = json.load(open(SPEC))
    total = 0.0
    for c in select(load_circuits(spec), settings):
        one, two = gate_counts(c)
        params = {"backend": backend_name, "qubits": c.num_qubits, "shots": shots,
                  "1q_gates": one, "2q_gates": two, "error_mitigation": str(debias).lower()}
        r = requests.get("https://api.ionq.co/v0.4/jobs/estimate", params=params,
                         headers={"Authorization": f"apiKey {os.environ.get('IONQ_API_KEY')}"}, timeout=60)
        if r.status_code != 200:
            print(f"{c.name}: estimate failed ({r.status_code}): {r.text[:300]}")
            continue
        data = r.json()
        val = float(data.get("estimated_total_cost", 0.0))
        total += val
        rates = (data.get("rate_card") or {}).get("rates") or [{}]
        print(f"{c.name:<14} 1q={one} 2q={two} shots={shots}: {val:,.2f} {data.get('estimated_unit', '')}"
              f"  (2q rate {rates[0].get('cost_2q_gate')}, minimum {rates[0].get('job_cost_minimum')}, "
              f"queue ~{(data.get('current_predicted_queue_time') or 0) / 60000:.0f} min)")
    print(f"total estimate: {total:,.2f} - nothing was submitted")


def run_ionq(backend_name, shots, noise, settings, debias):
    backend = _ionq_backend(backend_name, noise)
    spec = json.load(open(SPEC))
    circs = select(load_circuits(spec), settings)
    counts = {}
    for c in circs:
        kw = {} if backend_name.startswith("simulator") else {"error_mitigation": _mitigation(debias)}
        job = backend.run(c, shots=shots, **kw)
        print(c.name, "job id:", job.job_id())
        counts[c.name] = job.get_counts()
    save_counts(backend_name + (f"-{noise}" if noise else ""), counts, {"shots": shots, "debias": debias})


def run_fake(backend_name, shots):
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    fake = FakeProviderForBackendV2().backend(backend_name)
    sim = AerSimulator.from_backend(fake)
    spec = json.load(open(SPEC))
    circs = transpile(load_circuits(spec), backend=fake, optimization_level=3, seed_transpiler=7)
    for c in circs:
        print(c.name, "two-qubit gates:", sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")),
              "depth:", c.depth())
    res = sim.run(circs, shots=shots).result()
    save_counts(backend_name, {c.name: res.get_counts(i) for i, c in enumerate(circs)}, {"shots": shots})


def analyze(files):
    spec = json.load(open(SPEC))
    R = spec["R"]
    E = np.array(spec["energies"])
    Zindex = {tuple(z): k for k, z in enumerate(spec["Z"])}

    def idx(bits):
        bits = bits.replace(" ", "")[-R:]
        return Zindex[tuple(+1 if bits[R - 1 - i] == "0" else -1 for i in range(R))]

    print(f"{'circuit':<16}{'device':<26}{'P_down':>8}{'ideal':>8}{'uniform':>9}{'TVD vs ideal':>14}")
    summary = {}
    for fn in files:
        data = json.load(open(fn))
        dev = data["backend"]
        for c in spec["circuits"]:
            cnt = data["counts"].get(c["name"])
            if not cnt:
                continue
            p = np.zeros(2 ** R)
            for bits, n in cnt.items():
                p[idx(bits)] += n
            p /= p.sum()
            k = c["start_index"]
            pd = float(p[E < E[k]].sum())
            tvd = 0.5 * float(np.abs(p - np.array(c["ideal_distribution"])).sum())
            print(f"{c['name']:<16}{dev:<26}{pd:>8.3f}{c['ideal_p_down']:>8.3f}{c['uniform_p_down']:>9.3f}{tvd:>14.3f}")
            summary.setdefault(dev, {}).setdefault(c["setting"], []).append(
                dict(p_down=pd, ideal=c["ideal_p_down"], uniform=c["uniform_p_down"], tvd=tvd))
    print("\nper device and setting (mean over start states):")
    for dev, by in summary.items():
        for s, rows in sorted(by.items()):
            pd = np.mean([r["p_down"] for r in rows])
            idl = np.mean([r["ideal"] for r in rows])
            un = np.mean([r["uniform"] for r in rows])
            kept = (pd - un) / (idl - un) if idl > un else float("nan")
            print(f"  {dev:<26} setting {s}: P_down {pd:.3f} (ideal {idl:.3f}, uniform floor {un:.3f}) "
                  f"-> quantum signal kept: {kept:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "ibm", "ibm-retrieve", "ionq", "ionq-estimate", "cost", "fake", "analyze"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=None, help="default 2000 (IBM/fake), 1000 (IonQ)")
    ap.add_argument("--noise", default=None, help="IonQ simulator noise model, e.g. forte-1")
    ap.add_argument("--settings", default=None, help="IonQ: which settings to run, e.g. A or A,B (default all)")
    ap.add_argument("--no-debias", action="store_true", help="IonQ QPU: turn debiasing off (lower minimum cost)")
    a = ap.parse_args()
    debias = not a.no_debias
    if a.mode == "prepare":
        prepare()
    elif a.mode == "ibm":
        run_ibm(a.backend or "ibm_kingston", a.shots or 2000)
    elif a.mode == "ibm-retrieve":
        retrieve_ibm(a.files[0])
    elif a.mode == "cost":
        offline_cost(a.settings, a.shots or 1000, debias)
    elif a.mode == "ionq-estimate":
        ionq_estimate(a.backend or "qpu.forte-1", a.shots or 1000, a.settings, debias)
    elif a.mode == "ionq":
        run_ionq(a.backend or "qpu.forte-1", a.shots or 1000, a.noise, a.settings, debias)
    elif a.mode == "fake":
        run_fake(a.backend or "fake_kingston", a.shots or 2000)
    else:
        analyze(a.files)


if __name__ == "__main__":
    main()
