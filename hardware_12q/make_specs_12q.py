"""
Rebuild spec_12q.json, ladder/ladder_spec.json and the QASM files from ../main.py
and compare with the saved copies.  Usage: python make_specs_12q.py [--write]
"""
import json
import os
import sys

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import main as M

START = 743
R, D = M.R, 2 ** M.R


def ideal_distribution(steps, t):
    M.TROTTER_STEPS, M.T_EVOL = steps, t
    qc = M.proposal_circuit(START, cal["J"], cal["h"], measure=False)
    return qc, Statevector(qc).probabilities()   # index k: bit i of k = qubit i = region i


def report(name, new, path, key):
    old = np.array(json.load(open(path))[key] if key else json.load(open(path)))
    print(f"{name:34s} max |new - saved| = {np.abs(np.array(new) - old).max():.2e}")


cal = M.calibrate()
E, p = cal["E"], cal["p"]
lower = E < E[START]
qc3, ideal3 = ideal_distribution(3, 3.0)
spec = dict(name="warpcal_12q", start=START, gamma=M.GAMMA, t=3.0, steps=3, E=E.tolist(),
            p_posterior=p.tolist(), J=cal["J"].tolist(), h=cal["h"].tolist(), ideal=ideal3.tolist(),
            lower_mask=lower.tolist(), s_true=list(M.S_TRUE),
            bit_order="Qiskit little-endian: rightmost bit = qubit 0 = region 0; 1 = copper-rich")
saved = json.load(open(os.path.join(HERE, "spec_12q.json")))
for k in ("E", "p_posterior", "ideal"):
    print(f"spec_12q.{k:24s} max |new - saved| = {np.abs(np.array(spec[k]) - np.array(saved[k])).max():.2e}")
print(f"spec_12q.lower_mask identical: {spec['lower_mask'] == saved['lower_mask']};  "
      f"ideal P_better {ideal3[lower].sum():.3f}, floor {lower.mean():.3f}")

ladder = dict(start=START, circuits={})
qasms = {}
for steps in (1, 2, 3):
    qc, ideal = ideal_distribution(steps, float(steps))
    name = f"ladder_12q_{steps}step"
    ladder["circuits"][name] = dict(steps=steps, zz=qc.count_ops().get("rzz", 0), ideal=ideal.tolist(),
                                    xeb_den=float((ideal ** 2).sum() - 1 / D))
    qm = qc.copy()
    qm.measure_all(add_bits=False) if qm.num_clbits else None
    qasms[name] = qc
saved_l = json.load(open(os.path.join(HERE, "ladder", "ladder_spec.json")))
for n, c in ladder["circuits"].items():
    print(f"{n + '.ideal':34s} max |new - saved| = "
          f"{np.abs(np.array(c['ideal']) - np.array(saved_l['circuits'][n]['ideal'])).max():.2e}  ({c['zz']} ZZ)")

if "--write" in sys.argv:
    json.dump(spec, open(os.path.join(HERE, "spec_12q.json"), "w"))
    json.dump(ladder, open(os.path.join(HERE, "ladder", "ladder_spec.json"), "w"))
    M.TROTTER_STEPS, M.T_EVOL = 3, 3.0
    open(os.path.join(HERE, "warpcal_12q_rzz.qasm"), "w").write(qasm2.dumps(M.proposal_circuit(START, cal["J"], cal["h"])))
    for steps in (1, 2, 3):
        M.TROTTER_STEPS, M.T_EVOL = steps, float(steps)
        open(os.path.join(HERE, "ladder", f"ladder_12q_{steps}step.qasm"), "w").write(
            qasm2.dumps(M.proposal_circuit(START, cal["J"], cal["h"])))
    print("files written")
