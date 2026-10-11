"""Helpers for the 12-qubit hardware test. Bit order: Qiskit little-endian, bit i = region i."""
import json
import os

import numpy as np
from qiskit import qasm2

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "spec_12q.json")))
R = 12


def circuit():
    qc = qasm2.load(os.path.join(HERE, "warpcal_12q_rzz.qasm"),
                    custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    qc.name = SPEC["name"]
    return qc


def counts_to_index(bits, big_endian=False):
    b = bits.replace(" ", "")[-R:]
    if big_endian:  # some providers print qubit 0 first
        b = b[::-1]
    return int(b, 2)


def score(counts, big_endian=False):
    lower = np.array(SPEC["lower_mask"], bool)
    ideal = np.array(SPEC["ideal"])
    total = sum(counts.values())
    p = np.zeros(2 ** R)
    for bits, n in counts.items():
        p[counts_to_index(bits, big_endian)] += n
    p /= total
    pd = float(p[lower].sum())
    pi, pu = float(ideal[lower].sum()), float(lower.mean())
    err = np.sqrt(pd * (1 - pd) / total)
    top = np.argsort(-ideal)[:50]
    D = 2 ** R
    den = float((ideal ** 2).sum() - 1 / D)
    xs = np.repeat([counts_to_index(b, big_endian) for b in counts], list(counts.values()))
    xeb = float((ideal[xs].mean() - 1 / D) / den)
    xeb_err = float(ideal[xs].std() / np.sqrt(total) / den)
    return dict(xeb_fidelity=xeb, xeb_err=xeb_err, shots=total, p_better=pd, p_better_err=float(err), ideal=pi, floor=pu,
                signal_kept=(pd - pu) / (pi - pu), signal_kept_err=float(err / (pi - pu)),
                mass_on_ideal_top50=float(p[top].sum()), ideal_top50=float(ideal[top].sum()),
                uniform_top50=50 / 2 ** R, tvd_vs_ideal=float(0.5 * np.abs(p - ideal).sum()))


def save(name, counts, extra=None):
    out = dict(device=name, counts=counts, **(extra or {}))
    path = os.path.join(HERE, f"counts_{name}.json")
    json.dump(out, open(path, "w"))
    print("saved", path)
