"""Ideal and IBM noise-model simulations of the 12-qubit circuit.  Usage: python run_sims.py 300"""
import sys

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston, FakeMarrakesh

from common import circuit, save, score

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 300
qc = circuit()
ideal = AerSimulator(seed_simulator=1)
c = ideal.run(transpile(qc, ideal), shots=shots).result().get_counts()
save(f"sim_ideal_{shots}", c)
print("ideal:", score(c))
for Fake in (FakeKingston, FakeFez, FakeMarrakesh):
    be = Fake()
    sim = AerSimulator.from_backend(be, seed_simulator=1)
    t = transpile(qc, be, optimization_level=3, seed_transpiler=7)
    c = sim.run(t, shots=shots).result().get_counts()
    save(f"sim_{be.name}_{shots}", c)
    print(be.name, {k: v for k, v in t.count_ops().items() if k in ("cz", "ecr", "cx")}, score(c))
