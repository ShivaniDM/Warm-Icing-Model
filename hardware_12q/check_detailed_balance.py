"""
Check that the 12-qubit proposal is symmetric, Q(x->y) = Q(y->x), and compare
the Metropolis stationary distribution with the exact posterior; repeated with
an example 3% 1->0 readout bias.  Usage: python check_detailed_balance.py
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
import main as M
from qiskit.quantum_info import Operator
cal = M.calibrate(); E = cal["E"]; pi = cal["p"]
qc = M.proposal_circuit(0, cal["J"], cal["h"], measure=False)
U = Operator(qc).data            # index = Qiskit little-endian = map index
Q = np.abs(U) ** 2
Q = Q.T
print("row sums ok:", np.allclose(Q.sum(1), 1))
asym = np.abs(Q - Q.T)
print(f"Trotter circuit proposal: max |Q(x->y) - Q(y->x)| = {asym.max():.2e}, mean rel asym on moves with Q>1e-4: "
      f"{np.median((asym / np.maximum(Q, Q.T))[np.maximum(Q, Q.T) > 1e-4]):.3f}")
def stationary(P):
    s = pi.copy()
    for _ in range(4000):
        s = s @ P
    return s / s.sum()
def chain(Q, mh):
    ratio = np.exp(np.clip(E[:, None] - E[None, :], -700, 50))
    if mh: ratio = ratio * np.where(Q > 0, Q.T / np.where(Q > 0, Q, 1), 0)
    P = Q * np.minimum(1, ratio); np.fill_diagonal(P, 0); np.fill_diagonal(P, 1 - P.sum(1)); return P
for name, mh in [("plain Metropolis (assumes symmetric)", False), ("Metropolis-Hastings (q ratio)", True)]:
    s = stationary(chain(Q, mh)); print(f"{name:40s} TVD(stationary, posterior) = {0.5*np.abs(s-pi).sum():.4f}")
Qs = 0.5 * (Q + Q.T)
s = stationary(chain(Qs, False)); print(f"{'symmetrised proposal, plain Metropolis':40s} TVD = {0.5*np.abs(s-pi).sum():.4f}")
p1 = 0.03
S = ((np.arange(4096)[:, None] >> np.arange(12)) & 1)
R = np.ones((4096, 4096))
for i in range(12):
    a, b = S[:, i][:, None], S[:, i][None, :]
    R *= np.where((a == 1) & (b == 0), p1, np.where((a == 1) & (b == 1), 1 - p1, np.where((a == 0) & (b == 0), 1.0, 0.0)))
Qb = Qs @ R
s = stationary(chain(Qb, False)); print(f"{'with 3% 1->0 readout bias, plain':40s} TVD = {0.5*np.abs(s-pi).sum():.4f}")
