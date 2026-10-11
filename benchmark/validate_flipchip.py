"""
3D FEA of the flip-chip package in Tsai, Wang & Liu, Materials 14:3723 (2021),
compared with their shadow-moire data (Fig. 13). Materials from their Table 1;
geometry from the Cadence thermal-warpage white paper.
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import Basis, BilinearForm, ElementHex2, ElementVector, LinearForm, MeshHex, asm, condense, solve
from skfem.helpers import ddot, div, sym_grad, trace

L, DIE = 31.0, 10.0
TS, TU, TD = 1.08, 0.07, 0.785
a0, a1 = (L - DIE) / 2, (L + DIE) / 2
xs = np.unique(np.concatenate([np.linspace(0, a0, 6), np.linspace(a0, a1, 6), np.linspace(a1, L, 6)]))
zs = np.unique(np.concatenate([np.linspace(0, TS, 3), [TS + TU], np.linspace(TS + TU, TS + TU + TD, 3)]))
mesh = MeshHex.init_tensor(xs, xs, zs)
basis = Basis(mesh, ElementVector(ElementHex2()))
c = mesh.p[:, mesh.t].mean(axis=1)
inside = (c[0] > a0) & (c[0] < a1) & (c[1] > a0) & (c[1] < a1)
MAT = {"sub": (26e3, 0.39, 14.4e-6), "uf": (7.6e3, 0.32, 29e-6), "die": (131e3, 0.28, 2.8e-6), "void": (1e-3, 0.0, 0.0)}
mat = np.where(c[2] < TS, "sub", np.where(~inside, "void", np.where(c[2] < TS + TU, "uf", "die")))
E = np.array([MAT[m][0] for m in mat]); nu = np.array([MAT[m][1] for m in mat]); al = np.array([MAT[m][2] for m in mat])
nq = basis.X.shape[1]
rep = lambda q: np.repeat(q[:, None], nq, axis=1)
DT = 1.0


@BilinearForm
def k(u, v, w):
    return w["lam"] * trace(sym_grad(u)) * trace(sym_grad(v)) + 2 * w["mu"] * ddot(sym_grad(u), sym_grad(v))


@LinearForm
def f(v, w):
    return (3 * w["lam"] + 2 * w["mu"]) * w["a"] * DT * div(v)


p = basis.doflocs
comp = np.empty(basis.N, int)
for attr in ("nodal_dofs", "edge_dofs", "facet_dofs", "interior_dofs"):
    for kk, d in enumerate(getattr(basis, attr, [])):
        comp[d] = kk
node = lambda x, y, z: np.where((abs(p[0] - x) < 1e-9) & (abs(p[1] - y) < 1e-9) & (abs(p[2] - z) < 1e-9))[0]
n0, nx, ny = node(0, 0, 0), node(L, 0, 0), node(0, L, 0)
D = np.array(sorted(list(n0) + [d for d in nx if comp[d] in (1, 2)] + [d for d in ny if comp[d] == 2]))
diag = [(t, t) for t in np.unique(xs) if t >= L / 2]


# K and the thermal load are linear in (lambda, mu, alpha) per material set: assemble once
unit_l = BilinearForm(lambda u, v, w: trace(sym_grad(u)) * trace(sym_grad(v)))
unit_m = BilinearForm(lambda u, v, w: 2.0 * ddot(sym_grad(u), sym_grad(v)))
unit_f = LinearForm(lambda v, w: div(v))
BLOCKS = {}
for name in ("sub", "uf", "die", "void"):
    b = Basis(mesh, ElementVector(ElementHex2()), elements=np.where(mat == name)[0])
    BLOCKS[name] = (asm(unit_l, b).tocsr(), asm(unit_m, b).tocsr(), asm(unit_f, b))


def unit_response(uf):
    MAT["uf"] = uf
    K, Fv = 0, 0
    for name, (KL, KM, FV) in BLOCKS.items():
        E_, nu_, a_ = MAT[name]
        lam_, mu_ = E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_)), E_ / (2 * (1 + nu_))
        K = K + lam_ * KL + mu_ * KM
        Fv = Fv + (3 * lam_ + 2 * mu_) * a_ * DT * FV
    u = solve(*condense(K, Fv, D=D), solver=lambda A, b: spla.spsolve(A.tocsc(), b))
    uz = lambda x, y: u[[d for d in node(x, y, 0.0) if comp[d] == 2][0]]
    centre = uz(L / 2, L / 2)
    corners = np.mean([uz(L, L), uz(0, 0), uz(L, 0), uz(0, L)]) - centre
    prof = [(np.sqrt(2) * (x - L / 2), uz(x, y) - centre) for x, y in diag]
    return corners * 1e3, prof  # um per K


below, prof_b = unit_response((7.6e3, 0.32, 29e-6))
above, prof_a = unit_response((0.1e3, 0.32, 98e-6))  # underfill above Tg: Tsai et al. 2021, Table 1 (100 MPa, 98 ppm/K)
T = [25, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
moire_heat = [-58, -50, -38, -28, -18, 0, 12, 12, 14, 17, 19, 22, 27]
fem_paper = [-58, -49, -37, -24, -12, 0, 4, 8, 12, 16, 20, 24, 28]
ours = [below * (t - 120) if t <= 120 else above * (t - 120) for t in T]
print("T(C)  ours  paperFEM  moire(heating)   [um, corner c relative to centre; paper values read off Fig.13]")
for t, a, b2, c2 in zip(T, ours, fem_paper, moire_heat):
    print(f"{t:4d} {a:6.1f} {b2:6d} {c2:6d}")
o, m = np.array(ours), np.array(moire_heat)
print("RMS ours vs moire (heating):", round(float(np.sqrt(np.mean((o - m) ** 2))), 1), "um over a", int(m.max() - m.min()), "um range")
print("diagonal profile at 25 C (x mm, ours um):", [(round(x, 1), round(w * 1e3 * -95, 1)) for x, w in prof_b])
print("diagonal profile at 260 C (x mm, ours um):", [(round(x, 1), round(w * 1e3 * 140, 1)) for x, w in prof_a])
