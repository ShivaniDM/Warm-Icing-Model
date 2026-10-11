"""
3D linear thermoelastic FEA of a two-layer substrate (scikit-fem, 27-node hexahedra,
free body with 3-2-1 constraints). Output: top-surface warpage on a 25 x 25 grid,
best-fit plane removed. Module constants are placeholders; build_benchmark.configure()
sets the benchmark values.
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import Basis, BilinearForm, ElementHex2, ElementVector, LinearForm, MeshHex, asm, condense, solve
from skfem.helpers import ddot, div, sym_grad, trace

L = 15.0                 # mm
T_CORE, T_BU = 0.40, 0.05  # mm (core, top build-up)
NXY = 12
DT = -200.0              # K, cool from 220 C stress-free to 20 C

# E [MPa], nu, CTE [1/K]
CORE = dict(E=25e3, nu=0.20, a=12e-6)
BU_NOMINAL = dict(E=6e3, nu=0.30, a=40e-6)
BU_CU_RICH = dict(E=15e3, nu=0.32, a=18e-6)
REGIONS = (3, 3)


def build_mesh():
    xy = np.linspace(0.0, L, NXY + 1)
    z = np.concatenate([np.linspace(0.0, T_CORE, 3), np.linspace(T_CORE, T_CORE + T_BU, 3)[1:]])
    return MeshHex.init_tensor(xy, xy, z)


def element_materials(mesh, s):
    c = mesh.p[:, mesh.t].mean(axis=1)
    rx, ry = REGIONS
    E = np.empty(mesh.nelements)
    nu = np.empty(mesh.nelements)
    a = np.empty(mesh.nelements)
    for e in range(mesh.nelements):
        x, y, zc = c[:, e]
        if zc < T_CORE:
            m = CORE
        else:
            i = min(int(x / L * rx), rx - 1)
            j = min(int(y / L * ry), ry - 1)
            m = BU_CU_RICH if s[i * ry + j] else BU_NOMINAL
        E[e], nu[e], a[e] = m["E"], m["nu"], m["a"]
    return E, nu, a


@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return w["lam"] * trace(eu) * trace(ev) + 2.0 * w["mu"] * ddot(eu, ev)


@LinearForm
def thermal_load(v, w):
    return (3.0 * w["lam"] + 2.0 * w["mu"]) * w["a"] * DT * div(v)


class Substrate:
    def __init__(self):
        self.mesh = build_mesh()
        self.basis = Basis(self.mesh, ElementVector(ElementHex2()))
        self.nq = self.basis.X.shape[1]
        p = self.basis.doflocs
        ztop = T_CORE + T_BU
        tol = 1e-9
        # 3-2-1 constraints on bottom corners
        def node(x, y, z):
            return np.where((abs(p[0] - x) < tol) & (abs(p[1] - y) < tol) & (abs(p[2] - z) < tol))[0]
        n0, nx, ny = node(0, 0, 0), node(L, 0, 0), node(0, L, 0)
        comp = self._components()
        fix = list(n0) + [d for d in nx if comp[d] in (1, 2)] + [d for d in ny if comp[d] == 2]
        self.D = np.array(sorted(fix))
        g = np.linspace(0.0, L, 25)
        self.obs = np.array([[d for d in node(xx, yy, ztop) if comp[d] == 2][0] for xx in g for yy in g])
        ii, jj = np.meshgrid(g, g, indexing="ij")
        B = np.column_stack([np.ones(ii.size), ii.ravel(), jj.ravel()])
        self.P_plane = np.eye(ii.size) - B @ np.linalg.pinv(B)

    def _components(self):
        comp = np.empty(self.basis.N, dtype=int)
        for k, dofs in enumerate(self.basis.nodal_dofs):
            comp[dofs] = k
        for k, dofs in enumerate(getattr(self.basis, "edge_dofs", [])):
            comp[dofs] = k
        for k, dofs in enumerate(getattr(self.basis, "facet_dofs", [])):
            comp[dofs] = k
        for k, dofs in enumerate(getattr(self.basis, "interior_dofs", [])):
            comp[dofs] = k
        return comp

    def _element_sets(self):
        c = self.mesh.p[:, self.mesh.t].mean(axis=1)
        rx, ry = REGIONS
        reg = np.full(self.mesh.nelements, -1)
        top = c[2] > T_CORE
        i = np.minimum((c[0] / L * rx).astype(int), rx - 1)
        j = np.minimum((c[1] / L * ry).astype(int), ry - 1)
        reg[top] = (i * ry + j)[top]
        return [np.where(reg == r)[0] for r in range(-1, rx * ry)]

    def precompute(self, cache=None):
        import os
        cache = cache or "fea_blocks_%dx%d_L%g_c%g_b%g.npz" % (REGIONS + (L, T_CORE, T_BU))
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            self.KL, self.KM = list(z["KL"]), list(z["KM"])
            self.FV = [np.asarray(v, dtype=float) for v in z["FV"]]
            return
        unit_l = BilinearForm(lambda u, v, w: trace(sym_grad(u)) * trace(sym_grad(v)))
        unit_m = BilinearForm(lambda u, v, w: 2.0 * ddot(sym_grad(u), sym_grad(v)))
        unit_f = LinearForm(lambda v, w: div(v))
        self.KL, self.KM, self.FV = [], [], []
        for els in self._element_sets():
            b = Basis(self.mesh, ElementVector(ElementHex2()), elements=els)
            self.KL.append(asm(unit_l, b).tocsr())
            self.KM.append(asm(unit_m, b).tocsr())
            self.FV.append(asm(unit_f, b))
        np.savez(cache, KL=np.array(self.KL, dtype=object), KM=np.array(self.KM, dtype=object),
                 FV=np.array(self.FV, dtype=object))

    def solve(self, s):
        if not hasattr(self, "KL"):
            self.precompute()
        assert len(s) == REGIONS[0] * REGIONS[1]
        mats = [CORE] + [BU_CU_RICH if v else BU_NOMINAL for v in s]
        K = 0
        f = 0
        for m, KL, KM, FV in zip(mats, self.KL, self.KM, self.FV):
            lam = m["E"] * m["nu"] / ((1 + m["nu"]) * (1 - 2 * m["nu"]))
            mu = m["E"] / (2 * (1 + m["nu"]))
            K = K + lam * KL + mu * KM
            f = f + (3 * lam + 2 * mu) * m["a"] * DT * FV
        u = solve(*condense(K.tocsr(), f, D=self.D),
                  solver=lambda A, b: spla.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A"))
        uz = u[self.obs]
        return self.P_plane @ uz, u


def clt_curvature(top, bottom=CORE, t_top=T_BU, t_bot=T_CORE, dT=DT):
    zs = [(0.0, t_bot, bottom), (t_bot, t_bot + t_top, top)]
    A = B = Dm = Nt = Mt = 0.0
    for z0, z1, m in zs:
        Q = m["E"] / (1 - m["nu"])
        A += Q * (z1 - z0)
        B += Q * (z1 ** 2 - z0 ** 2) / 2
        Dm += Q * (z1 ** 3 - z0 ** 3) / 3
        Nt += Q * m["a"] * dT * (z1 - z0)
        Mt += Q * m["a"] * dT * (z1 ** 2 - z0 ** 2) / 2
    eps0, kappa = np.linalg.solve([[A, B], [B, Dm]], [Nt, Mt])
    return kappa, Dm - B ** 2 / A
