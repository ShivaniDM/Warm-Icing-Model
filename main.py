"""
Warpage calibration with quantum-enhanced MCMC proposals.

Calibrates a 12-region copper map from a 625-point warpage field (thin-plate model,
lamination-theory parameters, exact Bayesian posterior = Ising model), then runs one
Layden et al. (Nature 2023) proposal step from a false local optimum on `backend`,
which the Qollab runner provides. IonQ backends: one circuit, at most 300 shots.
"""
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

L_MM = 25.0
REGIONS = (3, 4)
R = REGIONS[0] * REGIONS[1]
NOISE_UM = 1.1548
# E [MPa], Poisson ratio, CTE [1/K], thickness [mm]; sources in benchmark/benchmark_config.json.
# Copper-rich build-up: 30% copper, Voigt modulus, stiffness-weighted CTE. Stress-free at 120 C,
# measured at 25 C.
CORE = dict(E=26e3, nu=0.39, a=14.4e-6, t=0.8)
BU_NOMINAL = dict(E=7.5e3, nu=0.30, a=23e-6, t=0.1)
_f, _Ecu = 0.30, 117e3
_E = (1 - _f) * 7.5e3 + _f * _Ecu
BU_CU_RICH = dict(E=_E, nu=(1 - _f) * 0.30 + _f * 0.34, a=((1 - _f) * 7.5e3 * 23e-6 + _f * _Ecu * 17e-6) / _E, t=0.1)
DT = 25.0 - 120.0

PROBE = list(range(625))
Y_PROBE = [23.468, 21.481, 19.481, 17.540, 13.844, 10.597, 11.911, 8.500, 6.114, 5.023, 5.351, 2.840, 4.942, 4.795, 7.508, 7.408, 7.224, 5.947, 8.804, 11.343, 14.644, 14.210, 18.387, 19.930, 26.377, 20.180, 16.735, 15.435, 12.029, 10.160, 8.816, 6.400, 4.874, 6.202, 3.423, 1.969, 4.078, 2.720, 3.841, 1.982, 3.181, 4.847, 6.311, 6.540, 11.033, 9.501, 13.744, 15.378, 18.209, 21.485, 20.785, 14.502, 10.557, 11.563, 8.479, 6.597, 5.221, 5.062, 0.748, 0.263, 0.833, -0.794, -0.389, -1.981, -0.784, 1.454, -2.299, 3.116, 1.315, 8.651, 9.305, 12.063, 14.683, 14.622, 18.848, 14.644, 13.494, 9.942, 8.419, 5.072, 3.783, 0.922, -0.261, -1.791, -2.182, -2.060, -1.745, -4.196, -2.146, -3.667, -0.345, -1.122, -0.877, 1.410, 4.252, 5.370, 8.215, 10.174, 15.618, 17.380, 13.846, 11.166, 10.701, 6.071, 3.711, 0.290, 0.822, -2.539, -3.327, -1.502, -5.809, -4.078, -5.964, -7.050, -5.716, -5.014, -3.533, -2.596, -1.217, 0.758, 4.296, 6.227, 6.971, 10.774, 13.340, 12.843, 7.006, 6.706, 4.455, 0.998, -1.164, -1.932, -5.618, -5.970, -7.793, -6.344, -8.468, -7.867, -9.768, -7.672, -5.154, -4.943, -2.683, -5.988, -4.203, 3.014, 3.771, 4.262, 9.655, 12.368, 10.307, 7.563, 5.350, 2.629, 0.025, -1.741, -3.735, -5.465, -6.804, -8.158, -8.321, -8.166, -9.283, -9.171, -6.696, -5.918, -6.941, -6.748, -2.783, -4.515, 1.066, 3.032, 3.550, 6.781, 8.846, 8.550, 4.172, 1.168, 0.549, -2.592, -4.024, -5.915, -7.968, -8.115, -10.398, -9.492, -11.114, -9.247, -11.086, -9.393, -8.497, -8.723, -8.868, -5.132, -5.747, -1.795, -1.732, 2.074, 5.223, 7.802, 7.375, 4.841, 1.670, -0.691, -1.571, -4.249, -6.409, -8.962, -10.884, -11.260, -10.191, -11.540, -11.013, -13.309, -13.509, -10.574, -12.739, -9.008, -9.084, -5.393, -3.001, -2.720, 2.577, 4.492, 7.028, 4.968, 2.090, 0.112, -0.945, -4.292, -8.075, -9.838, -9.046, -10.862, -10.500, -11.575, -11.592, -11.385, -11.621, -11.879, -13.898, -9.906, -8.525, -7.463, -7.776, -3.857, -0.923, 0.923, 2.800, 3.932, 4.459, 4.584, -0.122, -1.491, -5.047, -8.101, -6.942, -9.323, -9.707, -11.932, -14.066, -11.449, -13.658, -12.800, -12.446, -11.173, -10.733, -8.532, -8.827, -5.755, -3.183, -3.075, -1.512, 2.437, 7.132, 5.398, 3.236, -0.614, -5.228, -6.258, -8.788, -11.056, -10.609, -10.209, -11.734, -12.586, -12.911, -14.019, -11.654, -12.249, -11.813, -14.146, -9.322, -7.736, -7.450, -7.359, -2.094, -1.180, 1.779, 4.792, 4.559, 0.677, -0.366, -1.740, -5.440, -8.630, -9.396, -10.617, -11.839, -12.109, -13.276, -13.569, -13.669, -15.001, -12.774, -11.515, -11.380, -10.421, -5.929, -7.052, -3.098, -4.200, -1.800, 0.261, 3.811, 3.858, 1.900, -0.417, -3.485, -5.214, -7.962, -9.871, -11.397, -11.848, -10.793, -15.717, -12.563, -12.688, -13.878, -12.363, -10.765, -12.372, -10.091, -9.575, -7.661, -5.465, -3.256, -1.446, 1.474, 6.483, 7.127, 2.818, 0.778, -3.012, -5.299, -6.401, -6.973, -9.417, -12.378, -10.649, -9.883, -12.351, -14.865, -12.581, -10.214, -12.179, -10.553, -10.218, -7.993, -9.338, -5.841, -3.197, 0.272, 2.604, 4.542, 5.639, 1.707, -0.546, -2.532, -3.657, -3.922, -7.890, -9.529, -9.975, -13.615, -10.806, -10.868, -13.210, -12.689, -9.667, -11.035, -10.818, -9.951, -5.886, -6.304, -5.186, -1.671, 2.447, 1.924, 4.040, 8.755, 4.097, 2.806, 0.101, -4.443, -6.671, -5.025, -8.527, -8.556, -11.120, -9.597, -11.829, -12.240, -9.695, -9.825, -10.202, -7.947, -9.645, -5.596, -3.514, -2.543, -1.468, 0.263, 4.158, 5.772, 8.215, 4.838, 2.421, 0.593, -2.891, -3.617, -5.117, -8.417, -5.977, -8.885, -9.456, -8.521, -8.906, -11.960, -7.175, -6.907, -8.522, -6.298, -4.920, -5.102, -1.877, 0.394, 4.268, 2.611, 7.789, 8.192, 8.213, 2.959, 0.621, -1.270, -1.957, -3.380, -4.603, -6.603, -7.165, -8.631, -8.916, -8.997, -9.408, -5.424, -7.231, -6.398, -5.832, -4.353, -0.615, 1.412, 0.933, 3.776, 8.785, 8.293, 11.055, 9.078, 5.017, 4.586, -0.242, 0.431, -0.229, -4.661, -3.830, -3.447, -6.444, -6.254, -6.567, -7.971, -6.075, -3.471, -4.781, -2.110, -5.775, -0.956, -1.721, 1.810, 5.866, 9.917, 10.764, 14.510, 9.979, 7.317, 3.606, 1.875, 0.656, -0.221, 0.625, -0.476, -2.101, -3.866, -3.813, -3.005, -6.051, -4.223, -2.551, -3.891, 0.186, 0.212, 1.605, 3.319, 4.621, 8.065, 9.743, 12.588, 14.691, 13.647, 8.454, 8.320, 6.595, 3.417, 1.569, 1.337, -1.113, -3.167, -0.161, -3.842, -3.067, -3.122, -2.106, -0.329, -1.077, 0.115, 2.178, 2.647, 5.763, 8.761, 10.275, 14.580, 16.154, 17.761, 11.257, 11.266, 10.804, 6.227, 5.624, 5.290, 4.310, 2.143, 1.085, 0.947, -2.238, 0.249, -0.891, 0.118, -0.599, -0.378, 2.619, 4.795, 6.255, 5.974, 8.811, 10.948, 15.287, 18.913, 20.050, 17.394, 16.263, 12.018, 9.985, 9.473, 5.209, 4.086, 4.492, 5.770, 1.919, 2.750, 1.691, 3.485, 3.525, 3.741, 3.796, 5.109, 4.950, 7.240, 10.954, 9.431, 16.624, 16.462, 21.104, 23.951, 16.270, 15.317, 13.819, 12.826, 12.353, 11.720, 7.734, 8.776, 6.098, 5.353, 6.531, 4.298, 3.937, 7.201, 6.813, 7.394, 8.718, 7.508, 12.173, 15.522, 15.500, 18.637, 20.905, 25.077]
W_FULL = [
    [23.640, 20.820, 18.245, 15.869, 13.733, 11.772, 10.151, 8.697, 7.476, 6.532, 5.856, 5.451, 5.316, 5.451, 5.856, 6.531, 7.476, 8.692, 10.178, 11.935, 13.965, 16.266, 18.846, 21.693, 24.838],
    [20.768, 17.949, 15.365, 12.975, 10.816, 8.824, 7.154, 5.659, 4.414, 3.454, 2.764, 2.349, 2.206, 2.333, 2.732, 3.402, 4.342, 5.552, 7.034, 8.788, 10.814, 13.112, 15.689, 18.533, 21.674],
    [18.140, 15.316, 12.722, 10.318, 8.140, 6.123, 4.428, 2.908, 1.644, 0.669, -0.032, -0.456, -0.607, -0.486, -0.093, 0.570, 1.505, 2.711, 4.189, 5.939, 7.961, 10.256, 12.830, 15.671, 18.809],
    [15.724, 12.891, 10.288, 7.872, 5.680, 3.648, 1.938, 0.406, -0.870, -1.855, -2.564, -2.995, -3.153, -3.037, -2.650, -1.991, -1.061, 0.140, 1.613, 3.358, 5.377, 7.668, 10.238, 13.076, 16.211],
    [13.540, 10.698, 8.085, 5.659, 3.457, 1.418, -0.300, -1.839, -3.119, -4.108, -4.822, -5.258, -5.420, -5.309, -4.926, -4.272, -3.347, -2.150, -0.682, 1.059, 3.073, 5.360, 7.926, 10.760, 13.891],
    [11.580, 8.726, 6.101, 3.667, 1.462, -0.580, -2.293, -3.830, -5.108, -6.098, -6.812, -7.250, -7.414, -7.307, -6.928, -6.278, -5.357, -4.165, -2.702, -0.966, 1.043, 3.326, 5.887, 8.716, 11.844],
    [9.870, 7.000, 4.363, 1.923, -0.284, -2.316, -4.037, -5.566, -6.837, -7.821, -8.533, -8.970, -9.135, -9.030, -8.654, -8.007, -7.091, -5.904, -4.445, -2.715, -0.711, 1.566, 4.122, 6.946, 10.068],
    [8.355, 5.463, 2.811, 0.365, -1.834, -3.859, -5.537, -7.045, -8.303, -9.278, -9.984, -10.418, -10.582, -10.477, -10.104, -9.461, -8.549, -7.366, -5.914, -4.189, -2.191, 0.079, 2.629, 5.448, 8.565],
    [7.199, 4.269, 1.600, -0.844, -3.051, -5.034, -6.771, -8.263, -9.499, -10.464, -11.162, -11.591, -11.754, -11.650, -11.278, -10.638, -9.730, -8.554, -7.107, -5.389, -3.398, -1.134, 1.409, 4.221, 7.331],
    [6.227, 3.267, 0.586, -1.857, -4.057, -6.017, -7.738, -9.210, -10.423, -11.376, -12.064, -12.489, -12.649, -12.545, -12.175, -11.539, -10.636, -9.465, -8.025, -6.314, -4.331, -2.075, 0.460, 3.265, 6.367],
    [5.498, 2.529, -0.157, -2.597, -4.785, -6.730, -8.427, -9.875, -11.071, -12.010, -12.689, -13.109, -13.266, -13.161, -12.793, -12.161, -11.264, -10.100, -8.668, -6.966, -4.992, -2.745, -0.218, 2.577, 5.672],
    [5.058, 2.087, -0.597, -3.030, -5.208, -7.138, -8.821, -10.253, -11.434, -12.362, -13.033, -13.447, -13.602, -13.498, -13.132, -12.505, -11.614, -10.459, -9.036, -7.344, -5.380, -3.143, -0.627, 2.159, 5.244],
    [4.892, 1.927, -0.748, -3.173, -5.339, -7.256, -8.924, -10.343, -11.512, -12.429, -13.093, -13.502, -13.656, -13.552, -13.190, -12.569, -11.686, -10.540, -9.129, -7.448, -5.497, -3.272, -0.767, 2.008, 5.083],
    [5.005, 2.054, -0.609, -3.020, -5.174, -7.078, -8.733, -10.141, -11.299, -12.208, -12.866, -13.271, -13.423, -13.322, -12.964, -12.350, -11.477, -10.343, -8.946, -7.280, -5.343, -3.133, -0.641, 2.122, 5.187],
    [5.390, 2.458, -0.186, -2.581, -4.720, -6.610, -8.254, -9.650, -10.799, -11.700, -12.352, -12.755, -12.907, -12.807, -12.455, -11.850, -10.989, -9.871, -8.488, -6.840, -4.921, -2.727, -0.249, 2.502, 5.554],
    [6.056, 3.152, 0.532, -1.842, -3.965, -5.842, -7.473, -8.859, -10.000, -10.894, -11.541, -11.941, -12.093, -11.996, -11.650, -11.054, -10.208, -9.109, -7.753, -6.131, -4.234, -2.057, 0.406, 3.143, 6.184],
    [6.956, 4.098, 1.510, -0.842, -2.947, -4.809, -6.428, -7.804, -8.936, -9.824, -10.467, -10.864, -11.016, -10.922, -10.582, -9.997, -9.165, -8.091, -6.755, -5.158, -3.286, -1.126, 1.322, 4.046, 7.075],
    [8.020, 5.219, 2.665, 0.337, -1.749, -3.597, -5.204, -6.570, -7.694, -8.575, -9.213, -9.608, -9.759, -9.667, -9.330, -8.758, -7.926, -6.900, -5.508, -3.929, -2.083, 0.062, 2.495, 5.209, 8.226],
    [9.423, 6.661, 4.141, 1.837, -0.231, -2.065, -3.660, -5.016, -6.132, -7.006, -7.638, -8.029, -8.178, -8.087, -7.753, -7.187, -6.370, -5.356, -4.010, -2.455, -0.626, 1.504, 3.926, 6.630, 9.638],
    [11.001, 8.270, 5.776, 3.494, 1.444, -0.375, -1.959, -3.305, -4.412, -5.279, -5.905, -6.291, -6.436, -6.343, -6.009, -5.447, -4.634, -3.633, -2.284, -0.739, 1.078, 3.199, 5.613, 8.309, 11.309],
    [12.813, 10.105, 7.632, 5.370, 3.335, 1.530, -0.043, -1.379, -2.477, -3.336, -3.955, -4.335, -4.474, -4.377, -4.040, -3.475, -2.663, -1.663, -0.322, 1.218, 3.031, 5.147, 7.557, 10.247, 13.242],
    [14.832, 12.141, 9.685, 7.438, 5.417, 3.623, 2.062, 0.735, -0.354, -1.205, -1.817, -2.188, -2.320, -2.215, -1.870, -1.299, -0.479, 0.525, 1.873, 3.416, 5.231, 7.348, 9.757, 12.446, 15.438],
    [17.071, 14.392, 11.948, 9.713, 7.703, 5.920, 4.368, 3.050, 1.969, 1.127, 0.524, 0.162, 0.039, 0.155, 0.511, 1.095, 1.929, 2.948, 4.309, 5.863, 7.687, 9.810, 12.222, 14.913, 17.904],
    [19.514, 16.842, 14.406, 12.179, 10.178, 8.403, 6.859, 5.550, 4.478, 3.644, 3.051, 2.698, 2.586, 2.714, 3.085, 3.686, 4.541, 5.585, 6.973, 8.550, 10.391, 12.526, 14.945, 17.639, 20.633],
    [22.197, 19.523, 17.092, 14.870, 12.876, 11.108, 9.572, 8.270, 7.205, 6.380, 5.796, 5.453, 5.353, 5.495, 5.882, 6.503, 7.384, 8.462, 9.900, 11.519, 13.381, 15.531, 17.959, 20.659, 23.657],
]
S_TRUE = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]

# quantum proposal settings (Layden et al.): H = (1-g) alpha H_problem + g sum X
GAMMA, T_EVOL, TROTTER_STEPS = 0.30, 3.0, 3
SHOTS_DEFAULT, SHOTS_IONQ_CAP = 2000, 300

OFFLINE_HARDWARE = [
    ("IonQ Forte-1, run 1 (300 shots)", "98% +/- 11"),
    ("IonQ Forte-1, run 2 (300 shots)", "71% +/- 11"),
    ("IonQ Forte-1, run 3 (100 shots)", "81% +/- 19"),
    ("IonQ Forte Enterprise 1 (100 shots)", "60% +/- 18"),
    ("IBM Fez (2,000 shots)", "38% +/- 4"),
    ("IBM Kingston, run 1 (2,000 shots)", "30% +/- 4"),
    ("IBM Kingston, run 2 (2,000 shots)", "11% +/- 3"),
    ("IBM Marrakesh (2,000 shots)", "13% +/- 3"),
    ("Rigetti Cepheus (1,000 shots)", "9% +/- 5"),
]
OFFLINE_12Q_XEB = [
    ("IonQ Forte Enterprise 1 (100 shots)", "0.49 +/- 0.12"),
    ("IBM Kingston (2,000 shots)", "0.43 +/- 0.03"),
    ("Rigetti Cepheus (1,000 shots)", "0.02 +/- 0.02"),
]
OFFLINE_MIXING = [
    ("single-flip MCMC (classical)", "~0x"), ("uniform proposal (classical)", "1x"),
    ("quantum @ fidelity 0.10 (~Rigetti)", "1.4x"), ("quantum @ fidelity 0.30 (~IBM)", "2.3x"),
    ("quantum @ fidelity 0.84 (~IonQ)", "4.4x"), ("quantum, ideal", "5.1x"),
]


def clt(top, bot=CORE, dT=DT):
    A = B = D = Nt = Mt = 0.0
    z0 = 0.0
    for m in (bot, top):
        z1 = z0 + m["t"]
        Q = m["E"] / (1 - m["nu"])
        A += Q * (z1 - z0)
        B += Q * (z1 ** 2 - z0 ** 2) / 2
        D += Q * (z1 ** 3 - z0 ** 3) / 3
        Nt += Q * m["a"] * dT * (z1 - z0)
        Mt += Q * m["a"] * dT * (z1 ** 2 - z0 ** 2) / 2
        z0 = z1
    _, kappa = np.linalg.solve([[A, B], [B, D]], [Nt, Mt])
    return kappa, D - B ** 2 / A


class Plate:

    def __init__(self, N=25, regions=REGIONS):
        self.N, (self.rx, self.ry) = N, regions
        idx = lambda i, j: i * N + j

        def op(rows):
            r, c, v = [], [], []
            for k, entries in enumerate(rows):
                for col, val in entries:
                    r.append(k)
                    c.append(col)
                    v.append(val)
            return sp.csr_matrix((v, (r, c)), shape=(len(rows), N * N))

        xx = [([(idx(i - 1, j), 1), (idx(i, j), -2), (idx(i + 1, j), 1)], (i, j)) for i in range(1, N - 1) for j in range(N)]
        yy = [([(idx(i, j - 1), 1), (idx(i, j), -2), (idx(i, j + 1), 1)], (i, j)) for i in range(N) for j in range(1, N - 1)]
        xy = [([(idx(i + 1, j + 1), 1), (idx(i + 1, j), -1), (idx(i, j + 1), -1), (idx(i, j), 1)], (i + .5, j + .5))
              for i in range(N - 1) for j in range(N - 1)]
        self.Dxx, self.Dyy, self.Dxy = op([e for e, _ in xx]), op([e for e, _ in yy]), op([e for e, _ in xy])
        self.rxx = np.array([self._region(p) for _, p in xx])
        self.ryy = np.array([self._region(p) for _, p in yy])
        self.rxy = np.array([self._region(p) for _, p in xy])
        pinned = {idx(0, 0), idx(0, N - 1), idx(N - 1, 0)}
        self.free = np.array([k for k in range(N * N) if k not in pinned])
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        Bm = np.column_stack([np.ones(N * N), ii.ravel(), jj.ravel()])
        self.P = np.eye(N * N) - Bm @ np.linalg.pinv(Bm)

    def _region(self, p):
        a = min(int(p[0] / (self.N - 1) * self.rx), self.rx - 1)
        b = min(int(p[1] / (self.N - 1) * self.ry), self.ry - 1)
        return a * self.ry + b

    def solve(self, s, m0, dm, dD):
        mt = lambda r: m0 + dm * s[r]
        Df = lambda r: 1.0 + dD * s[r]
        K = (self.Dxx.T @ sp.diags(Df(self.rxx)) @ self.Dxx + self.Dyy.T @ sp.diags(Df(self.ryy)) @ self.Dyy
             + 2 * self.Dxy.T @ sp.diags(Df(self.rxy)) @ self.Dxy)
        f = self.Dxx.T @ mt(self.rxx) + self.Dyy.T @ mt(self.ryy)
        w = np.zeros(self.N ** 2)
        w[self.free] = spla.spsolve(K[self.free][:, self.free].tocsc(), -f[self.free])
        return self.P @ w


def calibrate():
    k0, D0 = clt(BU_NOMINAL)
    k1, D1 = clt(BU_CU_RICH)
    dD = D1 / D0 - 1
    prm = dict(m0=k0, dm=k1 * (1 + dD) - k0, dD=dD)
    plate = Plate()
    h_grid = L_MM / (plate.N - 1)
    model = lambda s: plate.solve(np.asarray(s, float), **prm) * h_grid ** 2 * 1e3  # um
    w0_full = model(np.zeros(R))
    A_full = np.column_stack([model(np.eye(R)[i]) - w0_full for i in range(R)])
    probe = np.array(PROBE)
    y, w0, A = np.array(Y_PROBE), w0_full[probe], A_full[probe]

    # all 2^R maps; index k <-> s_i = bit i of k (same order as Qiskit qubits)
    S = ((np.arange(2 ** R)[:, None] >> np.arange(R)) & 1).astype(float)
    resid = S @ A.T + w0 - y
    sse = np.sum(resid ** 2, 1)

    best = None
    for sig in np.geomspace(0.5, 50, 160):
        E = sse / (2 * sig ** 2)
        ll = -E.min() + np.log(np.exp(-(E - E.min())).sum()) - len(y) * np.log(sig)
        if best is None or ll > best[0]:
            best = (ll, sig)
    sigma = best[1]
    E = sse / (2 * sigma ** 2)
    E -= E.min()
    p = np.exp(-E)
    p /= p.sum()

    b = A.sum(1) / 2 - (y - w0)
    G = A.T @ A
    J = np.triu(G, 1) / (4 * sigma ** 2)
    h = -(A.T @ b) / (2 * sigma ** 2)
    return dict(S=S, E=E, p=p, sigma=sigma, J=J, h=h, A_full=A_full, w0_full=w0_full)


def local_minima(E):
    return [k for k in range(len(E)) if all(E[k ^ (1 << i)] >= E[k] for i in range(R))]


def proposal_circuit(start, J, h, measure=True):
    alpha = np.sqrt(R) / np.sqrt(np.sum(J ** 2) + np.sum(h ** 2))
    dt = T_EVOL / TROTTER_STEPS
    qc = QuantumCircuit(R, R if measure else 0)
    for i in range(R):
        if (start >> i) & 1:
            qc.x(i)
    for _ in range(TROTTER_STEPS):
        for i in range(R):
            for j in range(i + 1, R):
                qc.rzz(2 * dt * (1 - GAMMA) * alpha * J[i, j], i, j)
        for i in range(R):
            qc.rz(2 * dt * (1 - GAMMA) * alpha * h[i], i)
        for i in range(R):
            qc.rx(2 * dt * GAMMA, i)
    if measure:
        qc.measure(range(R), range(R))
    return qc


def run_on_backend(qc, shots):
    name = str(getattr(backend, "name", backend))
    is_ionq = "ionq" in name.lower()
    if is_ionq:
        shots = min(shots, SHOTS_IONQ_CAP)
        try:
            from qiskit_ionq import ErrorMitigation
            job = backend.run(qc, shots=shots, error_mitigation=ErrorMitigation.NO_DEBIASING)
        except Exception:
            job = backend.run(qc, shots=shots)
    else:
        try:
            tqc = transpile(qc, backend)
        except Exception:
            tqc = qc
        job = backend.run(tqc, shots=shots)
    for _ in range(720):
        try:
            st = str(job.status())
        except Exception:
            break
        if "DONE" in st:
            break
        if "ERROR" in st or "CANCEL" in st:
            raise RuntimeError(f"job ended with status {st}")
        time.sleep(5)
    return name, shots, job.result().get_counts()


def main(shots: int = 2000):
    t0 = time.time()
    print("=" * 72)
    print("The Warm Icing Model: warpage calibration with quantum-enhanced MCMC")
    print("=" * 72)

    c = calibrate()
    S, E, p = c["S"], c["E"], c["p"]
    s_true = np.array(S_TRUE)
    k_true = int(sum(int(v) << i for i, v in enumerate(S_TRUE)))
    k_map = int(np.argmax(p))
    marg = p @ S
    w_full = np.array(W_FULL).ravel()
    pred = p @ (S @ c["A_full"].T + c["w0_full"])
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))

    print(f"\n[1-4] Calibration from {len(PROBE)} measured points (12 regions, 4,096 possible maps)")
    print(f"  fitted noise level sigma = {c['sigma']:.2f} um (true measurement noise {NOISE_UM} um)")
    print("  region :  " + " ".join(f"{i:>4d}" for i in range(R)))
    print("  P(Cu)  :  " + " ".join(f"{m:4.2f}" for m in marg))
    print("  truth  :  " + " ".join(f"{int(v):>4d}" for v in s_true))
    print(f"  most likely map: {int((S[k_map] == s_true).sum())}/{R} regions correct; "
          f"true map has probability {p[k_true]:.3f} (rank {int((p > p[k_true]).sum()) + 1})")
    print(f"  maps holding 90% of probability: {int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.9) + 1)}")
    print(f"  warpage prediction error over the full surface (RMS):")
    print(f"     nominal design model : {rms(c['w0_full'] - w_full):6.2f} um")
    print(f"     calibrated (Bayesian): {rms(pred - w_full):6.2f} um")
    J = c["J"]
    off = J[np.triu_indices(R, 1)]
    print(f"  Ising couplings: {np.sum(off > 0.1 * off.max())}/{len(off)} pairs above 10% of the strongest "
          f"-> dense, all-to-all problem")

    mins = local_minima(E)
    false_mins = [k for k in mins if k != int(np.argmin(E))]
    best = None
    for k in false_mins:
        lower_k = E < E[k]
        if not 0.02 <= lower_k.mean() <= 0.30:
            continue
        ideal_k = Statevector(proposal_circuit(k, J, c["h"], measure=False)).probabilities()
        gain = ideal_k[lower_k].sum() - lower_k.mean()
        if best is None or gain > best[0]:
            best = (gain, k, ideal_k)
    _, start, ideal = best
    lower = E < E[start]
    print(f"\n[5] Quantum proposal from a false local optimum")
    print(f"  posterior has {len(mins)} local optima; single-region-flip MCMC can only leave one "
          f"by first accepting a worse map")
    print(f"  start map {start}: {100 * lower.mean():.0f}% of all maps fit the data better than it")

    qc = proposal_circuit(start, J, c["h"])
    pd_ideal = float(ideal[lower].sum())
    pd_uniform = float(lower.mean())
    print(f"  circuit: {R} qubits, {qc.count_ops().get('rzz', 0)} two-qubit ZZ interactions")
    print(f"  ideal P(better map proposed) = {pd_ideal:.3f};  random-guess floor = {pd_uniform:.3f};  single flip = 0.000")

    name, used_shots, counts = run_on_backend(qc, shots)
    hits = sum(n for bits, n in counts.items() if lower[int(bits.replace(' ', ''), 2)])
    total = sum(counts.values())
    pd = hits / total
    err = np.sqrt(pd * (1 - pd) / total)
    kept = (pd - pd_uniform) / (pd_ideal - pd_uniform)
    print(f"  backend {name}, {used_shots} shots: P(better map) = {pd:.3f} +/- {err:.3f} "
          f"-> signal kept {100 * kept:.0f}% +/- {100 * err / (pd_ideal - pd_uniform):.0f}%")

    print("\n[6] Context from our offline runs (same method, not recomputed here)")
    print("  Hardware, deep 9-qubit test circuit, signal kept:")
    for a, b2 in OFFLINE_HARDWARE:
        print(f"     {a:40s} {b2}")
    print("  Hardware, this 12-qubit circuit, XEB fidelity:")
    for a, b2 in OFFLINE_12Q_XEB:
        print(f"     {a:40s} {b2}")
    print("  Modelled mixing speed vs random proposals (exact spectral gaps, median, 9 regions, 3% noise):")
    for a, b2 in OFFLINE_MIXING:
        print(f"     {a:40s} {b2}")
    print("  Against classical block moves at equal posterior evaluations there is no advantage at 3% noise;")
    print("  exact enumeration is cheapest at this size (RESULTS_MAP.txt, R17).")
    print(f"\nDone in {time.time() - t0:.1f} s")

    out = dict(backend=name, shots=used_shots, start_state=start, p_down=pd, p_down_ideal=pd_ideal,
               p_down_uniform=pd_uniform, signal_kept=kept, map_correct=int((S[k_map] == s_true).sum()),
               region_prob_cu=[round(float(m), 3) for m in marg], counts=counts)
    print("RESULT_JSON_START")
    print(json.dumps(out))
    print("RESULT_JSON_END")
    return out
