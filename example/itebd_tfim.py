from __future__ import annotations

import math

import numpy as np

import tcapi


TEN_KIND = tcapi.TenKindF64


def exact_tfim_energy_per_site(J: float, g: float, num_points: int = 200001) -> float:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    k = np.linspace(0.0, 2.0 * np.pi, num_points, dtype=np.float64)
    integrand = np.sqrt(J * J + g * g - 2.0 * J * g * np.cos(k))
    return float(-(1.0 / (2.0 * np.pi)) * np.trapezoid(integrand, k))


def make_tfim_two_site_hamiltonian(ctx: tcapi.NumpyContext, J: float, g: float) -> np.ndarray:
    h = tcapi.zeros(ctx, TEN_KIND, (4, 4))
    h = tcapi.set_elem(ctx, h, (0, 0), J)
    h = tcapi.set_elem(ctx, h, (3, 3), J)
    h = tcapi.set_elem(ctx, h, (1, 1), -J)
    h = tcapi.set_elem(ctx, h, (2, 2), -J)

    g2 = -0.5 * g
    for coors in [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)]:
        h = tcapi.set_elem(ctx, h, coors, g2)
    return h


def make_imaginary_time_gate(ctx: tcapi.NumpyContext, h: np.ndarray, dt: float, d: int = 2) -> np.ndarray:
    w_diag, v = tcapi.eigh(ctx, h, 1)
    w_vector = tcapi.diag(ctx, w_diag)
    w_vector = tcapi.for_each(ctx, w_vector, lambda elem: math.exp(-dt * elem))
    w_matrix = tcapi.diag(ctx, w_vector)

    u = tcapi.contract(ctx, v, "ij", w_matrix, "jk", "ik")
    v_transpose = tcapi.transpose(ctx, v, (1, 0))
    u = tcapi.contract(ctx, u, "ij", v_transpose, "jk", "ik")
    return tcapi.reshape(ctx, u, (d, d, d, d))


def itebd_imaginary_time(
    ctx: tcapi.NumpyContext,
    gate: np.ndarray,
    N: int,
    dt: float,
    chi: int,
    s_min: float = 1.0e-10,
    seed: int = 280711,
) -> list[float]:

    rng = np.random.default_rng(seed)
    gen = lambda: float(rng.uniform(-1.0, 1.0))
    gamma = [
        tcapi.random(ctx, TEN_KIND, (chi, 2, chi), gen),
        tcapi.random(ctx, TEN_KIND, (chi, 2, chi), gen),
    ]
    lam = [
        tcapi.random(ctx, TEN_KIND, (chi,), gen),
        tcapi.random(ctx, TEN_KIND, (chi,), gen),
    ]

    energies: list[float] = []

    for i in range(N):
        a = i % 2
        b = (i + 1) % 2

        lam_a_full = tcapi.diag(ctx, lam[a])
        lam_b_full = tcapi.diag(ctx, lam[b])

        theta = tcapi.contract(ctx, lam_b_full, "ij", gamma[a], "jkl", "ikl")
        theta = tcapi.contract(ctx, theta, "ijk", lam_a_full, "kl", "ijl")
        theta = tcapi.contract(ctx, theta, "ijk", gamma[b], "klm", "ijlm")
        theta = tcapi.contract(ctx, theta, "ijkl", lam_b_full, "lm", "ijkm")
        theta = tcapi.contract(ctx, theta, "ijkl", gate, "jkmn", "imnl")

        gamma[a], sigma, gamma[b], trunc_err = tcapi.trunc_svd(ctx, theta, 2, chi, s_min)
        lam[a], _ = tcapi.normalize(ctx, tcapi.diag(ctx, sigma))

        lam_b_inv = tcapi.diag(ctx, tcapi.for_each(ctx, lam[b], lambda elem: 1.0 / elem))
        gamma[a] = tcapi.contract(ctx, lam_b_inv, "ij", gamma[a], "jkl", "ikl")
        gamma[b] = tcapi.contract(ctx, gamma[b], "ijk", lam_b_inv, "kl", "ijl")

        if i >= 1:
            theta_norm_squared = [0.0]
            tcapi.for_each(ctx, theta, lambda elem: theta_norm_squared.__setitem__(0, theta_norm_squared[0] + elem * elem))
            energies.append(-math.log(theta_norm_squared[0]) / dt / 2.0)

    return energies