from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

import tcapi


TEN_KIND = tcapi.TenKindF64
PHYSICAL_DIM = 2
TENSOR_LABELS = "lrdu"


def initial_ising_tensor(ctx: tcapi.NumpyContext, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    s = np.array([1.0, -1.0], dtype=np.float64)
    energy = (
        s[:, None, None, None] * s[None, None, None, :]  # s_l * s_d
        + s[None, :, None, None] * s[None, None, None, :]  # s_r * s_d
        + s[None, :, None, None] * s[None, None, :, None]  # s_r * s_u
        + s[None, None, :, None] * s[:, None, None, None]  # s_u * s_l
    )
    return np.exp(-energy / temperature)


def exact_logz_per_site(temperature: float) -> float:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")

    beta = 1.0 / temperature
    cosh_2beta = math.cosh(2.0 * beta)
    log_cosh_sq = math.log(cosh_2beta * cosh_2beta)
    log_k = -2.0 * math.log(math.sinh(2.0 * beta))
    k = math.exp(log_k)

    def integrand(theta: float) -> float:
        radical = max(1.0 + k * k - 2.0 * k * math.cos(2.0 * theta), 0.0)
        sqrt_term = math.sqrt(radical)
        log_sqrt = -math.inf if sqrt_term == 0.0 else math.log(sqrt_term)
        return float(np.logaddexp(log_cosh_sq, log_sqrt - log_k))

    integral, _ = quad(integrand, 0.0, math.pi, epsabs=1.0e-13, epsrel=1.0e-13, limit=400)
    return 0.5 * math.log(2.0) + integral / (2.0 * math.pi)


def _identity(ctx: tcapi.NumpyContext, dim: int) -> np.ndarray:

    return tcapi.diag(ctx, tcapi.fill(ctx, TEN_KIND, (dim,), 1.0))


def _relabel(
    ctx: tcapi.NumpyContext,
    tensor: np.ndarray,
    labels: str,
    old_label: str,
    new_label: str,
) -> np.ndarray:

    dim = int(tcapi.shape(ctx, tensor)[labels.index(old_label)])
    return tcapi.contract(
        ctx,
        tensor,
        labels,
        _identity(ctx, dim),
        old_label + new_label,
        labels.replace(old_label, new_label),
    )


def _factor(
    ctx: tcapi.NumpyContext,
    tensor: np.ndarray,
    row_labels: str,
    col_labels: str,
    new_label: str,
    maxdim: int,
) -> tuple[np.ndarray, np.ndarray]:
    axes_order = tuple(TENSOR_LABELS.index(label) for label in row_labels) + tuple(
        TENSOR_LABELS.index(label) for label in col_labels
    )
    rotated = tcapi.transpose(ctx, tensor, axes_order)

    u, sigma, v_dag, _ = tcapi.trunc_svd(ctx, rotated, 2, maxdim, 0.0)
    singular_values = np.sqrt(np.clip(tcapi.diag(ctx, sigma), 0.0, None))
    sqrt_sigma = tcapi.diag(ctx, singular_values)

    left = tcapi.contract(ctx, u, row_labels + "k", sqrt_sigma, "k" + new_label, row_labels + new_label)
    right = tcapi.contract(ctx, sqrt_sigma, "k" + new_label, v_dag, "k" + col_labels, new_label + col_labels)
    return left, right


def _trg_step(
    ctx: tcapi.NumpyContext,
    tensor: np.ndarray,
    maxdim: int,
) -> tuple[np.ndarray, float]:

    fl, fr = _factor(ctx, tensor, "rd", "lu", "a", maxdim)
    fr = _relabel(ctx, fr, "alu", "a", "b")

    fu, fd = _factor(ctx, tensor, "ld", "ur", "c", maxdim)
    fd = _relabel(ctx, fd, "cur", "c", "e")

    # Match the four outer bonds for the recombination, like ITensor
    # ``Fl *= delta(r,l); Fu *= delta(d,u); Fr *= delta(l,r); Fd *= delta(u,d)``.
    fl = _relabel(ctx, fl, "rda", "r", "l")
    fu = _relabel(ctx, fu, "ldc", "d", "u")
    fr = _relabel(ctx, fr, "blu", "l", "r")
    fd = _relabel(ctx, fd, "eur", "u", "d")

    # Recombine: ITensor ``A = Fl * Fu * Fr * Fd``.
    left_pair = tcapi.contract(ctx, fl, "lda", fu, "luc", "dauc")
    right_pair = tcapi.contract(ctx, fr, "bru", fd, "edr", "bued")
    coarse = tcapi.contract(ctx, left_pair, "dauc", right_pair, "bued", "abce")

    # Trace ``elt(A*delta(l,r)*delta(u,d))`` and normalize A /= TrA.
    traced = tcapi.contract(ctx, coarse, "abce", _identity(ctx, int(coarse.shape[0])), "ab", "ce")
    traced = tcapi.contract(ctx, traced, "ce", _identity(ctx, int(coarse.shape[2])), "ce", "")
    trace = float(traced)
    if not math.isfinite(trace) or trace <= 0.0:
        raise ValueError(f"non-finite or non-positive TRG trace: {trace}")

    normalized = tcapi.scale(ctx, coarse, 1.0 / trace)
    return normalized, trace


def trg_ising(
    temperature: float,
    maxdim: int,
    n_steps: int,
    ctx: tcapi.NumpyContext | None = None,
) -> float:

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if maxdim < 1:
        raise ValueError("maxdim must be positive")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")

    owns_context = ctx is None
    if ctx is None:
        ctx = tcapi.create_context(TEN_KIND)

    try:
        tensor = initial_ising_tensor(ctx, temperature)
        logz_per_tensor = 0.0
        for scale in range(n_steps):
            tensor, trace = _trg_step(ctx, tensor, maxdim)
            logz_per_tensor += math.log(trace) / (2 ** (scale + 1))
        return 0.5 * logz_per_tensor
    finally:
        if owns_context:
            tcapi.destroy_context(ctx)