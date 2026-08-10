import numpy as np
from typing import Callable
from .context import NumpyContext
from .ten_kind import TenKind
from .misc import _check_ctx


def allocate(ctx: NumpyContext, Ten: TenKind, shape: tuple) -> np.ndarray:
    _check_ctx(ctx)
    return np.empty(shape, dtype=Ten.dtype)


def zeros(ctx: NumpyContext, Ten: TenKind, shape: tuple) -> np.ndarray:
    _check_ctx(ctx)
    return np.zeros(shape, dtype=Ten.dtype)


def assign_from_range(ctx: NumpyContext, Ten: TenKind,
                      shape: tuple, values: list,
                      coors2idx) -> np.ndarray:
    _check_ctx(ctx)
    a = np.empty(shape, dtype=Ten.dtype)
    for coors in np.ndindex(tuple(shape)):
        idx = coors2idx(coors)
        a[coors] = values[idx]
    return a


def fill(ctx: NumpyContext, Ten: TenKind, shape: tuple, v) -> np.ndarray:
    _check_ctx(ctx)
    return np.full(shape, v, dtype=Ten.dtype)


def random(ctx: NumpyContext, Ten: TenKind,
           shape: tuple, gen: Callable) -> np.ndarray:
    """
    Creates a tensor filled by calling gen() once per element.
    gen is any zero-argument callable returning a scalar,
    matching the C++ generator convention in the spec.

    Example:
        rng = np.random.default_rng()
        gen = lambda: rng.uniform(-1.0, 1.0)
        psi = random(ctx, TenKindF64, (2, 2, 2), gen)

    Spec: tcapi::random(ctx, shape, gen)
    """
    _check_ctx(ctx)
    # Vectorise gen() across the full flat array, then reshape
    flat = np.array([gen() for _ in range(int(np.prod(shape)))],
                    dtype=Ten.dtype)
    return flat.reshape(shape)


def eye(ctx: NumpyContext, Ten: TenKind, N: int) -> np.ndarray:
    _check_ctx(ctx)
    return np.eye(N, dtype=Ten.dtype)


def copy(ctx: NumpyContext, orig: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    return orig.copy()