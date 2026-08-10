import numpy as np
from .context import NumpyContext
from .ten_kind import TenKind
from .misc import _check_ctx


def order(ctx: NumpyContext, a: np.ndarray) -> int:
    _check_ctx(ctx)
    return a.ndim


def shape(ctx: NumpyContext, a: np.ndarray) -> tuple:
    _check_ctx(ctx)
    return tuple(a.shape)


def size(ctx: NumpyContext, a: np.ndarray) -> int:
    _check_ctx(ctx)
    return int(a.size)


def size_bytes(ctx: NumpyContext, a: np.ndarray) -> int:
    _check_ctx(ctx)
    return int(a.nbytes)


def get_elem(ctx: NumpyContext, a: np.ndarray, coors: tuple):
    _check_ctx(ctx)
    return a[tuple(coors)]