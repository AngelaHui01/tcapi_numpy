from __future__ import annotations
import numpy as np
from .ten_kind import TenKind
from .context import NumpyContext



def create_context(Ten: TenKind, **options) -> NumpyContext:
    ctx = NumpyContext()
    ctx._active = True
    return ctx


def destroy_context(ctx: NumpyContext) -> None:
    if not ctx._active:
        raise RuntimeError("destroy_context called on an already-destroyed context.")
    ctx._active = False



def version(Ten: TenKind) -> str:
    return "1.0"



def show(ctx: NumpyContext, a: np.ndarray) -> None:
    _check_ctx(ctx)
    print(f"Tensor | shape={a.shape} | dtype={a.dtype} | order={a.ndim}")
    print(a)



def close(ctx: NumpyContext, a: np.ndarray, b: np.ndarray,
          epsilon: float) -> bool:
    _check_ctx(ctx)

    if epsilon < 0:
        raise ValueError("close: epsilon must be >= 0.")
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True

    return bool(np.max(np.abs(a - b)) <= epsilon)

def convert(ctx1: NumpyContext, t1: np.ndarray,
            ctx2: NumpyContext, Ten2: TenKind) -> np.ndarray:
    _check_ctx(ctx1)
    _check_ctx(ctx2)
    return t1.astype(Ten2.dtype, copy=True)


def to_range(ctx: NumpyContext, a: np.ndarray,
             coors2idx) -> list:
    _check_ctx(ctx)
    n = a.size
    result = [None] * n
    for coors in np.ndindex(a.shape):
        idx = coors2idx(coors)
        if not (0 <= idx < n):
            raise IndexError(
                f"to_range: coors2idx returned {idx} for coors {coors}, "
                f"which is out of range [0, {n})."
            )
        result[idx] = a[coors].item()
    return result

def _check_ctx(ctx: NumpyContext) -> None:
    if not ctx._active:
        raise RuntimeError(
            "TCAPI: operation called on a destroyed context. "
            "Call tcapi.create_context() first."
        )