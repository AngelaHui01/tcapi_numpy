# tcapi/__init__.py
import numpy as np

# ── Type system ──────────────────────────────────────────────────────────── #
from .ten_kind     import (TenKind,
                           TenKindF32, TenKindF64,
                           TenKindC64, TenKindC128)

# ── Context ──────────────────────────────────────────────────────────────── #
from .context      import NumpyContext

# ── Internal helpers (not public API) ────────────────────────────────────── #
from .misc         import _check_ctx
# ── Queries ──────────────────────────────────────────────────────────────── #
from .queries      import order, shape, size, size_bytes, get_elem

# ── Constructors ─────────────────────────────────────────────────────────── #
from .constructors import (allocate, zeros, assign_from_range, fill, random,
                           eye, copy)

# ── Manipulation ─────────────────────────────────────────────────────────── #
from .manipulation import (set_elem, transpose, reshape, cplx_conj,
                           to_cplx, real, imag, expand, shrink,
                           extract_sub, replace_sub, concatenate,
                           stack, for_each,
                           for_each_with_coors)

# ── Linear algebra ───────────────────────────────────────────────────────── #
from .linalg       import (norm, diag, normalize, scale, trace, contract, svd,
                           trunc_svd, inverse, linear_combine, exp,
                           lq, eigvals, eigvalsh, eigh, eig, qr)

# ── I/O (file operations only) ───────────────────────────────────────────── #
from .io           import save, load

# ── Miscellaneous routines ───────────────────────────────────────────────── #
from .misc         import (create_context, destroy_context, version, show, close,
                           convert, to_range)

# ── TenKind introspection helpers (Appendix C.1) ─────────────────────────── #
def ten_kind_ten_type(Ten: TenKind) -> type:
    return Ten.ten_type

def ten_kind_dtype(Ten: TenKind) -> np.dtype:
    return Ten.dtype

def ten_kind_real_dtype(Ten: TenKind) -> np.dtype:
    return Ten.real_dtype

def ten_kind_cplx_dtype(Ten: TenKind) -> np.dtype:
    return Ten.cplx_dtype

def real_ten_kind(Ten: TenKind) -> TenKind:
    return Ten.real_ten_kind

def cplx_ten_kind(Ten: TenKind) -> TenKind:
    return Ten.cplx_ten_kind

def ten_kind(ctx: NumpyContext, tensor: np.ndarray) -> TenKind:
    _check_ctx(ctx)
    return TenKind(tensor.dtype)

# ── Runtime diagnostics — MUST be last ───────────────────────────────────── #
from .diagnostics import verbose_wrap
import tcapi as _self
import types as _types

def _apply_verbose_wrap():
    for _name in list(vars(_self)):
        if _name.startswith("_"):
            continue
        _obj = getattr(_self, _name)
        if callable(_obj) and isinstance(_obj, _types.FunctionType):
            setattr(_self, _name, verbose_wrap(_obj))

_apply_verbose_wrap()
