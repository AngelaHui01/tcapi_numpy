import numpy as np

from .ten_kind import TenKind
from .context import NumpyContext
from .misc import _check_ctx

# TCAPI spec version this implementation targets
_TCAPI_VERSION = "1.0"


def save(ctx: NumpyContext, a: np.ndarray, strg: str) -> None:
    _check_ctx(ctx)
    np.save(strg, a)


def load(ctx: NumpyContext, Ten: TenKind, strg: str) -> np.ndarray:
    _check_ctx(ctx)
    # np.save appends .npy automatically; np.load needs it explicitly
    if not strg.endswith(".npy"):
        strg = strg + ".npy"
    a = np.load(strg, allow_pickle=False)
    if a.dtype != Ten.dtype:
        raise TypeError(
            f"load: file dtype {a.dtype} does not match "
            f"TenKind dtype {Ten.dtype}."
        )
    return a