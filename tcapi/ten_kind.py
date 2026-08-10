from __future__ import annotations
import numpy as np
from typing import Type


class TenKind:
    """
    Runtime tensor-kind descriptor for the NumPy TCAPI backend.

    In the Python TCAPI spec (Appendix C.1), TenKind is the runtime
    counterpart of the C++ compile-time tensor type TenT. It does NOT
    hold tensor data — it only carries type metadata used by TCAPI
    functions to create and inspect tensors.

    Required metadata fields per spec:
        ten_type      : the backend tensor class (np.ndarray for NumPy)
        dtype         : the backend-native element dtype
        real_dtype    : the real scalar dtype
        cplx_dtype    : the complex counterpart dtype
        real_ten_kind : TenKind for the real version of this family
        cplx_ten_kind : TenKind for the complex version of this family
    """

    # The backend tensor class — always np.ndarray for this implementation.
    # In C++ this is TenT itself; in Python it is stored as an attribute.
    ten_type: Type = np.ndarray

    def __init__(self, dtype: np.dtype | type) -> None:
        # Always normalise to a proper np.dtype object so comparisons
        # like TenKind(np.float32) == TenKind("float32") work correctly.
        self.dtype = np.dtype(dtype)

        # Validate: TCAPI restricts elements to real or complex fields only.
        # (Appendix C.1: "K is restricted to R or C")
        if not (
            np.issubdtype(self.dtype, np.floating) or
            np.issubdtype(self.dtype, np.complexfloating)
        ):
            raise TypeError(
                f"TenKind dtype must be real or complex floating-point, "
                f"got {self.dtype}. TCAPI restricts K to R or C."
            )

    # ------------------------------------------------------------------
    # Spec-required metadata properties
    # ------------------------------------------------------------------

    @property
    def real_dtype(self) -> np.dtype:
        if np.issubdtype(self.dtype, np.complexfloating):
            # np produces a zero-valued scalar; .real gives the real part;
            # .dtype extracts its dtype.
            return np.dtype(self.dtype.type(0).real.dtype)
        return self.dtype

    @property
    def cplx_dtype(self) -> np.dtype:
        if np.issubdtype(self.dtype, np.complexfloating):
            return self.dtype
        _promotion = {
            np.dtype(np.float32): np.dtype(np.complex64),
            np.dtype(np.float64): np.dtype(np.complex128),
        }
        return _promotion[self.dtype]

    @property
    def real_ten_kind(self) -> TenKind:
        return TenKind(self.real_dtype)

    @property
    def cplx_ten_kind(self) -> TenKind:
        return TenKind(self.cplx_dtype)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_real(self) -> bool:
        return np.issubdtype(self.dtype, np.floating)

    def is_complex(self) -> bool:
        return np.issubdtype(self.dtype, np.complexfloating)

    # ------------------------------------------------------------------
    # Python dunder methods — needed so TenKind works in dicts/sets
    # and prints usefully in the REPL
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TenKind(dtype={self.dtype})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TenKind):
            return NotImplemented
        return self.dtype == other.dtype

    def __hash__(self) -> int:
        # Required because we defined __eq__; allows TenKind as dict key.
        return hash(self.dtype)


# ----------------------------------------------------------------------
# Canonical pre-built instances
# ----------------------------------------------------------------------
# The spec says "concrete TenKind objects are provided and documented
# by each TCAPI implementation" (Appendix C.1). These are the four
# standard NumPy tensor kinds, covering the two precisions × real/complex.

TenKindF32  = TenKind(np.float32)     # 32-bit real
TenKindF64  = TenKind(np.float64)     # 64-bit real   ← recommended default
TenKindC64  = TenKind(np.complex64)   # 32-bit complex
TenKindC128 = TenKind(np.complex128)  # 64-bit complex