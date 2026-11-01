import numpy as np
from numbers import Integral, Real
from scipy.linalg import expm
from typing import Sequence

from .context import NumpyContext
from .misc import _check_ctx


def _matrixize(a: np.ndarray, num_of_bds_as_row: int):
    if (not isinstance(num_of_bds_as_row, Integral)
            or not 1 <= num_of_bds_as_row < a.ndim):
        raise ValueError(
            "num_of_bds_as_row must satisfy 1 <= k < a.ndim"
        )

    k = int(num_of_bds_as_row)
    row_shape = a.shape[:k]
    col_shape = a.shape[k:]
    rows = int(np.prod(row_shape))
    cols = int(np.prod(col_shape))
    return a.reshape(rows, cols), row_shape, col_shape


def norm(ctx: NumpyContext, a: np.ndarray) -> float:
    _check_ctx(ctx)
    return float(np.linalg.norm(a.ravel()))


def diag(ctx: NumpyContext, a: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    if a.ndim not in (1, 2):
        raise ValueError("diag: input must be first- or second-order")
    return np.diag(a)


def normalize(ctx: NumpyContext, a: np.ndarray):
    _check_ctx(ctx)
    original_norm = norm(ctx, a)
    if original_norm == 0:
        raise ValueError("normalize: cannot normalize a zero tensor")
    return a / original_norm, original_norm


def scale(ctx: NumpyContext, a: np.ndarray, s) -> np.ndarray:
    _check_ctx(ctx)
    return a * s

def trace(ctx: NumpyContext, a: np.ndarray, bdidx_pairs) -> np.ndarray:
    """
    Partially trace `a` over the specified pairs of bond indices.

    Each pair (i, j) identifies two distinct axes of equal dimension.
    The function sums over indices where those two axes are equal.
    Remaining axes retain their original relative order.

    Example:
        a.shape == (3, 4, 2, 4, 2)
        trace(ctx, a, [(1, 3), (2, 4)]).shape == (3,)

    Spec: tcapi::trace(ctx, a, bdidx_pairs)
    """
    _check_ctx(ctx)

    pairs = list(bdidx_pairs)
    used_axes = set()

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                "trace: each bond-index entry must be a pair"
            )

        i, j = pair
        if (not isinstance(i, Integral)
                or not isinstance(j, Integral)):
            raise TypeError(
                "trace: bond indices must be integers"
            )

        i, j = int(i), int(j)

        if not (0 <= i < a.ndim and 0 <= j < a.ndim):
            raise ValueError(
                f"trace: bond index out of range for shape {a.shape}"
            )
        if i == j:
            raise ValueError(
                "trace: a bond cannot be paired with itself"
            )
        if i in used_axes or j in used_axes:
            raise ValueError(
                "trace: each bond may appear in at most one pair"
            )
        if a.shape[i] != a.shape[j]:
            raise ValueError(
                f"trace: paired bonds {i} and {j} have unequal "
                f"dimensions {a.shape[i]} and {a.shape[j]}"
            )

        used_axes.add(i)
        used_axes.add(j)

    if not pairs:
        return a.copy()

    # Give each axis a unique integer label, then assign a common label
    # to the two axes in every traced pair. Repeated labels are summed.
    labels_in = list(range(a.ndim))
    for i, j in pairs:
        labels_in[int(j)] = labels_in[int(i)]

    labels_out = [
        labels_in[axis]
        for axis in range(a.ndim)
        if axis not in used_axes
    ]

    return np.einsum(a, labels_in, labels_out, optimize=True)

def contract(ctx: NumpyContext,
             a: np.ndarray, bd_labs_a,
             b: np.ndarray, bd_labs_b,
             bd_labs_c) -> np.ndarray:
    _check_ctx(ctx)

    if len(bd_labs_a) != a.ndim or len(bd_labs_b) != b.ndim:
        raise ValueError("contract: label count must match tensor order")

    all_strings = (
        isinstance(bd_labs_a, str)
        and isinstance(bd_labs_b, str)
        and isinstance(bd_labs_c, str)
    )
    all_sequences = (
        not isinstance(bd_labs_a, str)
        and not isinstance(bd_labs_b, str)
        and not isinstance(bd_labs_c, str)
    )

    if not (all_strings or all_sequences):
        raise ValueError(
            "contract: labels must all be strings or all be sequences"
        )

    if all_strings:
        la, lb, lo = bd_labs_a, bd_labs_b, bd_labs_c
    else:
        la, lb, lo = list(bd_labs_a), list(bd_labs_b), list(bd_labs_c)

    if len(set(la)) != len(la) or len(set(lb)) != len(lb):
        raise ValueError(
            "contract: repeated labels within one operand are unsupported; "
            "use trace first"
        )

    # Bond dimensions associated with identical labels must agree.
    label_dims: dict = {}
    for labs, tensor in ((la, a), (lb, b)):
        for axis, label in enumerate(labs):
            if label in label_dims and label_dims[label] != tensor.shape[axis]:
                raise ValueError(
                    f"contract: bond dimensions for label {label!r} disagree "
                    f"({label_dims[label]} vs {tensor.shape[axis]})"
                )
            label_dims[label] = tensor.shape[axis]

    try:
        if all_strings:
            # The string form "->" with two operands is invalid for
            # zeroth-order tensors (no subscripts). Handle separately.
            if a.ndim == 0 and b.ndim == 0 and la == "" and lb == "" and lo == "":
                return np.asarray(a * b)
            return np.einsum(f"{la},{lb}->{lo}", a, b, optimize=True)
        # Integer-label form handles 0-d inputs correctly via [] subscripts.
        return np.einsum(a, la, b, lb, lo, optimize=True)
    except ValueError as exc:
        raise ValueError(f"contract: {exc}") from exc


def svd(ctx: NumpyContext, a: np.ndarray, num_of_bds_as_row: int):
    _check_ctx(ctx)
    mat, row_shape, col_shape = _matrixize(a, num_of_bds_as_row)

    u_mat, s, v_dag_mat = np.linalg.svd(mat, full_matrices=False)
    chi = len(s)

    u = u_mat.reshape(*row_shape, chi)
    sigma = np.diag(s)
    v_dag = v_dag_mat.reshape(chi, *col_shape)
    return u, sigma, v_dag


def trunc_svd(ctx: NumpyContext, a: np.ndarray,
              num_of_bds_as_row: int,
              chi_max: int,
              s_min: Real = 0.0,
              chi_min: int = 1,
              target_trunc_err: Real = 0.0):
    """
    Supported forms:
        trunc_svd(ctx, a, k, chi_max, s_min)                                  # short form
        trunc_svd(ctx, a, k, chi_max, s_min, chi_min, target_trunc_err)       # full form

    Parameters
    ----------
    ctx : context handle.
    a : tensor to decompose.
    num_of_bds_as_row : number of leading axes grouped into the matrix row.
    chi_max : maximum number of retained singular values (positive int).
    s_min : discard singular values below this threshold (default 0.0).
    chi_min : minimum number of retained singular values (default 1).
    target_trunc_err : stop when the relative truncation error from discarding
        singular values falls at or below this value (default 0.0).

    The short form keeps all singular values ≥ s_min up to chi_max.
    The full form keeps at least chi_min singular values, at most chi_max,
    and returns the smallest number whose relative truncation error is
    ≤ target_trunc_err (subject to the s_min pre-filter).

    Returns
    -------
    u, sigma, v_dag, trunc_err
    """
    _check_ctx(ctx)

    if not isinstance(chi_max, Integral) or chi_max < 1:
        raise ValueError(
            "trunc_svd: chi_max must be a positive integer"
        )
    chi_max = int(chi_max)

    if not isinstance(s_min, Real):
        raise TypeError("trunc_svd: s_min must be real")
    if s_min < 0:
        raise ValueError("trunc_svd: s_min must be nonnegative")

    if not isinstance(chi_min, Integral) or chi_min < 1:
        raise ValueError(
            "trunc_svd: chi_min must be a positive integer"
        )
    chi_min = int(chi_min)

    if chi_min > chi_max:
        raise ValueError(
            "trunc_svd: chi_min must not exceed chi_max"
        )

    if not isinstance(target_trunc_err, Real):
        raise TypeError(
            "trunc_svd: target_trunc_err must be real"
        )
    if target_trunc_err < 0:
        raise ValueError(
            "trunc_svd: target_trunc_err must be nonnegative"
        )

    u, sigma, v_dag = svd(ctx, a, num_of_bds_as_row)
    s = np.diag(sigma)

    total_sq = float(np.sum(s ** 2))
    survivors = int(np.count_nonzero(s >= s_min))
    max_keep = min(chi_max, survivors)
    min_keep = min(chi_min, survivors)
    keep = max_keep

    for candidate in range(min_keep, max_keep + 1):
        discarded_sq = float(np.sum(s[candidate:] ** 2))
        candidate_err = (
            discarded_sq / total_sq
            if total_sq != 0.0 else 0.0
        )
        if candidate_err <= target_trunc_err:
            keep = candidate
            break

    discarded_sq = float(np.sum(s[keep:] ** 2))
    trunc_err = (
        discarded_sq / total_sq
        if total_sq != 0.0 else 0.0
    )

    return (
        u[..., :keep],
        np.diag(s[:keep]),
        v_dag[:keep, ...],
        trunc_err,
    )

def inverse(ctx: NumpyContext, a: np.ndarray,
            num_of_bds_as_row: int) -> np.ndarray:
    _check_ctx(ctx)
    mat, row_shape, col_shape = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("inverse: matricized tensor must be square")
    return np.linalg.inv(mat).reshape(*row_shape, *col_shape)


def linear_combine(ctx: NumpyContext, ins: Sequence[np.ndarray],
                   coefs: Sequence | None = None) -> np.ndarray:
    _check_ctx(ctx)

    if not ins:
        raise ValueError("linear_combine: input sequence must not be empty")

    if coefs is None:
        coefs = [1] * len(ins)

    if len(coefs) != len(ins):
        raise ValueError(
            "linear_combine: coefficient count must match input count"
        )

    shape = ins[0].shape
    dtype = ins[0].dtype
    for a in ins:
        if a.shape != shape:
            raise ValueError(
                "linear_combine: all input tensors must have same shape"
            )
        if a.dtype != dtype:
            raise TypeError(
                "linear_combine: all input tensors must have the same dtype"
            )

    if any(
        np.issubdtype(np.asarray(s).dtype, np.complexfloating)
        and not np.issubdtype(dtype, np.complexfloating)
        for s in coefs
    ):
        raise TypeError(
            "linear_combine: coefficients must be consistent with "
            "the input tensor kind"
        )

    result = np.zeros(shape, dtype=dtype)

    for a, s in zip(ins, coefs):
        result += s * a

    return result


def exp(ctx: NumpyContext, a: np.ndarray,
        num_of_bds_as_row: int) -> np.ndarray:
    _check_ctx(ctx)
    mat, row_shape, col_shape = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("exp: matricized tensor must be square")
    return expm(mat).reshape(*row_shape, *col_shape)


def qr(ctx: NumpyContext, a: np.ndarray,
       num_of_bds_as_row: int):
    _check_ctx(ctx)
    mat, row_shape, col_shape = _matrixize(a, num_of_bds_as_row)

    q_mat, r_mat = np.linalg.qr(mat, mode="reduced")
    chi = min(mat.shape)

    q = q_mat.reshape(*row_shape, chi)
    r = r_mat.reshape(chi, *col_shape)
    return q, r


def lq(ctx: NumpyContext, a: np.ndarray,
       num_of_bds_as_row: int):
    _check_ctx(ctx)
    mat, row_shape, col_shape = _matrixize(a, num_of_bds_as_row)

    qh, rh = np.linalg.qr(mat.conj().T, mode="reduced")
    l_mat = rh.conj().T
    q_mat = qh.conj().T
    chi = min(mat.shape)

    return (
        l_mat.reshape(*row_shape, chi),
        q_mat.reshape(chi, *col_shape),
    )

def eigvals(ctx: NumpyContext, a: np.ndarray,
            num_of_bds_as_row: int) -> np.ndarray:
    _check_ctx(ctx)
    mat, _, _ = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("eigvals: matricized tensor must be square")
    return np.linalg.eigvals(mat)


def eigvalsh(ctx: NumpyContext, a: np.ndarray,
             num_of_bds_as_row: int) -> np.ndarray:
    _check_ctx(ctx)
    mat, _, _ = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("eigvalsh: matricized tensor must be square")
    return np.linalg.eigvalsh(mat)


def eig(ctx: NumpyContext, a: np.ndarray,
        num_of_bds_as_row: int):
    _check_ctx(ctx)
    mat, row_shape, _ = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("eig: matricized tensor must be square")

    w, v_mat = np.linalg.eig(mat)
    return np.diag(w), v_mat.reshape(*row_shape, mat.shape[0])


def eigh(ctx: NumpyContext, a: np.ndarray,
         num_of_bds_as_row: int):
    _check_ctx(ctx)
    mat, row_shape, _ = _matrixize(a, num_of_bds_as_row)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("eigh: matricized tensor must be square")

    w, v_mat = np.linalg.eigh(mat)
    return np.diag(w), v_mat.reshape(*row_shape, mat.shape[0])