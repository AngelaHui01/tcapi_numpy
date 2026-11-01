import numpy as np
from typing import Callable, Mapping, Sequence
from numbers import Integral
from .context import NumpyContext
from .misc import _check_ctx


def set_elem(ctx: NumpyContext, a: np.ndarray,
             coors: Sequence[int], el) -> np.ndarray:
    _check_ctx(ctx)
    if len(coors) != a.ndim:
        raise ValueError(
            f"set_elem: expected {a.ndim} coordinates, got {len(coors)}"
        )

    out = a.copy()
    out[tuple(coors)] = el
    return out


def transpose(ctx: NumpyContext, a: np.ndarray,
              new_order: Sequence[int]) -> np.ndarray:
    _check_ctx(ctx)
    if len(new_order) != a.ndim:
        raise ValueError(
            f"transpose: expected {a.ndim} axes, got {len(new_order)}"
        )
    if sorted(new_order) != list(range(a.ndim)):
        raise ValueError(
            "transpose: axes must be a permutation of the tensor dimensions"
        )
    return np.transpose(a, tuple(new_order)).copy()

def reshape(ctx: NumpyContext, a: np.ndarray,
            new_shape: Sequence[int]) -> np.ndarray:
    _check_ctx(ctx)
    if any(not isinstance(dim, Integral) or dim < 0
           for dim in new_shape):
        raise ValueError(
            "reshape: all dimensions must be nonnegative integers"
        )

    new_shape = tuple(int(dim) for dim in new_shape)
    if int(np.prod(new_shape)) != a.size:
        raise ValueError(
            f"reshape: total size mismatch. Original size {a.size} "
            f"vs new shape {new_shape}"
        )
    return a.reshape(new_shape).copy()


def cplx_conj(ctx: NumpyContext, a: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    return np.conjugate(a).copy()


def to_cplx(ctx: NumpyContext, a: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    if np.issubdtype(a.dtype, np.complexfloating):
        return a.astype(a.dtype, copy=True)
    return a.astype(np.result_type(a.dtype, np.complex64), copy=True)


def real(ctx: NumpyContext, a: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    return np.real(a).copy()


def imag(ctx: NumpyContext, a: np.ndarray) -> np.ndarray:
    _check_ctx(ctx)
    return np.imag(a).copy()


def expand(ctx: NumpyContext, a: np.ndarray,
           bond_idx_increment_map: Mapping[int, int]) -> np.ndarray:
    _check_ctx(ctx)
    if a.ndim == 0:
        raise ValueError("expand: zeroth-order tensor has no bonds to expand")

    new_shape = list(a.shape)
    for bond_idx, increment in bond_idx_increment_map.items():
        if (not isinstance(bond_idx, Integral)
                or not 0 <= bond_idx < a.ndim):
            raise ValueError("expand: bond index out of range")

        if not isinstance(increment, Integral) or increment < 0:
            raise ValueError(
                "expand: increments must be nonnegative integers"
            )

        new_shape[int(bond_idx)] += int(increment)

    out = np.zeros(tuple(new_shape), dtype=a.dtype)
    out[tuple(slice(0, dim) for dim in a.shape)] = a
    return out


def shrink(ctx: NumpyContext, a: np.ndarray,
           bd_idx_el_coor_pair_map: Mapping[int, Sequence[int]]) -> np.ndarray:
    _check_ctx(ctx)
    if a.ndim == 0:
        raise ValueError("shrink: zeroth-order tensor has no bonds to shrink")

    selectors = [slice(None)] * a.ndim
    for bond_idx, bounds in bd_idx_el_coor_pair_map.items():
        if (not isinstance(bond_idx, Integral)
                or not 0 <= bond_idx < a.ndim):
            raise ValueError("shrink: bond index out of range")

        if len(bounds) != 2:
            raise ValueError("shrink: each range must be (start, stop)")

        start, stop = bounds
        if (not isinstance(start, Integral)
                or not isinstance(stop, Integral)
                or not 0 <= start <= stop <= a.shape[bond_idx]):
            raise ValueError("shrink: invalid coordinate range")

        selectors[int(bond_idx)] = slice(int(start), int(stop))

    return a[tuple(selectors)].copy()


def extract_sub(ctx: NumpyContext, a: np.ndarray,
                coor_pairs: Sequence[Sequence[int]]) -> np.ndarray:
    _check_ctx(ctx)
    if a.ndim == 0:
        raise ValueError("extract_sub: zeroth-order tensor has no bonds to extract")

    if len(coor_pairs) != a.ndim:
        raise ValueError(
            f"extract_sub: expected {a.ndim} coordinate pairs, "
            f"got {len(coor_pairs)}"
        )

    selectors = []
    for axis, bounds in enumerate(coor_pairs):
        if len(bounds) != 2:
            raise ValueError(
                "extract_sub: each entry must be (start, stop)"
            )

        start, stop = bounds
        if not (0 <= start <= stop <= a.shape[axis]):
            raise ValueError("extract_sub: invalid coordinate range")
        selectors.append(slice(start, stop))

    return a[tuple(selectors)].copy()


def replace_sub(ctx: NumpyContext, a: np.ndarray, sub: np.ndarray,
                begin_pt: Sequence[int]) -> np.ndarray:
    _check_ctx(ctx)
    if a.ndim == 0:
        raise ValueError("replace_sub: zeroth-order tensor has no bonds to replace")

    if len(begin_pt) != a.ndim or sub.ndim != a.ndim:
        raise ValueError(
            "replace_sub: tensors and begin_pt must have matching rank"
        )

    stops = tuple(
        start + extent for start, extent in zip(begin_pt, sub.shape)
    )
    if any(
        start < 0 or stop > dim
        for start, stop, dim in zip(begin_pt, stops, a.shape)
    ):
        raise ValueError("replace_sub: sub-tensor does not fit in target")

    out = a.copy()
    out[
        tuple(slice(start, stop) for start, stop in zip(begin_pt, stops))
    ] = sub
    return out


def concatenate(ctx: NumpyContext, ins: Sequence[np.ndarray],
                concat_bdidx: int) -> np.ndarray:
    _check_ctx(ctx)
    if not ins:
        raise ValueError("concatenate: tensor list must not be empty")
    if ins[0].ndim == 0:
        raise ValueError("concatenate: zeroth-order tensor has no bonds to concatenate")
    if not (0 <= concat_bdidx < ins[0].ndim):
        raise ValueError("concatenate: concat_bdidx is out of range")

    expected_ndim = ins[0].ndim
    dtype = ins[0].dtype
    for tensor in ins:
        if tensor.ndim != expected_ndim:
            raise ValueError(
                "concatenate: all tensors must have the same rank"
            )
        if tensor.dtype != dtype:
            raise TypeError(
                "concatenate: all input tensors must have the same dtype"
            )

    try:
        return np.concatenate(ins, axis=concat_bdidx)
    except ValueError as exc:
        raise ValueError(
            "concatenate: tensors have incompatible shapes "
            "for concatenation"
        ) from exc


def stack(ctx: NumpyContext, ins: Sequence[np.ndarray],
          stack_bdidx: int) -> np.ndarray:
    _check_ctx(ctx)
    if not ins:
        raise ValueError("stack: tensor list must not be empty")
    if ins[0].ndim == 0:
        raise ValueError("stack: zeroth-order tensor has no bonds to stack")
    if not (0 <= stack_bdidx <= ins[0].ndim):
        raise ValueError("stack: stack_bdidx is out of range")

    dtype = ins[0].dtype
    for tensor in ins:
        if tensor.dtype != dtype:
            raise TypeError(
                "stack: all input tensors must have the same dtype"
            )

    try:
        return np.stack(ins, axis=stack_bdidx)
    except ValueError as exc:
        raise ValueError(
            "stack: tensors have incompatible shapes for stacking"
        ) from exc


def for_each(ctx: NumpyContext, a: np.ndarray,
            f: Callable) -> np.ndarray:
    _check_ctx(ctx)

    out = a.copy()
    for coors in np.ndindex(a.shape):
        replacement = f(a[coors])
        if replacement is not None:
            out[coors] = replacement
    return out


def for_each_with_coors(ctx: NumpyContext, a: np.ndarray,
                       f: Callable) -> np.ndarray:
    _check_ctx(ctx)

    out = a.copy()
    for coors in np.ndindex(a.shape):
        replacement = f(a[coors], coors)
        if replacement is not None:
            out[coors] = replacement
    return out