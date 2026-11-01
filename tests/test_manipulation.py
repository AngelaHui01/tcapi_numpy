# tests/test_manipulation.py
import numpy as np
import pytest
import tcapi


TEN_F64 = tcapi.TenKindF64
TEN_F32 = tcapi.TenKindF32


@pytest.fixture
def ctx():
    context = tcapi.create_context(TEN_F64)
    yield context
    tcapi.destroy_context(context)


def tensor_from_values(ctx, shape, values, ten_kind=TEN_F64):
    """Build a row-major tensor from a flat Python sequence."""
    strides = np.cumprod((1,) + tuple(shape[:0:-1]))[::-1]

    return tcapi.assign_from_range(
        ctx,
        ten_kind,
        shape,
        values,
        lambda coors: sum(index * stride for index, stride in zip(coors, strides)),
    )


class TestManipulation:
    def test_transpose_matrix(self, ctx):
        a = tensor_from_values(ctx, (2, 3), range(6))

        b = tcapi.transpose(ctx, a, (1, 0))

        assert tuple(tcapi.shape(ctx, b)) == (3, 2)
        assert np.allclose(b, np.arange(6, dtype=np.float64).reshape(2, 3).T)

    def test_transpose_third_order(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3, 4), 1.0)

        b = tcapi.transpose(ctx, a, (2, 0, 1))

        assert tuple(tcapi.shape(ctx, b)) == (4, 2, 3)

    def test_transpose_identity_perm(self, ctx):
        a = tensor_from_values(ctx, (2, 3, 4), range(24))

        b = tcapi.transpose(ctx, a, (0, 1, 2))

        assert np.allclose(b, a)

    def test_reshape_basic(self, ctx):
        a = tensor_from_values(ctx, (12,), range(12))

        b = tcapi.reshape(ctx, a, (3, 4))

        assert tuple(tcapi.shape(ctx, b)) == (3, 4)

    def test_reshape_preserves_elements(self, ctx):
        a = tensor_from_values(ctx, (2, 3, 4), range(24))

        b = tcapi.reshape(ctx, a, (6, 4))

        assert np.allclose(b, np.arange(24, dtype=np.float64).reshape(6, 4))

    def test_reshape_size_mismatch_raises(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (3, 4), 1.0)

        with pytest.raises(ValueError):
            tcapi.reshape(ctx, a, (3, 5))

    def test_normalize_unit_norm(self, ctx):
        a = tensor_from_values(ctx, (2,), [3.0, 4.0])

        normalized, original_norm = tcapi.normalize(ctx, a)

        assert np.isclose(original_norm, 5.0)
        assert np.isclose(tcapi.norm(ctx, normalized), 1.0)

    def test_normalize_already_unit(self, ctx):
        a = tensor_from_values(ctx, (3,), [1.0, 0.0, 0.0])

        normalized, original_norm = tcapi.normalize(ctx, a)

        assert np.isclose(original_norm, 1.0)
        assert np.isclose(tcapi.norm(ctx, normalized), 1.0)

    def test_normalize_zero_raises(self, ctx):
        a = tcapi.zeros(ctx, TEN_F64, (3,))

        with pytest.raises(ValueError):
            tcapi.normalize(ctx, a)

    # for_each is the supported elementwise mapping API.
    def test_broadcast_sqrt(self, ctx):
        a = tensor_from_values(ctx, (3,), [4.0, 9.0, 16.0])

        b = tcapi.for_each(ctx, a, np.sqrt)

        assert np.allclose(b, [2.0, 3.0, 4.0])

    def test_broadcast_preserves_shape(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3, 4), 1.0)

        b = tcapi.for_each(ctx, a, lambda value: value * 2.0)

        assert tuple(tcapi.shape(ctx, b)) == (2, 3, 4)
        assert np.allclose(b, 2.0)

    def test_broadcast_preserves_dtype(self, ctx):
        a = tcapi.fill(ctx, TEN_F32, (3,), 1.0)

        b = tcapi.for_each(ctx, a, lambda value: value + 1.0)

        assert b.dtype == np.dtype(np.float32)

    def test_foreach_accumulate(self, ctx):
        values = tensor_from_values(ctx, (4,), [1.0, 2.0, 3.0, 4.0])
        total = 0.0

        def add(value):
            nonlocal total
            total += value
            return None

        result = tcapi.for_each(ctx, values, add)

        assert np.isclose(total, 10.0)
        assert np.allclose(result, values)

    def test_foreach_count_elements(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 1.0)
        count = 0

        def counter(_):
            nonlocal count
            count += 1
            return None

        tcapi.for_each(ctx, a, counter)

        assert count == 6

    def test_foreach_with_coors_collects(self, ctx):
        a = tensor_from_values(ctx, (2, 2), [1.0, 2.0, 3.0, 4.0])
        collected = {}

        def collect(value, coors):
            collected[tuple(coors)] = value
            return None

        tcapi.for_each_with_coors(ctx, a, collect)

        assert collected[(0, 0)] == 1.0
        assert collected[(1, 1)] == 4.0

    def test_concat_along_axis(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 1.0)
        b = tcapi.fill(ctx, TEN_F64, (4, 3), 2.0)

        c = tcapi.concatenate(ctx, [a, b], 0)

        assert c.shape == (6, 3)
        assert np.allclose(c[:2], 1.0)
        assert np.allclose(c[2:], 2.0)

    def test_stack_along_axis(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 1.0)
        b = tcapi.fill(ctx, TEN_F64, (2, 3), 2.0)

        c = tcapi.stack(ctx, [a, b], 0)

        assert c.shape == (2, 2, 3)
        assert np.allclose(c[0], 1.0)
        assert np.allclose(c[1], 2.0)

    def test_concat_mixed_dtype_raises(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 1.0)
        b = tcapi.fill(ctx, TEN_F32, (2, 3), 1.0)

        with pytest.raises(TypeError):
            tcapi.concatenate(ctx, [a, b], 0)

    def test_stack_mixed_dtype_raises(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 1.0)
        b = tcapi.fill(ctx, TEN_F32, (2, 3), 1.0)

        with pytest.raises(TypeError):
            tcapi.stack(ctx, [a, b], 0)

    def test_linear_combine(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2,), 1.0)
        b = tcapi.fill(ctx, TEN_F64, (2,), 3.0)

        c = tcapi.linear_combine(ctx, [a, b], [2.0, -1.0])

        assert np.allclose(c, -1.0)

    def test_linear_combine_mixed_dtype_raises(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2,), 1.0)
        b = tcapi.fill(ctx, TEN_F32, (2,), 3.0)

        with pytest.raises(TypeError):
            tcapi.linear_combine(ctx, [a, b], [2.0, -1.0])


TEN_C128 = tcapi.TenKindC128


class TestComplexOps:
    def test_cplx_conj_conjugates(self, ctx):
        a = tcapi.fill(ctx, TEN_C128, (2, 2), 1 + 2j)

        b = tcapi.cplx_conj(ctx, a)

        assert np.allclose(b, 1 - 2j)
        assert not np.shares_memory(a, b)

    def test_to_cplx_embeds_real(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 2), 1.5)

        b = tcapi.to_cplx(ctx, a)

        assert b.dtype == np.dtype(np.complex128)
        assert np.allclose(b, 1.5 + 0j)
        assert not np.shares_memory(a, b)

    def test_to_cplx_complex_returns_deep_copy(self, ctx):
        a = tcapi.fill(ctx, TEN_C128, (2,), 1 + 2j)

        b = tcapi.to_cplx(ctx, a)

        assert np.array_equal(b, a)
        assert not np.shares_memory(a, b)

    def test_real_of_complex(self, ctx):
        a = tcapi.fill(ctx, TEN_C128, (2,), 1 + 2j)

        b = tcapi.real(ctx, a)

        assert b.dtype == np.dtype(np.float64)
        assert np.allclose(b, 1.0)
        assert not np.shares_memory(a, b)

    def test_imag_of_complex(self, ctx):
        a = tcapi.fill(ctx, TEN_C128, (2,), 1 + 2j)

        b = tcapi.imag(ctx, a)

        assert b.dtype == np.dtype(np.float64)
        assert np.allclose(b, 2.0)

    def test_real_of_real_returns_deep_copy(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2,), 3.0)

        b = tcapi.real(ctx, a)

        assert np.array_equal(b, a)
        assert not np.shares_memory(a, b)

    def test_imag_of_real_is_zero(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2,), 3.0)

        b = tcapi.imag(ctx, a)

        assert b.dtype == np.dtype(np.float64)
        assert np.all(b == 0.0)


class TestStructuralOps:
    def test_expand_pads_with_zeros(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 2), 5.0)

        b = tcapi.expand(ctx, a, {1: 2})

        assert b.shape == (2, 4)
        assert np.all(b[:, :2] == 5.0)
        assert np.all(b[:, 2:] == 0.0)

    def test_expand_multiple_bonds(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 1, 2), 1.0)

        b = tcapi.expand(ctx, a, {0: 1, 2: 3})

        assert b.shape == (3, 1, 5)
        assert np.all(b[:2, 0, :2] == 1.0)
        assert np.all(b[2:, ...] == 0.0)
        assert np.all(b[:, 0, 2:] == 0.0)

    def test_shrink_to_ranges(self, ctx):
        a = tensor_from_values(ctx, (3, 4, 2), range(24))

        b = tcapi.shrink(ctx, a, {1: (1, 3)})

        assert b.shape == (3, 2, 2)
        assert np.array_equal(b, a[:, 1:3, :])

    def test_shrink_multiple_bonds(self, ctx):
        a = tensor_from_values(ctx, (3, 4, 5), range(60))

        b = tcapi.shrink(ctx, a, {0: (1, 3), 2: (0, 2)})

        assert b.shape == (2, 4, 2)
        assert np.array_equal(b, a[1:3, :, 0:2])

    def test_extract_sub(self, ctx):
        a = tensor_from_values(ctx, (4, 4), range(16))

        sub = tcapi.extract_sub(ctx, a, ((1, 3), (2, 4)))

        assert np.array_equal(sub, a[1:3, 2:4])

    def test_replace_sub(self, ctx):
        a = tcapi.zeros(ctx, TEN_F64, (4, 4))
        sub = tcapi.fill(ctx, TEN_F64, (2, 2), 9.0)

        b = tcapi.replace_sub(ctx, a, sub, (1, 2))

        expected = np.zeros((4, 4))
        expected[1:3, 2:4] = 9.0
        assert np.array_equal(b, expected)


class TestInputImmutability:
    """Verify that every TCAPI routine leaves its input tensors unmodified."""

    def snapshot(self, *tensors):
        return [np.copy(t) for t in tensors]

    def test_transpose_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (2, 3), range(6),
                                     lambda c: 3 * c[0] + c[1])
        saved = self.snapshot(a)
        _ = tcapi.transpose(ctx, a, (1, 0))
        assert np.array_equal(a, saved[0])

    def test_reshape_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (12,), range(12),
                                     lambda c: c[0])
        saved = self.snapshot(a)
        _ = tcapi.reshape(ctx, a, (3, 4))
        assert np.array_equal(a, saved[0])

    def test_scale_does_not_mutate(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (3,), 2.0)
        saved = self.snapshot(a)
        _ = tcapi.scale(ctx, a, 3.0)
        assert np.array_equal(a, saved[0])

    def test_normalize_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (2,), [3.0, 4.0],
                                     lambda c: c[0])
        saved = self.snapshot(a)
        _, _ = tcapi.normalize(ctx, a)
        assert np.array_equal(a, saved[0])

    def test_expand_does_not_mutate(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 2), 5.0)
        saved = self.snapshot(a)
        _ = tcapi.expand(ctx, a, {1: 2})
        assert np.array_equal(a, saved[0])

    def test_shrink_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (3, 4), range(12),
                                     lambda c: 4 * c[0] + c[1])
        saved = self.snapshot(a)
        _ = tcapi.shrink(ctx, a, {1: (1, 3)})
        assert np.array_equal(a, saved[0])

    def test_extract_sub_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (4, 4), range(16),
                                     lambda c: 4 * c[0] + c[1])
        saved = self.snapshot(a)
        _ = tcapi.extract_sub(ctx, a, ((1, 3), (2, 4)))
        assert np.array_equal(a, saved[0])

    def test_replace_sub_does_not_mutate(self, ctx):
        a = tcapi.zeros(ctx, TEN_F64, (4, 4))
        sub = tcapi.fill(ctx, TEN_F64, (2, 2), 9.0)
        saved = self.snapshot(a, sub)
        _ = tcapi.replace_sub(ctx, a, sub, (1, 2))
        assert np.array_equal(a, saved[0])
        assert np.array_equal(sub, saved[1])

    def test_for_each_does_not_mutate(self, ctx):
        a = tcapi.fill(ctx, TEN_F64, (2, 3), 5.0)
        saved = self.snapshot(a)
        _ = tcapi.for_each(ctx, a, lambda v: v * 2.0)
        assert np.array_equal(a, saved[0])

    def test_for_each_with_coors_does_not_mutate(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (2, 2), [1.0, 2.0, 3.0, 4.0],
                                     lambda c: 2 * c[0] + c[1])
        saved = self.snapshot(a)
        _ = tcapi.for_each_with_coors(ctx, a, lambda v, c: v)
        assert np.array_equal(a, saved[0])

    def test_set_elem_does_not_mutate(self, ctx):
        a = tcapi.zeros(ctx, TEN_F64, (3, 3))
        saved = self.snapshot(a)
        _ = tcapi.set_elem(ctx, a, (1, 2), 42.0)
        assert np.array_equal(a, saved[0])

    def test_copy_returns_deep_copy(self, ctx):
        a = tcapi.assign_from_range(ctx, TEN_F64, (3,), range(3),
                                     lambda c: c[0])
        b = tcapi.copy(ctx, a)
        assert not np.shares_memory(a, b)