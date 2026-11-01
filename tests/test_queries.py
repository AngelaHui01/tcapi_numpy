# tests/test_queries.py
import numpy as np
import pytest
import tcapi


TEN = tcapi.TenKindF64
F32_TEN = tcapi.TenKindF32


@pytest.fixture
def ctx():
    context = tcapi.create_context(TEN)
    yield context
    tcapi.destroy_context(context)


def row_major_index(shape):
    strides = []
    stride = 1

    for dimension in reversed(shape):
        strides.insert(0, stride)
        stride *= dimension

    def coors_to_index(coors):
        return sum(coord * stride for coord, stride in zip(coors, strides))

    return coors_to_index


def tensor_from_values(ctx, shape, values, ten_kind=TEN):
    return tcapi.assign_from_range(
        ctx,
        ten_kind,
        shape,
        values,
        row_major_index(shape),
    )


class TestQueries:
    # --- order ---
    def test_order_scalar(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 5.0)

        assert tcapi.order(ctx, a) == 0

    def test_order_vector(self, ctx):
        a = tcapi.fill(ctx, TEN, (4,), 1.0)

        assert tcapi.order(ctx, a) == 1

    def test_order_matrix(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 4), 1.0)

        assert tcapi.order(ctx, a) == 2

    def test_order_higher(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 3, 4, 5), 1.0)

        assert tcapi.order(ctx, a) == 4

    # --- shape ---
    def test_shape_vector(self, ctx):
        a = tcapi.fill(ctx, TEN, (6,), 1.0)

        assert tuple(tcapi.shape(ctx, a)) == (6,)

    def test_shape_matrix(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 4), 1.0)

        assert tuple(tcapi.shape(ctx, a)) == (3, 4)

    def test_shape_third_order(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 3, 4), 1.0)

        assert tuple(tcapi.shape(ctx, a)) == (2, 3, 4)

    # --- size ---
    def test_size_vector(self, ctx):
        a = tcapi.fill(ctx, TEN, (6,), 1.0)

        assert tcapi.size(ctx, a) == 6

    def test_size_matrix(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 4), 1.0)

        assert tcapi.size(ctx, a) == 12

    # --- size_bytes ---
    def test_size_bytes_f64_dense_tensor(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 4), 1.0)

        assert tcapi.size_bytes(ctx, a) == 12 * 8

    def test_size_bytes_f32_dense_tensor(self, ctx):
        a = tcapi.fill(ctx, F32_TEN, (3, 4), 1.0)

        assert tcapi.size_bytes(ctx, a) == 12 * 4

    # --- get_elem ---
    def test_get_elem_vector(self, ctx):
        a = tensor_from_values(ctx, (3,), [10.0, 20.0, 30.0])

        assert tcapi.get_elem(ctx, a, (2,)) == 30.0

    def test_get_elem_matrix(self, ctx):
        a = tensor_from_values(ctx, (2, 3), range(6))

        assert tcapi.get_elem(ctx, a, (1, 2)) == 5.0

    def test_get_elem_third_order(self, ctx):
        a = tensor_from_values(ctx, (2, 3, 4), range(24))

        assert tcapi.get_elem(ctx, a, (1, 2, 3)) == 23.0


class TestZerothOrderQueries:
    def test_order(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 1.0)
        assert tcapi.order(ctx, a) == 0

    def test_shape(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 1.0)
        assert tcapi.shape(ctx, a) == ()

    def test_size(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 1.0)
        assert tcapi.size(ctx, a) == 1

    def test_get_elem(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 3.14)
        assert tcapi.get_elem(ctx, a, ()) == 3.14

    def test_set_elem(self, ctx):
        a = tcapi.zeros(ctx, TEN, ())
        b = tcapi.set_elem(ctx, a, (), 99.0)
        assert tcapi.get_elem(ctx, b, ()) == 99.0