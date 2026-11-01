# tests/test_constructors.py
import numpy as np
import pytest
import tcapi


@pytest.fixture
def ctx():
    """Provide a fresh NumPy TCAPI context for each test."""
    c = tcapi.create_context(tcapi.TenKindF64)
    yield c
    tcapi.destroy_context(c)


class TestConstructors:
    def test_allocate_shape_and_dtype(self, ctx):
        a = tcapi.allocate(ctx, tcapi.TenKindF64, (2, 3))

        assert a.shape == (2, 3)
        assert a.dtype == np.dtype(np.float64)
        a[0, 0] = 5.0
        assert a[0, 0] == 5.0

    def test_zeros_shape_and_dtype(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, (3, 4))

        assert a.shape == (3, 4)
        assert a.dtype == np.dtype(np.float64)
        assert np.all(a == 0)

    def test_fill_value(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF32, (2, 2), 7.0)

        assert np.all(a == 7.0)
        assert a.dtype == np.dtype(np.float32)

    def test_eye_is_identity(self, ctx):
        identity = tcapi.eye(ctx, tcapi.TenKindF64, 4)

        assert np.array_equal(identity, np.eye(4))

    def test_copy_is_independent(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, (3,))
        b = tcapi.copy(ctx, a)

        b[0] = 99.0

        assert a[0] == 0.0
        assert b[0] == 99.0
        assert not np.shares_memory(a, b)

    def test_set_and_get_elem(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, (3, 3))

        a = tcapi.set_elem(ctx, a, (1, 2), 42.0)

        assert tcapi.get_elem(ctx, a, (1, 2)) == 42.0


class TestZerothOrder:
    def test_order_zero(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, ())
        assert tcapi.order(ctx, a) == 0

    def test_shape_empty(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, ())
        assert tcapi.shape(ctx, a) == ()

    def test_size_one(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, ())
        assert tcapi.size(ctx, a) == 1

    def test_allocate_zeroth(self, ctx):
        a = tcapi.allocate(ctx, tcapi.TenKindF64, ())
        assert a.shape == ()
        assert a.dtype == np.dtype(np.float64)

    def test_zeros_zeroth(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, ())
        assert tcapi.get_elem(ctx, a, ()) == 0.0

    def test_fill_zeroth(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (), 7.5)
        assert tcapi.get_elem(ctx, a, ()) == 7.5

    def test_random_zeroth(self, ctx):
        call_count = 0

        def gen():
            nonlocal call_count
            call_count += 1
            return 3.14

        a = tcapi.random(ctx, tcapi.TenKindF64, (), gen)
        assert call_count == 1
        assert tcapi.get_elem(ctx, a, ()) == 3.14

    def test_assign_from_range_zeroth(self, ctx):
        a = tcapi.assign_from_range(
            ctx, tcapi.TenKindF64, (), [99.0],
            lambda coors: 0 if coors == () else -1,
        )
        assert tcapi.get_elem(ctx, a, ()) == 99.0

    def test_set_elem_zeroth(self, ctx):
        a = tcapi.zeros(ctx, tcapi.TenKindF64, ())
        a = tcapi.set_elem(ctx, a, (), 42.0)
        assert tcapi.get_elem(ctx, a, ()) == 42.0

    def test_get_elem_zeroth(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (), -1.5)
        assert tcapi.get_elem(ctx, a, ()) == -1.5

    def test_reshape_zeroth_to_vector(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (), 5.0)
        b = tcapi.reshape(ctx, a, (1,))
        assert tcapi.shape(ctx, b) == (1,)

    def test_reshape_vector_to_zeroth(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (1,), 5.0)
        b = tcapi.reshape(ctx, a, ())
        assert tcapi.shape(ctx, b) == ()
        assert tcapi.get_elem(ctx, b, ()) == 5.0

    def test_reshape_zeroth_to_1x1(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (), 3.0)
        b = tcapi.reshape(ctx, a, (1, 1))
        assert tcapi.shape(ctx, b) == (1, 1)
        assert tcapi.get_elem(ctx, b, (0, 0)) == 3.0