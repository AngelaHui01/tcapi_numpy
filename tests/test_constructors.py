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