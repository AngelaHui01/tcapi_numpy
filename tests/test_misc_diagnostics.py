import numpy as np
import pytest
import tcapi


TEN = tcapi.TenKindF64
CPLX_TEN = tcapi.TenKindC128
F32_TEN = tcapi.TenKindF32


@pytest.fixture
def ctx():
    context = tcapi.create_context(TEN)
    yield context
    tcapi.destroy_context(context)


def row_major_index(shape):
    """Return a coordinate-to-row-major-index mapping."""
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


class TestVersion:
    def test_returns_string(self):
        assert isinstance(tcapi.version(TEN), str)

    def test_format_major_minor(self):
        parts = tcapi.version(TEN).split(".")
        assert len(parts) == 2
        assert all(part.isdigit() for part in parts)

    def test_is_1_0(self):
        assert tcapi.version(TEN) == "1.0"


class TestShow:
    def test_show_runs_and_emits_output(self, ctx, capsys):
        a = tcapi.fill(ctx, TEN, (2, 3), 5.0)

        tcapi.show(ctx, a)

        assert capsys.readouterr().out

    def test_show_scalar_tensor_runs(self, ctx, capsys):
        a = tcapi.fill(ctx, TEN, (), 3.14)

        tcapi.show(ctx, a)

        assert capsys.readouterr().out


class TestClose:
    def test_identical_tensors(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 3), 1.0)

        assert tcapi.close(ctx, a, a, 0.0) is True

    def test_close_within_eps(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 3), 5.0)
        b = tcapi.fill(ctx, TEN, (2, 3), 5.001)

        assert tcapi.close(ctx, a, b, 0.01) is True

    def test_not_close_tight_eps(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 3), 5.0)
        b = tcapi.fill(ctx, TEN, (2, 3), 5.001)

        assert tcapi.close(ctx, a, b, 1e-6) is False

    def test_exact_boundary(self, ctx):
        a = tcapi.fill(ctx, TEN, (2,), 0.0)
        b = tcapi.fill(ctx, TEN, (2,), 1e-4)

        assert tcapi.close(ctx, a, b, 1e-4) is True

    def test_shape_mismatch_returns_false(self, ctx):
        a = tcapi.zeros(ctx, TEN, (2, 3))
        b = tcapi.zeros(ctx, TEN, (3, 2))

        assert tcapi.close(ctx, a, b, 1.0) is False

    def test_negative_epsilon_raises(self, ctx):
        a = tcapi.zeros(ctx, TEN, (2,))

        with pytest.raises(ValueError):
            tcapi.close(ctx, a, a, -1e-6)

    def test_complex_tensors(self, ctx):
        a = tensor_from_values(ctx, (2,), [1 + 2j, 3 + 4j], CPLX_TEN)
        b = tensor_from_values(ctx, (2,), [1 + 2j, 3 + 4j], CPLX_TEN)

        assert tcapi.close(ctx, a, b, 0.0) is True

    def test_copy_is_close(self, ctx):
        rng = np.random.default_rng(42)
        a = tcapi.random(ctx, TEN, (4, 4), rng.random)
        b = tcapi.copy(ctx, a)

        assert tcapi.close(ctx, a, b, 0.0) is True


class TestConvert:
    def test_f64_to_f32(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 3), 2.0)

        b = tcapi.convert(ctx, a, ctx, F32_TEN)

        assert b.dtype == np.dtype(np.float32)

    def test_f64_to_c128(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 3), 2.0)

        b = tcapi.convert(ctx, a, ctx, CPLX_TEN)

        assert b.dtype == np.dtype(np.complex128)

    def test_values_preserved_on_promotion(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 2), 5.0)

        b = tcapi.convert(ctx, a, ctx, CPLX_TEN)

        assert np.allclose(b.real, 5.0)
        assert np.allclose(b.imag, 0.0)

    def test_same_kind_is_deep_copy(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 2), 3.0)

        b = tcapi.convert(ctx, a, ctx, TEN)
        modified_b = tcapi.set_elem(ctx, b, (0, 0), 99.0)

        assert np.array_equal(a, b)
        assert tcapi.get_elem(ctx, modified_b, (0, 0)) == 99.0
        assert tcapi.get_elem(ctx, a, (0, 0)) == 3.0

    def test_f32_to_f64_promotion(self, ctx):
        a = tcapi.fill(ctx, F32_TEN, (3,), 1.0)

        b = tcapi.convert(ctx, a, ctx, TEN)

        assert b.dtype == np.dtype(np.float64)

    def test_f32_to_f64_values_preserved(self, ctx):
        a = tcapi.fill(ctx, F32_TEN, (3,), np.float32(3.14159))
        b = tcapi.convert(ctx, a, ctx, TEN)
        assert b.dtype == np.dtype(np.float64)
        assert np.allclose(a, b.astype(np.float32))

    def test_f64_to_f32_values_preserved(self, ctx):
        a = tcapi.fill(ctx, TEN, (3,), 2.71828)
        b = tcapi.convert(ctx, a, ctx, F32_TEN)
        assert b.dtype == np.dtype(np.float32)
        assert np.allclose(a, b.astype(np.float64), atol=1e-6)

    def test_convert_ten_kind_correct(self, ctx):
        a = tcapi.fill(ctx, TEN, (2,), 1.0)
        b = tcapi.convert(ctx, a, ctx, F32_TEN)
        assert tcapi.ten_kind(ctx, b) == F32_TEN

    def test_convert_complex_to_real(self, ctx):
        a = tcapi.fill(ctx, CPLX_TEN, (3,), 1 + 0j)
        b = tcapi.convert(ctx, a, ctx, TEN)
        assert b.dtype == np.dtype(np.float64)
        assert np.allclose(b, 1.0)

    def test_convert_real_to_complex_ten_kind(self, ctx):
        a = tcapi.fill(ctx, TEN, (3,), 1.0)
        b = tcapi.convert(ctx, a, ctx, CPLX_TEN)
        assert tcapi.ten_kind(ctx, b) == CPLX_TEN


class TestToRange:
    def test_roundtrip_with_assign_from_range(self, ctx):
        shape = (2, 3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        coors_to_index = row_major_index(shape)

        a = tcapi.assign_from_range(
            ctx,
            TEN,
            shape,
            values,
            coors_to_index,
        )

        output = tcapi.to_range(ctx, a, coors_to_index)

        assert output == pytest.approx(values)

    def test_length_equals_size(self, ctx):
        a = tcapi.zeros(ctx, TEN, (3, 4))

        output = tcapi.to_range(ctx, a, row_major_index((3, 4)))

        assert len(output) == tcapi.size(ctx, a)

    def test_values_correct(self, ctx):
        a = tensor_from_values(ctx, (2, 3), [0, 1, 2, 3, 4, 5])

        output = tcapi.to_range(ctx, a, row_major_index((2, 3)))

        assert output == pytest.approx([0, 1, 2, 3, 4, 5])

    def test_bad_index_raises(self, ctx):
        a = tcapi.zeros(ctx, TEN, (2, 2))

        with pytest.raises(IndexError):
            tcapi.to_range(ctx, a, lambda _: 999)


class TestZerothOrderMisc:
    def test_to_range_zeroth(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 42.0)
        output = tcapi.to_range(ctx, a, lambda coors: 0)
        assert len(output) == 1
        assert output[0] == pytest.approx(42.0)

    def test_assign_from_range_zeroth(self, ctx):
        values = [7.0]
        a = tcapi.assign_from_range(
            ctx, TEN, (), values,
            lambda coors: 0 if coors == () else -1,
        )
        assert tcapi.get_elem(ctx, a, ()) == 7.0

    def test_for_each_zeroth(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 10.0)
        call_count = 0

        def f(val):
            nonlocal call_count
            call_count += 1
            return val + 1.0

        b = tcapi.for_each(ctx, a, f)
        assert call_count == 1
        assert tcapi.get_elem(ctx, b, ()) == 11.0

    def test_for_each_with_coors_zeroth(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 5.0)
        recorded = {}

        def f(val, coors):
            recorded[coors] = val
            return None

        tcapi.for_each_with_coors(ctx, a, f)
        assert recorded == {(): 5.0}

    def test_convert_preserves_zeroth(self, ctx):
        a = tcapi.fill(ctx, TEN, (), 3.14)
        b = tcapi.convert(ctx, a, ctx, F32_TEN)
        assert b.shape == ()
        assert b.dtype == np.dtype(np.float32)
        assert np.isclose(float(b), 3.14, atol=1e-6)


class TestAssignFromRange:
    def test_basic(self, ctx):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        a = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 3),
            values,
            row_major_index((2, 3)),
        )

        assert tcapi.get_elem(ctx, a, (1, 1)) == pytest.approx(5.0)
        assert tcapi.get_elem(ctx, a, (0, 2)) == pytest.approx(3.0)

    def test_dtype_from_ten_kind(self, ctx):
        a = tensor_from_values(ctx, (3,), [1.0, 2.0, 3.0], F32_TEN)

        assert a.dtype == np.dtype(np.float32)

    def test_shape_correct(self, ctx):
        a = tensor_from_values(ctx, (3, 4), range(12))

        assert tuple(tcapi.shape(ctx, a)) == (3, 4)


class TestVerbose:
    def test_verbose_0_silent(self, ctx, capsys, monkeypatch):
        monkeypatch.setenv("TCAPI_VERBOSE", "0")

        tcapi.zeros(ctx, TEN, (2, 2))

        assert capsys.readouterr().out == ""

    def test_verbose_1_reports_call(self, ctx, capsys, monkeypatch):
        monkeypatch.setenv("TCAPI_VERBOSE", "1")

        tcapi.zeros(ctx, TEN, (2, 2))

        assert capsys.readouterr().out

    def test_verbose_2_reports_call(self, ctx, capsys, monkeypatch):
        monkeypatch.setenv("TCAPI_VERBOSE", "2")

        tcapi.zeros(ctx, TEN, (2, 2))

        assert capsys.readouterr().out

    def test_verbose_does_not_affect_return_value(self, ctx, monkeypatch):
        monkeypatch.setenv("TCAPI_VERBOSE", "2")

        a = tcapi.fill(ctx, TEN, (2, 2), 3.0)

        assert np.allclose(a, 3.0)

    def test_destroyed_context_cannot_be_reused(self):
        dead_ctx = tcapi.create_context(TEN)
        tcapi.destroy_context(dead_ctx)

        with pytest.raises(RuntimeError):
            tcapi.zeros(dead_ctx, TEN, (2, 2))