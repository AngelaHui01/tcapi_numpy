# tests/test_io.py
import os

import numpy as np
import pytest
import tcapi


@pytest.fixture
def ctx():
    c = tcapi.create_context(tcapi.TenKindF64)
    yield c
    tcapi.destroy_context(c)


@pytest.fixture
def tmp_path_npy(tmp_path):
    """Provide a temporary filename stem."""
    return str(tmp_path / "test_tensor")


class TestIO:
    # --- version ---
    def test_version_returns_string(self):
        version = tcapi.version(tcapi.TenKindF64)

        assert isinstance(version, str)

    def test_version_value(self):
        assert tcapi.version(tcapi.TenKindF64) == "1.0"

    # --- save & load ---
    def test_save_creates_file(self, ctx, tmp_path_npy):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (2, 3), 5.0)

        tcapi.save(ctx, a, tmp_path_npy)

        # The TCAPI specification leaves the on-disk representation
        # backend-defined, so do not require a NumPy ".npy" suffix.
        assert os.path.exists(tmp_path_npy) or os.path.exists(tmp_path_npy + ".npy")

    def test_load_roundtrip_values(self, ctx, tmp_path_npy):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (2, 3), 5.0)
        tcapi.save(ctx, a, tmp_path_npy)

        b = tcapi.load(ctx, tcapi.TenKindF64, tmp_path_npy)

        assert np.allclose(a, b)

    def test_load_roundtrip_shape(self, ctx, tmp_path_npy):
        a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        tcapi.save(ctx, a, tmp_path_npy)

        b = tcapi.load(ctx, tcapi.TenKindF64, tmp_path_npy)

        assert tuple(tcapi.shape(ctx, b)) == (2, 3, 4)

    def test_load_dtype_mismatch_raises(self, ctx, tmp_path_npy):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)
        tcapi.save(ctx, a, tmp_path_npy)

        with pytest.raises(TypeError):
            tcapi.load(ctx, tcapi.TenKindF32, tmp_path_npy)

    # --- show ---
    def test_show_runs_without_error(self, ctx, capsys):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (2, 2), 1.0)

        tcapi.show(ctx, a)

        captured = capsys.readouterr()
        assert captured.out

    # --- close ---
    def test_close_identical_tensors(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)
        b = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)

        assert tcapi.close(ctx, a, b, 0.0) is True

    def test_close_within_tolerance(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)
        b = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.005)

        assert tcapi.close(ctx, a, b, 0.01) is True

    def test_close_outside_tolerance(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)
        b = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.005)

        assert tcapi.close(ctx, a, b, 1e-6) is False

    def test_close_shape_mismatch_returns_false(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 1.0)
        b = tcapi.fill(ctx, tcapi.TenKindF64, (4,), 1.0)

        assert tcapi.close(ctx, a, b, 0.01) is False

    # --- convert ---
    def test_convert_real_to_complex(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 2.0)

        b = tcapi.convert(ctx, a, ctx, tcapi.TenKindC128)

        assert b.dtype == np.dtype(np.complex128)

    def test_convert_f64_to_f32(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 2.0)

        b = tcapi.convert(ctx, a, ctx, tcapi.TenKindF32)

        assert b.dtype == np.dtype(np.float32)

    def test_convert_preserves_values(self, ctx):
        a = tcapi.fill(ctx, tcapi.TenKindF64, (3,), 2.0)

        b = tcapi.convert(ctx, a, ctx, tcapi.TenKindF32)

        assert np.allclose(a, b.astype(np.float64))