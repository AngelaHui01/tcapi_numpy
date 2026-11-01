# tests/test_ten_kind.py
import numpy as np
import pytest
import tcapi


Ten = tcapi.TenKindF64
F32Ten = tcapi.TenKindF32
C64Ten = tcapi.TenKindC64
C128Ten = tcapi.TenKindC128


@pytest.fixture
def ctx():
    c = tcapi.create_context(Ten)
    yield c
    tcapi.destroy_context(c)


class TestTenKind:
    def test_dtype_f32(self):
        assert tcapi.ten_kind_dtype(F32Ten) == np.dtype(np.float32)

    def test_dtype_f64(self):
        assert tcapi.ten_kind_dtype(Ten) == np.dtype(np.float64)

    def test_real_dtype_of_real(self):
        assert tcapi.ten_kind_real_dtype(Ten) == np.dtype(np.float64)

    def test_real_dtype_of_complex(self):
        assert tcapi.ten_kind_real_dtype(C64Ten) == np.dtype(np.float32)

    def test_cplx_dtype_of_real(self):
        assert tcapi.ten_kind_cplx_dtype(F32Ten) == np.dtype(np.complex64)

    def test_cplx_dtype_of_complex(self):
        assert tcapi.ten_kind_cplx_dtype(C128Ten) == np.dtype(np.complex128)

    def test_real_ten_kind_of_complex(self):
        assert tcapi.real_ten_kind(C64Ten) == F32Ten

    def test_real_ten_kind_of_real(self):
        assert tcapi.real_ten_kind(Ten) == Ten

    def test_cplx_ten_kind_of_real(self):
        assert tcapi.cplx_ten_kind(Ten) == C128Ten

    def test_cplx_ten_kind_of_complex(self):
        assert tcapi.cplx_ten_kind(C128Ten) == C128Ten

    def test_ten_type_matches_returned_tensor(self, ctx):
        a = tcapi.zeros(ctx, F32Ten, (2, 3))

        assert isinstance(a, tcapi.ten_kind_ten_type(F32Ten))

    def test_ten_kind_recovers_kind_from_tensor(self, ctx):
        a = tcapi.zeros(ctx, F32Ten, (2, 3))

        assert tcapi.ten_kind(ctx, a) == F32Ten

    # ── supported_ten_kinds ─────────────────────────────────────────── #
    def test_supported_ten_kinds_is_tuple(self):
        kinds = tcapi.supported_ten_kinds
        assert isinstance(kinds, tuple)

    def test_supported_ten_kinds_non_empty(self):
        assert len(tcapi.supported_ten_kinds) > 0

    def test_supported_ten_kinds_no_duplicates(self):
        kinds = tcapi.supported_ten_kinds
        assert len(kinds) == len(set(kinds))

    def test_supported_ten_kinds_roundtrip_dtype(self):
        for ten in tcapi.supported_ten_kinds:
            assert tcapi.ten_kind_dtype(ten) == ten.dtype

    def test_supported_ten_kinds_roundtrip_real_dtype(self):
        for ten in tcapi.supported_ten_kinds:
            assert tcapi.ten_kind_real_dtype(ten) == ten.real_dtype

    def test_supported_ten_kinds_roundtrip_cplx_dtype(self):
        for ten in tcapi.supported_ten_kinds:
            assert tcapi.ten_kind_cplx_dtype(ten) == ten.cplx_dtype

    def test_supported_ten_kinds_roundtrip_real_ten_kind(self):
        for ten in tcapi.supported_ten_kinds:
            assert tcapi.real_ten_kind(ten) == ten.real_ten_kind

    def test_supported_ten_kinds_roundtrip_cplx_ten_kind(self):
        for ten in tcapi.supported_ten_kinds:
            assert tcapi.cplx_ten_kind(ten) == ten.cplx_ten_kind

    def test_supported_ten_kinds_ten_kind_from_tensor(self, ctx):
        for ten in tcapi.supported_ten_kinds:
            t = tcapi.zeros(ctx, ten, (2, 2))
            assert tcapi.ten_kind(ctx, t) == ten