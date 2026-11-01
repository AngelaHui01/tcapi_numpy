# tests/test_linalg.py
import numpy as np
import pytest
import tcapi


TEN = tcapi.TenKindF64


@pytest.fixture
def ctx():
    context = tcapi.create_context(TEN)
    yield context
    tcapi.destroy_context(context)


class TestLinearAlgebra:
    def test_norm_known_vector(self, ctx):
        a = tcapi.assign_from_range(
            ctx,
            TEN,
            (2,),
            [3.0, 4.0],
            lambda coors: coors[0],
        )

        assert np.isclose(tcapi.norm(ctx, a), 5.0)

    def test_diag_roundtrip(self, ctx):
        vector = tcapi.assign_from_range(
            ctx,
            TEN,
            (3,),
            [1.0, 2.0, 3.0],
            lambda coors: coors[0],
        )

        diagonal = tcapi.diag(ctx, vector)
        recovered_vector = tcapi.diag(ctx, diagonal)

        assert np.allclose(recovered_vector, vector)

    def test_contract_matmul(self, ctx):
        identity = tcapi.eye(ctx, TEN, 3)
        matrix = tcapi.fill(ctx, TEN, (3, 3), 2.0)

        result = tcapi.contract(ctx, identity, "ij", matrix, "jk", "ik")

        assert np.allclose(result, matrix)

    def test_contract_scalar(self, ctx):
        vector = tcapi.assign_from_range(
            ctx,
            TEN,
            (3,),
            [1.0, 2.0, 3.0],
            lambda coors: coors[0],
        )

        dot_product = tcapi.contract(ctx, vector, "i", vector, "i", "")

        assert np.isclose(float(dot_product), 14.0)

    def test_contract_integer_labels_match_strings(self, ctx):
        rng = np.random.default_rng(0)
        a = tcapi.random(ctx, TEN, (3, 4, 2), lambda: rng.uniform())
        rng = np.random.default_rng(1)
        b = tcapi.random(ctx, TEN, (2, 4, 5), lambda: rng.uniform())

        via_ints = tcapi.contract(ctx, a, (1, -1, -2), b, (-2, -1, 0), (0, 1))
        via_strings = tcapi.contract(ctx, a, "ijk", b, "kjl", "li")

        assert via_ints.shape == (5, 3)
        assert np.allclose(via_ints, via_strings)

    def test_contract_dimension_mismatch_raises(self, ctx):
        a = tcapi.fill(ctx, TEN, (3, 4, 2), 1.0)
        b = tcapi.fill(ctx, TEN, (2, 3, 3), 1.0)

        with pytest.raises(ValueError, match="disagree"):
            tcapi.contract(ctx, a, "ijk", b, "kjl", "il")

    def test_svd_reconstruction(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 3),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            lambda coors: 3 * coors[0] + coors[1],
        )

        u, sigma, v_dag = tcapi.svd(ctx, matrix, 1)
        us = tcapi.contract(ctx, u, "ik", sigma, "kl", "il")
        reconstructed = tcapi.contract(ctx, us, "ik", v_dag, "kj", "ij")

        assert np.allclose(reconstructed, matrix, atol=1e-12)

    def test_trunc_svd_max_bonds(self, ctx):
        rng = np.random.default_rng(42)
        matrix = tcapi.random(
            ctx,
            TEN,
            (6, 6),
            lambda: rng.uniform(),
        )

        _, sigma, _, trunc_err = tcapi.trunc_svd(
            ctx,
            matrix,
            1,
            3,
            0.0,
        )

        assert tcapi.shape(ctx, sigma) == (3, 3)
        assert 0.0 <= trunc_err <= 1.0

    def test_trunc_svd_target_error_strategy(self, ctx):
        rng = np.random.default_rng(7)
        matrix = tcapi.random(
            ctx,
            TEN,
            (6, 6),
            lambda: rng.uniform(),
        )

        _, sigma, _, trunc_err = tcapi.trunc_svd(
            ctx,
            matrix,
            1,
            chi_max=5,
            chi_min=2,
            target_trunc_err=1.0e-3,
        )

        assert tcapi.shape(ctx, sigma)[0] >= 2
        assert 0.0 <= trunc_err <= 1.0e-3

    def test_trunc_svd_spec_example(self, ctx):
        """Reproduce the spec-v2 example:
        trunc_svd(ctx, a, 2, 6, 1e-12, 3, 1e-2) on a 3x4x5x6 tensor.
        """
        rng = np.random.default_rng(42)
        a = tcapi.random(ctx, TEN, (3, 4, 5, 6), lambda: rng.uniform())

        u, sigma, v_t, trunc_err = tcapi.trunc_svd(ctx, a, 2, 6, 1e-12, 3, 1e-2)

        chi = tcapi.shape(ctx, sigma)[0]
        assert 3 <= chi <= 6, f"chi {chi} not in [3, 6]"

        # Verify trunc_err is consistent with discarded singular values
        # by performing a full SVD.
        u_full, sigma_full, _ = tcapi.svd(ctx, a, 2)
        s_full = np.diag(sigma_full)
        total_sq = float(np.sum(s_full ** 2))
        expected_err = (
            float(np.sum(s_full[chi:] ** 2)) / total_sq
            if total_sq != 0.0 else 0.0
        )
        assert np.isclose(trunc_err, expected_err, atol=1e-12)

    def test_eigh_known_matrix(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 2),
            [2.0, 1.0, 1.0, 2.0],
            lambda coors: 2 * coors[0] + coors[1],
        )

        eigenvalue_matrix, _ = tcapi.eigh(ctx, matrix, 1)
        eigenvalues = tcapi.diag(ctx, eigenvalue_matrix)

        assert np.allclose(eigenvalues, [1.0, 3.0])

    def test_scale_returns_scaled_tensor(self, ctx):
        original = tcapi.assign_from_range(
            ctx,
            TEN,
            (3,),
            [1.0, 2.0, 3.0],
            lambda coors: coors[0],
        )

        scaled = tcapi.scale(ctx, original, 2.0)

        assert np.allclose(scaled, [2.0, 4.0, 6.0])
        assert np.allclose(original, [1.0, 2.0, 3.0])

    def test_inverse_of_identity(self, ctx):
        identity = tcapi.eye(ctx, TEN, 3)

        inverse = tcapi.inverse(ctx, identity, 1)

        assert np.allclose(inverse, identity)

    def test_trace_spec_example(self, ctx):
        rng = np.random.default_rng(0)
        a = tcapi.random(ctx, TEN, (3, 4, 2, 4, 2), lambda: rng.uniform())

        t = tcapi.trace(ctx, a, ((1, 3), (2, 4)))

        assert t.shape == (3,)
        manual = np.einsum(np.asarray(a), [0, 1, 2, 1, 2], [0])
        assert np.allclose(t, manual)

    def test_trace_no_pairs_returns_copy(self, ctx):
        a = tcapi.fill(ctx, TEN, (2, 3), 1.0)

        b = tcapi.trace(ctx, a, ())

        assert np.array_equal(b, a)
        assert not np.shares_memory(a, b)

    def test_eig_reconstruction(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 2),
            [1.0, 2.0, 2.0, 1.0],
            lambda coors: 2 * coors[0] + coors[1],
        )

        lambda_mat, v = tcapi.eig(ctx, matrix, 1)

        assert lambda_mat.shape == (2, 2)
        assert v.shape == (2, 2)
        assert np.allclose(np.asarray(matrix) @ v, v @ lambda_mat, atol=1e-12)

    def test_eigvals_known_matrix(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 2),
            [1.0, 2.0, 2.0, 1.0],
            lambda coors: 2 * coors[0] + coors[1],
        )

        eigenvalues = tcapi.eigvals(ctx, matrix, 1)

        assert np.allclose(np.sort(eigenvalues), [-1.0, 3.0])

    def test_eigvalsh_known_symmetric(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (3, 3),
            [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
            lambda coors: 3 * coors[0] + coors[1],
        )

        eigenvalues = tcapi.eigvalsh(ctx, matrix, 1)

        assert np.allclose(np.sort(eigenvalues), [2.0, 3.0, 4.0])

    def test_exp_of_diagonal(self, ctx):
        matrix = tcapi.assign_from_range(
            ctx,
            TEN,
            (2, 2),
            [2.0, 0.0, 0.0, 3.0],
            lambda coors: 2 * coors[0] + coors[1],
        )

        result = tcapi.exp(ctx, matrix, 1)

        assert np.allclose(result, np.diag(np.exp([2.0, 3.0])))

    def test_qr_reconstruction(self, ctx):
        rng = np.random.default_rng(3)
        matrix = tcapi.random(ctx, TEN, (2, 3), lambda: rng.uniform())

        q, r = tcapi.qr(ctx, matrix, 1)

        assert q.shape == (2, 2)
        assert r.shape == (2, 3)
        assert np.allclose(q @ r, matrix)
        assert np.allclose(q.T @ q, np.eye(2), atol=1e-12)

    def test_lq_reconstruction(self, ctx):
        rng = np.random.default_rng(4)
        matrix = tcapi.random(ctx, TEN, (3, 2), lambda: rng.uniform())

        l, q = tcapi.lq(ctx, matrix, 1)

        assert l.shape == (3, 2)
        assert q.shape == (2, 2)
        assert np.allclose(l @ q, matrix)
        assert np.allclose(q.T @ q, np.eye(2), atol=1e-12)