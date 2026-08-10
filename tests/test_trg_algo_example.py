from __future__ import annotations

import numpy as np
import pytest

import tcapi

from example import trg


@pytest.fixture
def ctx():
    context = tcapi.create_context(tcapi.TenKindF64)
    yield context
    tcapi.destroy_context(context)


def test_invalid_arguments(ctx):
    with pytest.raises(ValueError):
        trg.initial_ising_tensor(ctx, 0.0)

    with pytest.raises(ValueError):
        trg.trg_ising(temperature=3.0, maxdim=0, n_steps=1)

    with pytest.raises(ValueError):
        trg.trg_ising(temperature=3.0, maxdim=4, n_steps=-1)

    with pytest.raises(ValueError):
        trg.exact_logz_per_site(0.0)


def test_initial_tensor_values(ctx):
    temperature = 3.0
    tensor = trg.initial_ising_tensor(ctx, temperature)

    assert tcapi.shape(ctx, tensor) == (2, 2, 2, 2)

    def spin(bit: int) -> int:
        return 1 - 2 * bit

    expected = np.empty((2, 2, 2, 2), dtype=np.float64)
    for l in range(2):
        for r in range(2):
            for u in range(2):
                for d in range(2):
                    energy = (
                        spin(l) * spin(d)
                        + spin(d) * spin(r)
                        + spin(r) * spin(u)
                        + spin(u) * spin(l)
                    )
                    expected[l, r, u, d] = np.exp(-energy / temperature)

    assert np.allclose(tensor, expected)


def test_agrees_with_exact_result_at_t3(ctx):
    temperature = 3.0
    maxdim = 16
    n_steps = 5

    exact = trg.exact_logz_per_site(temperature)
    computed = trg.trg_ising(
        temperature=temperature,
        maxdim=maxdim,
        n_steps=n_steps,
        ctx=ctx,
    )

    print(
        "\nTRG benchmark (compact, ITensor per-physical-spin convention)"
        f"\n  Temperature T:             {temperature:.8f}"
        f"\n  maxdim:                    {maxdim}"
        f"\n  n_steps:                   {n_steps}"
        f"\n  TRG log(Z)/N_s:            {computed:.12f}"
        f"\n  Exact log(Z)/N_s:          {exact:.12f}"
        f"\n  Absolute difference:       {abs(computed - exact):.12e}"
    )

    assert np.isfinite(computed)
    assert np.isclose(computed, exact, rtol=5.0e-3, atol=5.0e-3)


def test_convergence_with_scale(ctx):
    exact = trg.exact_logz_per_site(3.0)
    coarse = trg.trg_ising(temperature=3.0, maxdim=4, n_steps=2, ctx=ctx)
    finer = trg.trg_ising(temperature=3.0, maxdim=8, n_steps=5, ctx=ctx)

    coarse_error = abs(coarse - exact)
    finer_error = abs(finer - exact)

    assert finer_error <= coarse_error