from __future__ import annotations

import numpy as np
import pytest

import tcapi

from example import itebd_tfim


@pytest.fixture
def ctx():
    context = tcapi.create_context(tcapi.TenKindF64)
    yield context
    tcapi.destroy_context(context)


def test_itebd_imaginary_time_converges_to_tfim_ground_state(ctx):
    J, g = 1.0, 1.0
    dt = 0.005
    chi = 8
    N = 8000
    d = 2

    h = itebd_tfim.make_tfim_two_site_hamiltonian(ctx, J, g)
    gate = itebd_tfim.make_imaginary_time_gate(ctx, h, dt, d)
    energies = itebd_tfim.itebd_imaginary_time(ctx, gate, N, dt, chi)

    exact = itebd_tfim.exact_tfim_energy_per_site(J, g)
    final_estimate = energies[-1]
    abs_diff = abs(final_estimate - exact)
    last_100 = energies[-100:]
    stabilization_span = max(last_100) - min(last_100)

    print("Appendix A: TFIM imaginary-time iTEBD")
    print(f"parameters: J={J:g}, g={g:g}, dt={dt:g}, chi={chi}, N={N}")
    print(f"final E_iTEBD       = {final_estimate:.15f}")
    print(f"exact energy/site   = {exact:.15f}")
    print(f"absolute difference = {abs_diff:.3e}")
    print(f"final 100-iteration span = {stabilization_span:.3e}")

    # Critical TFIM energy per site: J=1, g=1 → exact = ~ -4/π ≈ -1.2732395447.
    # See Appendix A of arXiv:2512.23917v2.
    # At the critical point, finite χ truncation introduces small residual
    # fluctuations, so we use a relaxed stability threshold.
    assert np.isfinite(final_estimate)
    assert abs_diff < 1.0e-4
    assert stabilization_span < 1.0e-6