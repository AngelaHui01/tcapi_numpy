# TCAPI - NumPy Backend

[TCAPI (Tensor Computing API)](https://arxiv.org/abs/2512.23917) is a common tensor-computing interface for tensor-network applications. This repository provides a pure Python/NumPy backend that implements the TCAPI function set for dense tensors.

The installed Python package is named `tcapi`; the project distribution is `tcapi-numpy`.

## Requirements

- **Python 3.12+**
- **NumPy 2.5+**
- **SciPy 1.18+**
- **pytest 9.1+** for tests

The project is configured with `pyproject.toml` and uses Hatchling as the build backend.

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd tcapi_numpy
```

### 2. Install Dependencies

Using `uv`:

```bash
uv sync --dev
```

Or with `pip` in an existing Python 3.12+ environment:

```bash
python -m pip install -e .
python -m pip install "pytest>=9.1.1"
```

### 3. Run Tests

```bash
uv run pytest
```

or:

```bash
python -m pytest
```

## Usage Example

```python
import tcapi

Ten = tcapi.TenKindF64
ctx = tcapi.create_context(Ten)

try:
    a = tcapi.fill(ctx, Ten, (2, 3), 2.0)
    b = tcapi.eye(ctx, Ten, 3)

    print(tcapi.shape(ctx, a))       # (2, 3)
    print(tcapi.size(ctx, a))        # 6
    print(tcapi.norm(ctx, b))        # sqrt(3)

    doubled = tcapi.for_each(ctx, a, lambda value: 2.0 * value)
    tcapi.show(ctx, doubled)
finally:
    tcapi.destroy_context(ctx)
```

## API Coverage

This backend implements dense-tensor versions of the main TCAPI categories:

| Category | Functions |
|---|---|
| Context and metadata | `create_context`, `destroy_context`, `version`, `ten_kind` |
| Queries | `order`, `shape`, `size`, `size_bytes`, `get_elem` |
| Construction | `allocate`, `zeros`, `assign_from_range`, `fill`, `random`, `eye`, `copy` |
| I/O | `save`, `load` using NumPy `.npy` serialization |
| Manipulation | `set_elem`, `transpose`, `reshape`, `cplx_conj`, `to_cplx`, `real`, `imag`, `expand`, `shrink`, `extract_sub`, `replace_sub`, `concatenate`, `stack`, `for_each`, `for_each_with_coors` |
| Linear algebra | `norm`, `diag`, `normalize`, `scale`, `trace`, `contract`, `linear_combine`, `exp`, `inverse`, `svd`, `trunc_svd`, `qr`, `lq`, `eigvals`, `eigvalsh`, `eig`, `eigh` |
| Utilities | `show`, `close`, `convert`, `to_range` |
| Diagnostics | `TCAPI_VERBOSE` environment variable for runtime call diagnostics — level `1` prints TCAPI function names and compact argument summaries, level `2` also prints measured wall-clock time per wrapped TCAPI call |

Design notes:

- Tensor storage is always dense NumPy storage.
- Most mutating-style TCAPI routines return a new tensor rather than modifying the input in place.
- Symmetry-aware storage, GPU execution, and automatic differentiation are not implemented in this backend.

## Linear Algebra Example

Tensor contractions are backed by `numpy.einsum` and accept either string labels or non-string label sequences:

```python
import tcapi

Ten = tcapi.TenKindF64
ctx = tcapi.create_context(Ten)

try:
    identity = tcapi.eye(ctx, Ten, 3)
    matrix = tcapi.fill(ctx, Ten, (3, 3), 2.0)

    product = tcapi.contract(ctx, identity, "ij", matrix, "jk", "ik")
    assert tcapi.close(ctx, product, matrix, 0.0)

    u, sigma, v_dag = tcapi.svd(ctx, matrix, 1)
finally:
    tcapi.destroy_context(ctx)
```

For tensor decompositions, `num_of_bds_as_row` controls how many leading tensor axes are grouped into the matrix row index before applying the NumPy/SciPy linear algebra routine.

## Examples

The `example/` directory contains tensor-network demonstrations built on the TCAPI NumPy backend. Each module is kept definition-only (no `__main__` entry point) and is exercised through pytest.

- `example/trg.py`: Levin-Nave TRG for the infinite square-lattice Ising model, in a compact form that mirrors the [ITensor TRG tutorial](https://itensor.org/docs.cgi?page=book/trg). Exposes `trg_ising`, `initial_ising_tensor`, and the Onsager exact reference `exact_logz_per_site`.
- `example/itebd_tfim.py`: imaginary-time iTEBD for the 1D transverse-field Ising model, mirroring Appendix A of the TCAPI paper. Exposes `itebd_imaginary_time`, the gate/exact-energy helpers, and the exact reference `exact_tfim_energy_per_site`.

Run their tests with (use `-s` to see the printed benchmark output):

```bash
uv run pytest -v -s tests/test_trg_algo_example.py
uv run pytest -v -s tests/test_itebd_example.py
```

Both examples are deliberately heavier than the basic unit tests: the TRG test runs multiple truncation steps with bond dimension up to 16, and the iTEBD test iterates thousands of imaginary-time steps and validates the converged energy against the exact TFIM ground state.

## Development

Useful commands:

```bash
# Install project and development dependencies
uv sync --dev

# Run the complete test suite (all test codes)
uv run pytest

# Run a focused test file
uv run pytest tests/test_linalg.py

# Build a wheel/sdist
uv build
```

## License
This project is licensed under the [Apache License 2.0](LICENSE).