# AGENTS.md

## Project purpose

This repository implements a dense NumPy backend for the Python TCAPI tensor interface.
The package source is `tcapi/`; it provides a reference implementation for tensor
semantics and tensor-network examples.

Prioritize correctness, explicit validation, readable NumPy code, and reproducible tests
over premature performance optimization.

## Repository map

- `tcapi/`: Public package implementation.
  - `__init__.py`: Public API exports. Add a newly public function here only after
    its implementation and tests are complete.
  - `constructors.py`: Tensor creation and copying.
  - `queries.py`: Read-only tensor queries.
  - `io.py`: Save/load operations.
  - `manipulation.py`: Shape, slicing, element, and structural tensor operations.
  - `linalg.py`: Linear-algebra operations and decompositions.
  - `misc.py`: Context lifecycle, conversion, display, and version utilities.
  - `ten_kind.py`: `TenKind` descriptors and dtype-related behaviour.
  - `diagnostics.py`: Diagnostic and runtime-reporting utilities.
- `tests/`: pytest tests, normally one test module per source module.
- `example/`: Executable tensor-network reference examples; do not turn these into
  pytest tests unless they are fast, deterministic, and assertion-based.
- `pyproject.toml`: Project metadata and dependencies.
- `uv.lock`: Locked dependency versions; keep it committed.

## Environment and commands

Use `uv`; do not install project dependencies with bare `pip`.

```bash
# Run the full test suite
uv run pytest -v

# Run one test module
uv run pytest -v tests/test_linalg.py

# Show print output while running a test
uv run pytest -s -v tests/test_trg_algo.py

# Run an example
uv run python example/itebd_tfim.py
```

Run the smallest relevant test module after each change. Run the full suite before
claiming that a change is complete.

## Implementation rules

- Use `numpy` as the dense tensor backend. Do not introduce PyTorch, JAX, ITensor,
  or other numerical backends unless explicitly requested.
- Preserve the existing public function signatures and argument order unless the
  task explicitly requires an API change.
- Keep tensor validation explicit: check expected dimensions, shapes, axes, dtypes,
  and invalid arguments early; raise informative Python exceptions.
- Use NumPy-native operations and avoid element-by-element Python loops when a clear
  vectorized expression is available.
- Maintain real/complex dtype behaviour through `TenKind` and existing conventions.
- Do not silently change numerical conventions, tensor-axis ordering, normalisation
  definitions, or returned values.
- Add type hints and concise docstrings to new public functions.
- Keep unrelated refactors out of task-focused commits.

## Tests and numerical validation

- Add or update pytest coverage whenever behaviour changes.
- Test normal cases, edge cases, invalid inputs, shapes, dtypes, and complex-valued
  behaviour when relevant.
- Use `numpy.testing` or `numpy.allclose` for floating-point arrays; do not use exact
  equality for results from decompositions or iterative numerical algorithms.
- For SVD, QR, eigenvalue, contraction, and truncation routines, test mathematical
  invariants or reconstruction identities rather than a particular non-unique factorisation.
- Keep tests deterministic: use explicit RNG seeds whenever randomness is required.
- Tests must not leave generated artifacts in the repository root. Use pytest
  temporary directories (`tmp_path`) for test-generated files such as `.npy` files.

## Examples and research claims

- Treat examples as reference implementations, not proof of correctness.
- Clearly distinguish a smoke test from a physical/numerical validation.
- Do not claim agreement with the TCAPI paper or another backend without an explicit,
  reproducible comparison and stated tolerance.
- Record parameter values, bond dimensions, tolerances, normalization conventions,
  runtime environment, and measured metrics in benchmark or validation outputs.
- For TRG/TEBD-style algorithms, state that finite bond dimension and SVD truncation
  make the result approximate.

## Version-control rules

- Keep source, tests, documentation, `pyproject.toml`, and `uv.lock` tracked.
- Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, generated `.npy` files,
  `.DS_Store`, or personal VS Code workspace settings.
- Do not edit, delete, or stage unrelated files.
- Before committing, inspect changes with `git status` and `git diff`.