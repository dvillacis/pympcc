# Installation

**Requirements:** Python ≥ 3.11. The default IPOPT backend also requires a working IPOPT installation and `cyipopt`.

## System IPOPT

```bash
# macOS (Homebrew)
brew install ipopt

# Linux (Debian/Ubuntu)
sudo apt-get install coinor-libipopt-dev
```

## Package install

```bash
pip install "pympcc[ipopt]"          # default IPOPT backend
pip install "pympcc[scipy]"          # SciPy backend only (no IPOPT)
pip install "pympcc[dev]"            # adds pytest / coverage / lint
pip install "pympcc[numba]"          # Numba JIT kernels for hot paths
pip install "pympcc[jax]"            # JAX-backed derivatives & solve_jax
```

## Optional: custom IPOPT linear-solver bridge

Only needed when passing `linear_solver_fn`. Build in-place after installing IPOPT and the dev dependencies:

```bash
uv run python setup.py build_ext --inplace

# Custom IPOPT prefix:
IPOPT_INCLUDE_DIR=/path/to/include/coin-or IPOPT_LIB_DIR=/path/to/lib \
  uv run python setup.py build_ext --inplace
```

## From source

```bash
git clone https://github.com/dvillacis/pympcc.git
cd pympcc
uv pip install -e ".[dev,jax,docs]"
uv run pytest tests/ -v
```
