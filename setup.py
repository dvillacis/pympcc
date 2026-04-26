"""Build the optional Cython bridge for custom IPOPT linear solvers.

Usage:
    uv run python setup.py build_ext --inplace

The extension is not required for normal pympcc usage. It is only loaded when
``linear_solver_fn`` is supplied. IPOPT discovery checks ``IPOPT_PREFIX``,
``CONDA_PREFIX``, the active Python prefix, and common system prefixes. Set
``IPOPT_INCLUDE_DIR`` and ``IPOPT_LIB_DIR`` to override discovery explicitly.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

ROOT = Path(__file__).parent.resolve()
CYTHON_DIR = ROOT / "pympcc" / "cython"


def _sdk_path():
    """Return the active Xcode/CLT SDK path."""
    try:
        return subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()
    except Exception:
        return ""


def _cxx_stdlib_include() -> str:
    """Return the C++ stdlib include path (needed for Cython-generated includes)."""
    sdk = _sdk_path()
    if sdk:
        path = Path(sdk) / "usr" / "include" / "c++" / "v1"
        if path.is_dir():
            return str(path)
    return ""


def _candidate_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    for env_name in ("IPOPT_PREFIX", "CONDA_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            prefixes.append(Path(value))
    prefixes.extend([Path(sys.prefix), Path("/opt/homebrew"), Path("/usr/local"), Path("/usr")])
    return prefixes


def _has_ipopt_headers(path: Path) -> bool:
    return (path / "IpIpoptApplication.hpp").is_file()


def _has_ipopt_library(path: Path) -> bool:
    return bool(
        list(path.glob("libipopt*"))
        or (path / "ipopt.lib").is_file()
    )


def _first_existing(paths: list[Path], predicate) -> str | None:
    for path in paths:
        if path.is_dir() and predicate(path):
            return str(path)
    return None


def _ipopt_paths() -> tuple[str, str]:
    include_override = os.environ.get("IPOPT_INCLUDE_DIR")
    lib_override = os.environ.get("IPOPT_LIB_DIR")
    include_paths: list[Path] = []
    lib_paths: list[Path] = []

    if include_override:
        include_paths.append(Path(include_override))
    if lib_override:
        lib_paths.append(Path(lib_override))

    for prefix in _candidate_prefixes():
        include_paths.extend([
            prefix / "include" / "coin-or",
            prefix / "include" / "coin",
        ])
        lib_paths.append(prefix / "lib")

    include_dir = _first_existing(include_paths, _has_ipopt_headers)
    lib_dir = _first_existing(lib_paths, _has_ipopt_library)
    if include_dir and lib_dir:
        return include_dir, lib_dir

    raise RuntimeError(
        "Could not find IPOPT headers/libraries. Set IPOPT_PREFIX, or set "
        "IPOPT_INCLUDE_DIR and IPOPT_LIB_DIR before running "
        "`python setup.py build_ext --inplace`."
    )


IPOPT_INCLUDE, IPOPT_LIB = _ipopt_paths()
NUMPY_INCLUDE = numpy.get_include()
SDK_PATH = _sdk_path()
CXX_STDLIB_INCLUDE = _cxx_stdlib_include()
DARWIN_SDK_ARGS = [f"-isysroot{SDK_PATH}"] if sys.platform == "darwin" and SDK_PATH else []

ext = Extension(
    name="pympcc.cython._custom_solver",
    sources=[str(CYTHON_DIR / "_custom_solver.pyx")],
    include_dirs=[
        IPOPT_INCLUDE,
        NUMPY_INCLUDE,
        str(CYTHON_DIR),  # for _custom_solver_helper.hpp and _ipopt_cpp.pxd
    ] + ([CXX_STDLIB_INCLUDE] if CXX_STDLIB_INCLUDE else []),
    library_dirs=[IPOPT_LIB],
    libraries=["ipopt"],
    language="c++",
    extra_compile_args=[
        "-std=c++14",
        "-O2",
        "-fvisibility=hidden",  # macOS: avoid symbol conflicts with IPOPT
        "-Wno-unused-variable",
        "-Wno-deprecated-declarations",
    ] + DARWIN_SDK_ARGS,
    extra_link_args=[
        f"-Wl,-rpath,{IPOPT_LIB}",  # embed rpath so the dylib is found at runtime
    ] + DARWIN_SDK_ARGS,
)

setup(
    name="pympcc-cython-ext",
    ext_modules=cythonize(
        [ext],
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound":  False,
            "cdivision":   True,
        },
    ),
)
