"""Sphinx configuration for the pympcc documentation."""
from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "pympcc"
author = "David Villacis"
copyright = f"{datetime.now().year}, {author}"

from pympcc import __version__ as _version  # noqa: E402

version = _version
release = _version

extensions = [
    "myst_nb",                     # markdown + executable notebooks (also provides myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# -- myst-nb (executable notebooks) -------------------------------------------
nb_execution_mode = "cache"        # only re-execute when source changes
nb_execution_timeout = 90          # seconds per cell (IPOPT solves can be slow)
nb_execution_excludepatterns = []
nb_merge_streams = True
nb_execution_raise_on_error = True

myst_enable_extensions = [
    "amsmath",
    "deflist",
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3

# Let myst-nb register the .md parser automatically (it provides
# myst_parser internally).  Explicitly listing ``"markdown"`` here
# breaks parser lookup because myst-nb registers under a different name.
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"pympcc {version}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/dvillacis/pympcc",
    "source_branch": "master",
    "source_directory": "docs/",
}

mathjax3_config = {
    "tex": {
        "macros": {
            "RR": r"\mathbb{R}",
            "argmin": r"\operatorname*{arg\,min}",
        }
    }
}
