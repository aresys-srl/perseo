# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Configuration file for the Sphinx documentation builder."""

import re
from pathlib import Path

project = "perseo-perturbations"
base_name = "perseo_perturbations"
copyright = "2025, Aresys S.R.L."
author = "Aresys S.R.L."

project_version_match = re.search(
    r"__version__\s*=\s*(.*)", Path(__file__).parent.parent.parent.joinpath("src", "perseo_perturbations", "__init__.py").read_text(encoding="UTF-8")
)
VERSION = project_version_match.group(1).replace('"', "")

# stripped version
stripped_version = VERSION[: VERSION[: VERSION.rfind(".")].rfind(".")]
# The full version, including alpha/beta/rc tags
release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "8.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",  # for numpy docstring
    "sphinx.ext.mathjax",
    "nbsphinx",
]

python_use_unqualified_type_names = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {'.rst': 'restructuredtext'}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Aresys",
            "url": "https://www.aresys.it/",
            "icon": "_static/icons/aresys_logo.svg",
            "type": "local",
        }
    ],
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_title = project

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

html_copy_source = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = base_name

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, base_name, "Documentation", [author], 1)]

# -- Extension configuration -------------------------------------------------

autodoc_default_options = {"members": True, "undoc-members": True}
autodoc_member_order = "bysource"  # alphabetical, groupwise
autoclass_content = "both"  # class, init, both
autodoc_preserve_defaults = True
autodoc_type_aliases = {"npt.ArrayLike": "ArrayLike", "ArrayLike": "ArrayLike"}
autodoc_mock_imports = ["perseo_perturbations.geodynamics.solid"]

napoleon_use_param = True
napoleon_preprocess_types = True
napoleon_type_aliases = {}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
