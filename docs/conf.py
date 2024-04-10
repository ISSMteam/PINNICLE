# Configuration file for the Sphinx documentation builder.
from importlib.metadata import version

# -- Project information

project = 'PINNICLE'
copyright = '2024, Cheng Gong'
author = 'Cheng Gong'

# The short X.Y version
version = version("pinnicle")
version = '0.1'
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# The master toctree
master_doc = "index"

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# The suffix(es) of source filenames.
source_suffix = ".rst"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
