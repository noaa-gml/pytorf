# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pytorf'
copyright = '2025, Sergio Ibarra-Espinosa'
author = 'Sergio Ibarra-Espinosa'
release = '2/20/2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# https://blog.ceshine.net/post/sphinx-python-doc/
import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # or "../../src

extensions = [
    "sphinx.ext.autodoc",  # automatically generate documentation for modules
    "sphinx.ext.napoleon",  # to read Google-style or Numpy-style docstrings
    "sphinx.ext.viewcode",  # to allow vieing the source code in the web page
    "autodocsumm",  # to generate tables of functions, attributes, methods, etc.
]

#html_theme = 'sphinx_rtd_theme'

# don't include docstrings from the parent class
autodoc_inherit_docstrings = False
# Show types only in descriptions, not in signatures
autodoc_typehints = "description"