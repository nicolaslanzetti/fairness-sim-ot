# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'fairness-sim-ot'
copyright = '2024, Shubham Chowdhary'
author = 'Shubham Chowdhary'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # 'alabaster'
html_static_path = ['_static']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For NumPy or Google style docstrings
    'sphinx.ext.viewcode',  # Adds links to highlighted source code
    'sphinx.ext.autosummary',  # Creates summary tables for modules/classes
    'sphinx_autodoc_typehints',  # Type hints support
]

sys.path.insert(
    0, os.path.abspath('.'))  # Adjust path based on your folder structure

# global config options
LOG_LEVEL = "INFO"  # only DEBUG and INFO supported, anything else leads ...
# ... to CRITICAL
LOG_FILE = "./run_docs.log"

PROJECT_ROOT = os.getcwd()
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
print("Project Root:", PROJECT_ROOT)

os.environ["LOG_LEVEL"] = LOG_LEVEL
print("Log level for libraries:", LOG_LEVEL)

os.environ["LOG_FILE"] = PROJECT_ROOT + "/" + LOG_FILE
print("Doc logs are present at", os.environ["LOG_FILE"])
