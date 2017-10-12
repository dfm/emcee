# -*- coding: utf-8 -*-

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
from emcee import __version__  # NOQA


# Override CSS
# def setup(app):
#     app.add_stylesheet("custom.css")


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'emcee'
copyright = '2012-2017, Dan Foreman-Mackey & contributors'
version = __version__
release = __version__

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# pygments_style = 'sphinx'
# html_theme = 'default'
html_static_path = ['_static']
# html_logo = "_static/logo-sidebar.png"
# html_theme_options = {"logo_only": True}
