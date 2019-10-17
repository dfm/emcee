# -*- coding: utf-8 -*-

import glob
import os
import subprocess

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("emcee").version
except DistributionNotFound:
    __version__ = "unknown version"


# Convert the tutorials
for fn in glob.glob("_static/notebooks/*.ipynb"):
    name = os.path.splitext(os.path.split(fn)[1])[0]
    outfn = os.path.join("tutorials", name + ".rst")
    print("Building {0}...".format(name))
    subprocess.check_call(
        "jupyter nbconvert --template tutorials/tutorial_rst --to rst "
        + fn
        + " --output-dir tutorials",
        shell=True,
    )
    subprocess.check_call("python fix_internal_links.py " + outfn, shell=True)


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "emcee"
copyright = "2012-2019, Dan Foreman-Mackey & contributors"
version = __version__
release = __version__

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme

    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_logo = "_static/logo2.png"
html_theme_options = {"logo_only": True}
