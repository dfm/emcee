#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__EMCEE_SETUP__ = True
import emcee

setup(
    name="emcee",
    version=emcee.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["emcee"],
    url="http://dan.iel.fm/emcee/",
    license="MIT",
    description="Kick ass affine-invariant ensemble MCMC sampling",
    long_description=(open("README.rst").read() + "\n\n"
                      + "Changelog\n"
                      + "---------\n\n"
                      + open("HISTORY.rst").read()),
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=True,
    install_requires=[
        "numpy >= 1.6"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
