#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

vre = re.compile("__version__ = \"(.*?)\"")
m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "emcee", "__init__.py"))
version = vre.findall(m)[0]


setup(
    name="emcee",
    version=version,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["emcee"],
    url="http://dan.iel.fm/emcee/",
    license="MIT",
    description="Kick ass affine-invariant ensemble MCMC sampling",
    long_description=rd("README.rst") + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + rd("HISTORY.rst"),
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=True,
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
