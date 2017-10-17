#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Hackishly inject a constant into builtins to enable importing of the
# package before all the dependencies are built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__EMCEE_SETUP__ = True
import emcee  # NOQA

setup(
    name="emcee",
    version=emcee.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    packages=["emcee", "emcee.moves", "emcee.backends"],
    url="http://emcee.readthedocs.io",
    license="MIT",
    description=("The Python ensemble sampling toolkit for affine-invariant "
                 "MCMC"),
    long_description=open("README.rst").read(),
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
    zip_safe=True,
)
