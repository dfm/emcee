# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

if "test" in sys.argv:
    version = "0.0.0"

else:
    # Hackishly inject a constant into builtins to enable importing of the
    # package even if numpy isn't installed. Only do this if we're not
    # running the tests!
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins
    builtins.__EMCEE_SETUP__ = True
    import emcee
    version = emcee.__version__

# Publish to PyPI.
if "publish" in sys.argv:
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_wheel upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()


# Testing.
class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name="emcee",
    version=version,
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
    package_data={"": ["LICENSE", "*.rst"]},
    include_package_data=True,
    install_requires=[
        "numpy >= 1.7"
    ],
    tests_require=[
        "pytest",
        "pytest-cov",
    ],
    cmdclass = {"test": PyTest},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
    ],
)
