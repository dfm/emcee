import os
import sys
import emcee

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    name="emcee",
    version=emcee.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["emcee"],
    url="http://danfm.ca/emcee/",
    license=open("LICENSE").read(),
    description="Kick ass affine-invariant ensemble MCMC sampling",
    long_description=open("README.rst").read() + "\n\n" +
                     open("HISTORY.rst").read(),
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
