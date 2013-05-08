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


vre = re.compile("__version__ = \"(.*?)\"")
m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "emcee", "__init__.py")).read()
version = vre.findall(m)[0]


setup(
    name="emcee",
    version=version,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["emcee"],
    url="http://danfm.ca/emcee/",
    license="GPLv2",
    description="Kick ass affine-invariant ensemble MCMC sampling",
    long_description=open("README.rst").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.rst").read(),
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=True,
    install_requires=["numpy >= 1.6.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
