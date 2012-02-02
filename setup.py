from distutils.core import setup

setup(
    name="emcee",
    version="1.0.0",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["emcee"],
    url="http://danfm.ca/emcee/",
    license="GPL",
    description="Kick ass affine-invariant ensemble MCMC sampling",
    long_description="Read the docs at http://danfm.ca/emcee/",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

