.. _install:

Installation
============

Since emcee is a pure Python module, it should be pretty easy to install.
All you'll need `numpy <https://numpy.org/>`_.

.. note:: For pre-release versions of emcee, you need to follow the
    instructions in :ref:`source`.


Package managers
----------------

The recommended way to install the stable version of emcee is using
`pip <http://www.pip-installer.org/>`_

.. code-block:: bash

    python -m pip install -U pip
    pip install -U setuptools setuptools_scm pep517
    pip install -U emcee

or `conda <https://conda.io>`_

.. code-block:: bash

    conda update conda
    conda install -c conda-forge emcee

Distribution packages
---------------------

Some distributions contain `emcee` packages that can be installed with the
system package manager as listed in the `Repology packaging status
<https://repology.org/project/python:emcee/versions>`_. Note that the packages
in some of these distributions may be out-of-date. You can always get the
current stable version via `pip` or `conda`, or the latest development version
as described in :ref:`source` below.

.. image:: https://repology.org/badge/vertical-allrepos/python:emcee.svg?header=emcee%20packaging%20status
    :target: https://repology.org/project/python:emcee/versions

.. _source:

From source
-----------

emcee is developed on `GitHub <https://github.com/dfm/emcee>`_ so if you feel
like hacking or if you like all the most recent shininess, you can clone the
source repository and install from there

.. code-block:: bash

    python -m pip install -U pip
    python -m pip install -U setuptools setuptools_scm pep517
    git clone https://github.com/dfm/emcee.git
    cd emcee
    python -m pip install -e .


Test the installation
---------------------

To make sure that the installation went alright, you can execute some unit and
integration tests.
To do this, you'll need the source (see :ref:`source` above) and
`py.test <https://docs.pytest.org>`_.
You'll execute the tests by running the following command in the root
directory of the source code:

.. code-block:: bash

    python -m pip install -U pytest h5py
    python -m pytest -v src/emcee/tests

This might take a few minutes but you shouldn't get any errors if all went
as planned.
