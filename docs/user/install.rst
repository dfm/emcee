.. _install:

Installation
============

Since emcee is a pure Python module, it should be pretty easy to install.
All you'll need `numpy <http://numpy.scipy.org/>`_.

.. note:: For pre-release versions of emcee, you need to follow the
    instructions in :ref:`source`.


Package managers
----------------

The easiest way to install the stable version of emcee is using
`pip <http://www.pip-installer.org/>`_ or `conda <https://conda.io>`_

.. code-block:: bash

    pip install emcee
    # or...
    conda install -c conda-forge emcee


.. _source:

From source
-----------

emcee is developed on `GitHub <https://github.com/dfm/emcee>`_ so if you feel
like hacking or if you like all the most recent shininess, you can clone the
source repository and install from there

.. code-block:: bash

    git clone https://github.com/dfm/emcee.git
    cd emcee
    python setup.py install


Test the installation
---------------------

To make sure that the installation went alright, you can execute some unit and
integration tests.
To do this, you'll need the source (see :ref:`source` above) and
`py.test <https://docs.pytest.org>`_.
You'll execute the tests by running the following command in the root
directory of the source code:

.. code-block:: bash

    py.test -v tests

This might take a few minutes but you shouldn't get any errors if all went
as planned.
