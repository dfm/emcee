.. _install:

Installation
============

Since ``emcee`` is a pure Python module, it should be pretty easy to install.
All you'll need `numpy <http://numpy.scipy.org/>`_.


Using pip
---------

The easiest way to install the most recent stable version of ``emcee`` is
with `pip <http://www.pip-installer.org/>`_. Run this from the command line:

.. code-block:: bash

    pip install emcee

You might need to run this using ``sudo`` depending on your Python
installation. You can also use ``easy_install``:

.. code-block:: bash

    easy_install emcee

but ``pip`` is probably better.


From source
-----------

Alternatively, you can get the source by downloading a
`tarball <https://github.com/dfm/emcee/tarball/master>`_:

.. code-block:: bash

    curl -OL https://github.com/dfm/emcee/tarball/master

or `zip archive <https://github.com/dfm/emcee/zipball/master>`_:

.. code-block:: bash

    curl -OL https://github.com/dfm/emcee/zipball/master

Once you've downloaded and unpacked the source, you can navigate into the
root source directory and run:

.. code-block:: bash

    python setup.py install


Bleeding edge development version
---------------------------------

``emcee`` is being developed actively on `GitHub
<https://github.com/dfm/emcee>`_ so if you feel like hacking, you can clone
the source repository

.. code-block:: bash

    git clone https://github.com/dfm/emcee.git

or `fork the repository <https://github.com/dfm/emcee>`_.


Test the installation
---------------------

To test the installation, install `py.test <http://pytest.org/>`_ and run:

.. code-block:: bash

    py.test -v --pyargs emcee

This might take a few minutes but you shouldn't get any errors if all went
as planned.
