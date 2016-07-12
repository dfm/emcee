.. _install:

Installation
============

Since ``emcee`` is a pure Python module, it should be pretty easy to install.
All you'll need `numpy <http://numpy.scipy.org/>`_. There are a bunch of
different ways to install and I'll mention a few below but by far the best
is to install into a `virtual environment <http://www.virtualenv.org/>`_
using `pip <http://www.pip-installer.org/>`_.


Using pip
---------

The easiest way to install the most recent stable version of ``emcee`` is
with `pip <http://www.pip-installer.org/>`_:

::

    $ pip install emcee

You might need to run this using ``sudo`` depending on your Python
installation. You can also use ``easy_install``:

::

    $ easy_install emcee

but ``pip`` is probably better.


From source
-----------

Alternatively, you can get the source by downloading a
`tarball <https://github.com/dfm/emcee/tarball/master>`_:

::

    $ curl -OL https://github.com/dfm/emcee/tarball/master

or `zip archive <https://github.com/dfm/emcee/zipball/master>`_:

::

    $ curl -OL https://github.com/dfm/emcee/zipball/master

Once you've downloaded and unpacked the source, you can navigate into the
root source directory and run:

::

    $ python setup.py install


Bleeding edge development version
---------------------------------

``emcee`` is being developed actively on `GitHub
<https://github.com/dfm/emcee>`_ so if you feel like hacking, you can clone
the source repository

::

    git clone https://github.com/dfm/emcee.git

or `fork the repository <https://github.com/dfm/emcee>`_.


Test the installation
---------------------

To make sure that the installation went alright, you can run some unit tests
by running:

::

    python -c 'import emcee; emcee.test()'

or, if you have `nose <http://nose.readthedocs.org/>`_:

::

    nosetests

This might take a few minutes but you shouldn't get any errors if all went
as planned.
