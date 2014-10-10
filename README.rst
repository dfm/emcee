emcee
=====

**The Python ensemble sampling toolkit for affine-invariant MCMC**

.. image:: https://secure.travis-ci.org/dfm/emcee.svg?branch=master
        :target: http://travis-ci.org/dfm/emcee
.. image:: https://pypip.in/d/emcee/badge.svg
        :target: https://pypi.python.org/pypi/emcee/
.. image:: https://pypip.in/v/emcee/badge.svg
        :target: https://pypi.python.org/pypi/emcee/

emcee is a stable, well tested Python implementation of the affine-invariant
ensemble sampler for Markov chain Monte Carlo (MCMC)
proposed by
`Goodman & Weare (2010) <http://cims.nyu.edu/~weare/papers/d13.pdf>`_.
The code is open source and has
already been used in several published projects in the Astrophysics
literature.

Documentation
-------------

Read the docs at `dan.iel.fm/emcee <http://dan.iel.fm/emcee/>`_.

Attribution
-----------

Please cite `Foreman-Mackey, Hogg, Lang & Goodman (2012)
<http://arxiv.org/abs/1202.3665>`_ if you find this code useful in your
research and add your paper to `the testimonials list
<https://github.com/dfm/emcee/blob/master/docs/testimonials.rst>`_.
The BibTeX entry for the paper is::

    @article{emcee,
       author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Lang}, D. and {Goodman}, J.},
        title = {emcee: The MCMC Hammer},
      journal = {PASP},
         year = 2013,
       volume = 125,
        pages = {306-312},
       eprint = {1202.3665},
          doi = {10.1086/670067}
    }

License
-------

Copyright 2010-2013 Dan Foreman-Mackey and contributors.

emcee is free software made available under the MIT License. For details see
the LICENSE file.
