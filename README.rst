emcee
=====

**The Python ensemble sampling toolkit for affine-invariant MCMC**

.. image:: https://img.shields.io/badge/GitHub-dfm%2Femcee-blue.svg?style=flat
    :target: https://github.com/dfm/emcee
.. image:: http://img.shields.io/travis/dfm/emcee/master.svg?style=flat
    :target: http://travis-ci.org/dfm/emcee
.. image:: https://ci.appveyor.com/api/projects/status/p8smxvleh8mrcn6m?svg=true&style=flat
    :target: https://ci.appveyor.com/project/dfm/emcee
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/dfm/emcee/blob/master/LICENSE
.. image:: http://img.shields.io/badge/arXiv-1202.3665-orange.svg?style=flat
    :target: http://arxiv.org/abs/1202.3665
.. image:: https://coveralls.io/repos/github/dfm/emcee/badge.svg?branch=master&style=flat&v=2
    :target: https://coveralls.io/github/dfm/emcee?branch=master
.. image:: https://readthedocs.org/projects/emcee/badge/?version=latest
    :target: http://emcee.readthedocs.io/en/latest/?badge=latest


emcee is a stable, well tested Python implementation of the affine-invariant
ensemble sampler for Markov chain Monte Carlo (MCMC)
proposed by
`Goodman & Weare (2010) <http://cims.nyu.edu/~weare/papers/d13.pdf>`_.
The code is open source and has
already been used in several published projects in the Astrophysics
literature.

Documentation
-------------

Read the docs at `emcee.readthedocs.io <http://emcee.readthedocs.io/>`_.

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

Copyright 2010-2017 Dan Foreman-Mackey and contributors.

emcee is free software made available under the MIT License. For details see
the LICENSE file.
