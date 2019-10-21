.. _faq:

FAQ
===

**The not-so-frequently asked questions that still have useful answers**

What are "walkers"?
-------------------

Walkers are the members of the ensemble. They are almost like separate
Metropolis-Hastings chains but, of course, the proposal distribution for
a given walker depends on the positions of all the other walkers in the
ensemble. See `Goodman & Weare (2010)
<https://msp.org/camcos/2010/5-1/p04.xhtml>`_ for more details.


How should I initialize the walkers?
------------------------------------

The best technique seems to be to start in a small ball around the a priori
preferred position. Don't worry, the walkers quickly branch out and explore
the rest of the space.


Parameter limits
----------------

In order to confine the walkers to a finite volume of the parameter space, have
your function return negative infinity outside of the volume corresponding to
the logarithm of 0 prior probability using

.. code-block:: python

    return -numpy.inf
