.. _faq:

FAQ
===

**The not-so-frequently asked questions that still have useful answers**

.. _walkers:

What are "walkers"?
-------------------

Walkers are the members of the ensemble. They are almost like separate
Metropolis-Hastings chains but, of course, the proposal distribution for
a given walker depends on the positions of all the other walkers in the
ensemble. See `Goodman & Weare (2010)
<http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ for more details.


How should I initialize the walkers?
------------------------------------

The best technique seems to be to start in a small ball around the a priori
preferred position. Don't worry, the walkers quickly branch out and explore
the rest of the space.


Troubleshooting
---------------

**I'm getting weird spikes in my data/I have low acceptance fractions/both...
what should I do?**

Double the number of walkers. If that doesn't work, double it again. And
again. Until you run out of RAM. At that point, I don't know!


**The walkers are getting stuck in "islands" of low likelihood. How can I
fix that?**

Try increasing the number of walkers. If that doesn't work, you can try
pruning using a clustering algorithm like the one found in
`arxiv:1104.2612 <http://arxiv.org/abs/1104.2612>`_.


Attribution
-----------

If you find this useful, please `cite us <http://arxiv.org/abs/1202.3665>`_
and add your paper to the :ref:`testimonials`.
Also, please `fork us on GitHub <https://github.com/dfm/emcee>`_ so we can
all benefit from any changes you come up with!
