.. _moves-user:

Moves
=====

emcee was originally built on the "stretch move" ensemble method from `Goodman
& Weare (2010) <https://msp.org/camcos/2010/5-1/p04.xhtml>`_, but
starting with version 3, emcee nows allows proposals generated from a mixture
of "moves".
This can be used to get a more efficient sampler for models where the stretch
move is not well suited, such as high dimensional or multi-modal probability
surfaces.

A "move" is an algorithm for updating the coordinates of walkers in an
ensemble sampler based on the current set of coordinates in a manner that
satisfies detailed balance.
In most cases, the update for each walker is based on the coordinates in some
other set of walkers, the complementary ensemble.

These moves have been designed to update the ensemble in parallel following
the prescription from `Foreman-Mackey et al. (2013)
<https://arxiv.org/abs/1202.3665>`_.
This means that computationally expensive models can take advantage of
multiple CPUs to accelerate sampling (see the :ref:`parallel` tutorial for
more information).

The moves are selected using the ``moves`` keyword for the
:class:`EnsembleSampler` and the mixture can optionally be a weighted mixture
of moves.
During sampling, at each step, a move is randomly selected from the mixture
and used as the proposal.

The default move is still the :class:`moves.StretchMove`, but the others are
described below.
Many standard ensemble moves are available with parallelization provided by
the :class:`moves.RedBlueMove` abstract base class that implements the
parallelization method described by `Foreman-Mackey et al. (2013)
<https://arxiv.org/abs/1202.3665>`_.
In addition to these moves, there is also a framework for building
Metropolis–Hastings proposals that update the walkers using independent
proposals.
:class:`moves.MHMove` is the base class for this type of move and a concrete
implementation of a Gaussian Metropolis proposal is found in
:class:`moves.GaussianMove`.

.. note:: The :ref:`moves` tutorial shows a concrete example of how to use
    this interface.

Ensemble moves
--------------

.. autoclass:: emcee.moves.RedBlueMove
    :members:

.. autoclass:: emcee.moves.StretchMove
    :members:

.. autoclass:: emcee.moves.WalkMove
    :members:

.. autoclass:: emcee.moves.KDEMove
    :members:

.. autoclass:: emcee.moves.DEMove
    :members:

.. autoclass:: emcee.moves.DESnookerMove
    :members:

Metropolis–Hastings moves
-------------------------

.. autoclass:: emcee.moves.MHMove
    :members:

.. autoclass:: emcee.moves.GaussianMove
    :members:
