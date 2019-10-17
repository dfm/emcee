---
title: 'emcee v3: A Python ensemble sampling toolkit for affine-invariant MCMC'
tags:
  - Python
  - astronomy
authors:
  - name: Daniel Foreman-Mackey
    orcid: 0000-0003-0872-7098
    affiliation: 1
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute
   index: 1
date: 17 October 2019
bibliography: paper.bib
---

# Summary

``emcee`` is a Python library implementing a class of affine-invariant ensemble samplers for Markov chain Monte Carlo (MCMC).
This package has been widely applied to probabilistic modeling problems in astrophysics where it was originally published [@Foreman-Mackey:2013], with some applications in other fields.
When it was first released in 2012, the interface implemented in ``emcee`` was fundamentally different from the MCMC libraries that were popular at the time, such as ``PyMC``, because it was specifically designed to work with "black box" models instead of structured graphical models.
This has been a popular interface for applications in astrophysics where it can sometimes be non-trivial to implement physics within the modeling frameworks required by other libraries, and since then some other libraries have been developed with similar interfaces, such as ``dynesty`` [@Speagle:2019].
The version 3.0 release of ``emcee`` is the first major release of the library in about 6 years and it includes a full re-write of the computational backend, several commonly requested features, and a set of new "move" implementations.

The new features include small quality of life improvements—like a progress bar using [``tqdm``](https://tqdm.github.io)—and some more involved features.
One major feature is the new ``backends`` interface which implements real time serialization of the sampling results.
By default ``emcee`` saves its results in memory (as in the original implementation), but it now also includes a ``HDFBackend`` class that serializes the chain to disk in an HDF5 file, via [h5py](https://www.h5py.org).

The most important new feature included in the version 3.0 release of ``emcee`` is the new ``moves`` interface.
Originally, ``emcee`` was implementation of the affine-invariant "stretch move" proposed by @Goodman:2010, but there are other ensemble proposals that can sometimes get better performance.
``emcee`` now includes implementations of several other ensemble moves and an interface for defining custom proposals.
The implemented moves include:

- The "stretch move" proposed by @Goodman:2010,
- The "differential evolution" and "differential evolution snooker update" moves [@Ter-Braak:2006; @Ter-Braak:2008], and
- A "kernel density proposal" based on the implementation in [the ``kombine`` library](https://github.com/bfarr/kombine) [@Farr:2015].

``emcee`` has been widely used and the original paper has been highly cited, but there have been many contributions from members of the community.
This paper is meant to highlight these contributions and provide citation credit to the academic contributors.

# References
