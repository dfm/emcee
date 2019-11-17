---
title: 'emcee v3: A Python ensemble sampling toolkit for affine-invariant MCMC'
tags:
  - Python
  - astronomy
authors:
  - name: Daniel Foreman-Mackey
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Will M. Farr
    orcid: 0000-0003-1540-8562
    affiliation: "1, 2"
  - name: Manodeep Sinha
    orcid: 0000-0002-4845-1228
    affiliation: "3, 4"
  - name: Anne M. Archibald
    orcid: 0000-0003-0638-3340
    affiliation: 5
  - name: David W. Hogg
    orcid: 0000-0003-2866-9403
    affiliation: "1, 6"
  - name: Jeremy S. Sanders
    orcid: 0000-0003-2189-4501
    affiliation: 7
  - name: Joe Zuntz
    orcid: 0000-0001-9789-9646
    affiliation: 8
  - name: Peter K. G. Williams
    orcid: 0000-0003-3734-3587
    affiliation: "9, 10"
  - name: Andrew R. J. Nelson
    orcid: 0000-0002-4548-3558
    affiliation: 11
  - name: Miguel de Val-Borro
    orcid: 0000-0002-0455-9384
    affiliation: 12
  - name: Tobias Erhardt
    orcid: 0000-0002-6683-6746
    affiliation: 13
  - name: Ilya Pashchenko
    orcid: 0000-0002-9404-7023
    affiliation: 14
  - name: Oriol Abril Pla
    orcid: 0000-0002-1847-9481
    affiliation: 15
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute
    index: 1
  - name: Department of Physics and Astronomy, Stony Brook University, United States
    index: 2
  - name: Centre for Astrophysics & Supercomputing, Swinburne University of Technology, Australia
    index: 3
  - name: ARC Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D)
    index: 4
  - name: University of Newcastle
    index: 5
  - name: Center for Cosmology and Particle Physics, Department of Physics, New York University
    index: 6
  - name: Max Planck Institute for Extraterrestrial Physics
    index: 7
  - name: Institute for Astronomy, University of Edinburgh, Edinburgh, EH9 3HJ, UK
    index: 8
  - name: "Center for Astrophysics | Harvard & Smithsonian"
    index: 9
  - name: American Astronomical Society
    index: 10
  - name: Australian Nuclear Science and Technology Organisation, NSW, Australia
    index: 11
  - name: Planetary Science Institute, 1700 East Fort Lowell Rd., Suite 106, Tucson, AZ 85719, USA
    index: 12
  - name: Climate and Environmental Physics and Oeschger Center for Climate Change Research, University of Bern, Bern, Switzerland
    index: 13
  - name: P.N. Lebedev Physical Institute of the Russian Academy of Sciences, Moscow, Russia
    index: 14
  - name: Universitat Pompeu Fabra, Barcelona
    index: 15

date: 17 October 2019
bibliography: paper.bib
---

# Summary

``emcee`` is a Python library implementing a class of affine-invariant ensemble samplers for Markov chain Monte Carlo (MCMC).
This package has been widely applied to probabilistic modeling problems in astrophysics where it was originally published [@Foreman-Mackey:2013], with some applications in other fields.
When it was first released in 2012, the interface implemented in ``emcee`` was fundamentally different from the MCMC libraries that were popular at the time, such as ``PyMC``, because it was specifically designed to work with "black box" models instead of structured graphical models.
This has been a popular interface for applications in astrophysics because it is often non-trivial to implement realistic physics within the modeling frameworks required by other libraries.
Since ``emcee``'s release, other libraries have been developed with similar interfaces, such as ``dynesty`` [@Speagle:2019].
The version 3.0 release of ``emcee`` is the first major release of the library in about 6 years and it includes a full re-write of the computational backend, several commonly requested features, and a set of new "move" implementations.

This new release includes both small quality of life improvements—like a progress bar using [``tqdm``](https://tqdm.github.io)—and larger features.
For example, the new ``backends`` interface implements real time serialization of sampling results.
By default ``emcee`` saves its results in memory (as in the original implementation), but it now also includes a ``HDFBackend`` class that serializes the chain to disk using [h5py](https://www.h5py.org).

The most important new feature included in the version 3.0 release of ``emcee`` is the new ``moves`` interface.
Originally, ``emcee`` implemented the affine-invariant "stretch move" proposed by @Goodman:2010, but there are other ensemble proposals that can get better performance for certain applications.
``emcee`` now includes implementations of several other ensemble moves and an interface for defining custom proposals.
The implemented moves include:

- The "stretch move" proposed by @Goodman:2010,
- The "differential evolution" and "differential evolution snooker update" moves [@Ter-Braak:2006; @Ter-Braak:2008], and
- A "kernel density proposal" based on the implementation in [the ``kombine`` library](https://github.com/bfarr/kombine) [@Farr:2015].

``emcee`` has been widely used and the original paper has been highly cited, but there have been many contributions from members of the community.
This paper is meant to highlight these contributions and provide citation credit to the academic contributors.
A full up-to-date list of contributors can always be found [on GitHub](https://github.com/dfm/emcee/graphs/contributors).

# References
