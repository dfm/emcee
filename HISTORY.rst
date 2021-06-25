.. :changelog:

3.1.0 (2021-06-25)
++++++++++++++++++

- Added preliminary support for named parameters `#386 <https://github.com/dfm/emcee/pull/386>`_
- Improved handling of blob dtypes `#363 <https://github.com/dfm/emcee/pull/363>`_
- Fixed various small bugs and documentation issues


3.0.2 (2019-11-15)
++++++++++++++++++

- Added tutorial for moves interface
- Added information about contributions to documentation
- Improved documentation for installation and testing
- Fixed dtype issues and instability in linear dependence test
- Final release for `JOSS <https://joss.theoj.org/>`_ submission


3.0.1 (2019-10-28)
++++++++++++++++++

- Added support for long double dtypes
- Prepared manuscript to submit to `JOSS <https://joss.theoj.org/>`_
- Improved packaging and release infrastructure
- Fixed bug in initial linear dependence test


3.0.0 (2019-09-30)
++++++++++++++++++

- Added progress bars using `tqdm <https://github.com/tqdm/tqdm>`_.
- Added HDF5 backend using `h5py <http://www.h5py.org/>`_.
- Added new ``Move`` interface for more flexible specification of proposals.
- Improved autocorrelation time estimation algorithm.
- Switched documentation to using Jupyter notebooks for tutorials.
- More details can be found `on the docs <https://emcee.readthedocs.io/en/v3.0.0/user/upgrade/>`_.

2.2.0 (2016-07-12)
++++++++++++++++++

- Improved autocorrelation time computation.
- Numpy compatibility issues.
- Fixed deprecated integer division behavior in PTSampler.


2.1.0 (2014-05-22)
++++++++++++++++++

- Removing dependence on ``acor`` extension.
- Added arguments to ``PTSampler`` function.
- Added automatic load-balancing for MPI runs.
- Added custom load-balancing for MPI and multiprocessing.
- New default multiprocessing pool that supports ``^C``.


2.0.0 (2013-11-17)
++++++++++++++++++

- **Re-licensed under the MIT license!**
- Clearer less verbose documentation.
- Added checks for parameters becoming infinite or NaN.
- Added checks for log-probability becoming NaN.
- Improved parallelization and various other tweaks in ``PTSampler``.


1.2.0 (2013-01-30)
++++++++++++++++++

- Added a parallel tempering sampler ``PTSampler``.
- Added instructions and utilities for using ``emcee`` with ``MPI``.
- Added ``flatlnprobability`` property to the ``EnsembleSampler`` object
  to be consistent with the ``flatchain`` property.
- Updated document for publication in PASP.
- Various bug fixes.


1.1.3 (2012-11-22)
++++++++++++++++++

- Made the packaging system more robust even when numpy is not installed.


1.1.2 (2012-08-06)
++++++++++++++++++

- Another bug fix related to metadata blobs: the shape of the final ``blobs``
  object was incorrect and all of the entries would generally be identical
  because we needed to copy the list that was appended at each step. Thanks
  goes to Jacqueline Chen (MIT) for catching this problem.


1.1.1 (2012-07-30)
++++++++++++++++++

- Fixed bug related to metadata blobs. The sample function was yielding
  the ``blobs`` object even when it wasn't expected.


1.1.0 (2012-07-28)
++++++++++++++++++

- Allow the ``lnprobfn`` to return arbitrary "blobs" of data as well as the
  log-probability.
- Python 3 compatible (thanks Alex Conley)!
- Various speed ups and clean ups in the core code base.
- New documentation with better examples and more discussion.


1.0.1 (2012-03-31)
++++++++++++++++++

- Fixed transpose bug in the usage of ``acor`` in ``EnsembleSampler``.


1.0.0 (2012-02-15)
++++++++++++++++++

- Initial release.
