.. :changelog:

1.1.2 (2012-08-06)
++++++++++++++++++

- Another bug fix related to metadata blobs: the shape of the final `blobs`
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
