# -*- coding: utf-8 -*-

"""
Python's multiprocessing.Pool class doesn't interact well with
``KeyboardInterrupt`` signals, as documented in places such as:

* `<http://stackoverflow.com/questions/1408356/>`_
* `<http://stackoverflow.com/questions/11312525/>`_
* `<http://noswap.com/blog/python-multiprocessing-keyboardinterrupt>`_

Various workarounds have been shared. Here, we adapt the one proposed in the
last link above, by John Reese, and shared as

* `<https://github.com/jreese/multiprocessing-keyboardinterrupt/>`_

Our version is a drop-in replacement for multiprocessing.Pool ... as long as
the map() method is the only one that needs to be interrupt-friendly.

Contributed by Peter K. G. Williams <peter@newton.cx>.

*Added in version 2.1.0*

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["InterruptiblePool"]

import signal
import functools
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError


def _initializer_wrapper(actual_initializer, *rest):
    """
    We ignore SIGINT. It's up to our parent to kill us in the typical
    condition of this arising from ``^C`` on a terminal. If someone is
    manually killing us with that signal, well... nothing will happen.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if actual_initializer is not None:
        actual_initializer(*rest)


class InterruptiblePool(Pool):
    """
    A modified version of :class:`multiprocessing.pool.Pool` that has better
    behavior with regard to ``KeyboardInterrupts`` in the :func:`map` method.

    :param processes: (optional)
        The number of worker processes to use; defaults to the number of CPUs.

    :param initializer: (optional)
        Either ``None``, or a callable that will be invoked by each worker
        process when it starts.

    :param initargs: (optional)
        Arguments for *initializer*; it will be called as
        ``initializer(*initargs)``.

    :param kwargs: (optional)
        Extra arguments. Python 2.7 supports a ``maxtasksperchild`` parameter.

    """
    wait_timeout = 3600

    def __init__(self, processes=None, initializer=None, initargs=(),
                 **kwargs):
        new_initializer = functools.partial(_initializer_wrapper, initializer)
        super(InterruptiblePool, self).__init__(processes, new_initializer,
                                                initargs, **kwargs)

    def map(self, func, iterable, chunksize=None):
        """
        Equivalent of ``map()`` built-in, without swallowing
        ``KeyboardInterrupt``.

        :param func:
            The function to apply to the items.

        :param iterable:
            An iterable of items that will have `func` applied to them.

        """
        # The key magic is that we must call r.get() with a timeout, because
        # a Condition.wait() without a timeout swallows KeyboardInterrupts.
        r = self.map_async(func, iterable, chunksize)

        while True:
            try:
                return r.get(self.wait_timeout)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise
            # Other exceptions propagate up.
