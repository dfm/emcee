from __future__ import print_function


__all__ = ["sample_ball", "MH_proposal_axisaligned"]


import numpy as np

# If mpi4py is installed, import it.
try:
    from mpi4py import MPI
    MPI = MPI
except ImportError:
    MPI = None


def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.

    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.

    """
    assert(len(p0) == len(std))
    return np.vstack([p0 + std * np.random.normal(size=len(p0))
                        for i in range(size)])


class MH_proposal_axisaligned(object):
    """
    A Metropolis-Hastings proposal, with axis-aligned Gaussian steps,
    for convenient use as the ``mh_proposal`` option to
    :func:`EnsembleSampler.sample` .

    """
    def __init__(self, stdev):
        self.stdev = stdev

    def __call__(self, X):
        (nw, npar) = X.shape
        assert(len(self.stdev) == npar)
        return X + self.stdev * np.random.normal(size=X.shape)


if MPI is not None:
    class _close_pool_message(object):
        def __repr__(self):
            return u"<Close pool message>"

    class _function_wrapper(object):
        def __init__(self, function):
            self.function = function

    def _error_function(task):
        raise RuntimeError(u"Pool was sent tasks before being told what "
                           u"function to apply.")

    class MPIPool(object):
        """
        A pool that distributes tasks over a set of MPI processes. MPI is an
        API for distributed memory parallelism.  This pool will let you run
        emcee without shared memory, letting you use much larger machines
        with emcee.

        The pool only support the :func:`map` method at the moment because
        this is the only functionality that emcee needs. That being said,
        this pool is fairly general and it could be used for other purposes.

        Contributed by `Joe Zuntz <https://github.com/joezuntz>`_.

        :param comm: (optional)
            The ``mpi4py`` communicator.

        :param debug: (optional)
            If ``True``, print out a lot of status updates at each step.

        """
        def __init__(self, comm=MPI.COMM_WORLD, debug=False):
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size() - 1
            self.debug = debug
            self.function = _error_function
            if self.size == 0:
                raise ValueError(u"Tried to create an MPI pool, but there "
                                 u"was only one MPI process available. "
                                 u"Need at least two.")

        def is_master(self):
            """
            Is the current process the master?

            """
            return self.rank == 0

        def wait(self):
            """
            If this isn't the master process, wait for instructions.

            """
            if self.is_master():
                raise RuntimeError(u"Master node told to await jobs.")

            status = MPI.Status()

            while True:
                # Event loop.
                # Sit here and await instructions.
                if self.debug:
                    print(u"Worker {0} waiting for task.".format(self.rank))

                # Blocking receive to wait for instructions.
                task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if self.debug:
                    print(u"Worker {0} got task {1} with tag {2}."
                                     .format(self.rank, task, status.tag))

                # Check if message is special sentinel signaling end.
                # If so, stop.
                if isinstance(task, _close_pool_message):
                    if self.debug:
                        print(u"Worker {0} told to quit.".format(self.rank))
                    break

                # Check if message is special type containing new function
                # to be applied
                if isinstance(task, _function_wrapper):
                    self.function = task.function
                    if self.debug:
                        print(u"Worker {0} replaced its task function: {1}."
                                .format(self.rank, self.function))
                    continue

                # If not a special message, just run the known function on
                # the input and return it asynchronously.
                result = self.function(task)
                if self.debug:
                    print(u"Worker {0} sending answer {1} with tag {2}."
                            .format(self.rank, result, status.tag))
                self.comm.isend(result, dest=0, tag=status.tag)

        def map(self, function, tasks):
            """
            Like the built-in :func:`map` function, apply a function to all
            of the values in a list and return the list of results.

            :param function:
                The function to apply to the list.

            :param tasks:
                The list of elements.

            """
            ntask = len(tasks)

            # If not the master just wait for instructions.
            if not self.is_master():
                self.wait()
                return

            F = _function_wrapper(function)

            # Tell all the workers what function to use.
            requests = []
            for i in range(self.size):
                r = self.comm.isend(F, dest=i + 1)
                requests.append(r)

            # Wait until all of the workers have responded. See:
            #       https://gist.github.com/4176241
            MPI.Request.waitall(requests)

            # Send all the tasks off and wait for them to be received.
            # Again, see the bug in the above gist.
            requests = []
            for i, task in enumerate(tasks):
                worker = i % self.size + 1
                if self.debug:
                    print(u"Sent task {0} to worker {1} with tag {2}."
                            .format(task, worker, i))
                r = self.comm.isend(task, dest=worker, tag=i)
                requests.append(r)
            MPI.Request.waitall(requests)

            # Now wait for the answers.
            results = []
            for i in range(ntask):
                worker = i % self.size + 1
                if self.debug:
                    print(u"Master waiting for worker {0} with tag {1}"
                            .format(worker, i))
                result = self.comm.recv(source=worker, tag=i)
                results.append(result)
            return results

        def close(self):
            """
            Just send a message off to all the pool members which contains
            the special :class:`_close_pool_message` sentinel.

            """
            if self.is_master():
                for i in range(self.size):
                    self.comm.isend(_close_pool_message(), dest=i + 1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
