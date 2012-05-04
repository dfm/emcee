"""An emcee example which gets probabilties from a set of external
processes, rather than from a Python function. We use a Pool-like
object which provides map to pass to emcee.

This example starts the remote() method of itself in different
processes to compute the lnprob. The remote process returns the
probability for a chi2 fit of a+b*x to some data.

Note that by using a command line using the "ssh" command, this
example can be extended to run on many computers simultaneously.

Also note that the reliance on select.Poll means this will not work on
Windows.

Jeremy Sanders 2012
"""

import subprocess
import select
import atexit
import collections
import os
import sys
import re

import numpy as np
import emcee

# make sure pools are finished at end
_pools = []
def _finish_pools():
    while len(_pools) > 0:
        _pools[0].finish()
atexit.register(_finish_pools)

class Pool(object):
    """Pool object manages external commands and sends and receives
    values."""

    def __init__(self, commands):
        """Start up remote procesess."""

        # list of open subprocesses
        self.popens = []
        # object to poll stdouts of subprocesses
        self.poll = select.poll()
        # mapping of output filedescriptors to popens
        self.fdmap = {}
        # input text buffers for processes
        self.buffer = collections.defaultdict(str)

        for cmd in commands:
            p = subprocess.Popen(cmd,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            self.init_subprocess(p)
            self.popens.append(p)

            fd = p.stdout.fileno()
            self.fdmap[fd] = p
            self.poll.register(fd, select.POLLIN)

        # keep track of open pool objects
        _pools.append(self)

    def finish(self):
        """Finish all processes."""
        # tell processes to finish
        for p in self.popens:
            self.poll.unregister(p.stdout.fileno())
            self.close_subprocess(p)
        # wait until they have closed
        for p in self.popens:
            p.wait()
        del self.popens[:]
        # make sure we don't finish twice
        del _pools[ _pools.index(self) ]

    def init_subprocess(self, popen):
        """Initialise the subprocess given by popen.
        Override this if required."""

    def close_subprocess(self, popen):
        """Finish process given by popen.
        Override this if required
        """
        popen.stdin.close()

    def send_parameters(self, popen, params):
        """Send parameters to remote subprocess.
        By default just writes a line with parameters + \n

        Override this for more complex behaviour
        """
        txt = ' '.join([str(x) for x in params])
        popen.stdin.write(txt + '\n')
        popen.stdin.flush()

    def identify_lnprob(self, text):
        """Is the log probability in this text from the remote
        process. Return value if yes, or None.

        Override this
        """
        if text[-1] != '\n':
            return None
        try:
            return float(text.strip())
        except ValueError:
            return None

    def get_lnprob(self, popen):
        """Called when the subprocess has written something to stdout.
        If the process has returned a lnprob, return its value.
        If it has not, return None.
        """

        # Read text available. This is more complex than we expect as
        # we might not get the full text.
        txt = os.read(popen.stdout.fileno(), 4096)
        # add to buffered text
        self.buffer[popen] += txt

        val = self.identify_lnprob(self.buffer[popen])
        if val is not None:
            self.buffer[popen] = ''
            return val
        else:
            return None

    def map(self, function, paramlist):
        """Return a list of lnprob values for the list parameter sets
        given.

        Note: function is never called!
        """

        # create a map of index to parameter set
        inparams = zip(range(len(paramlist)), paramlist)

        # what we're going to return
        results = [None]*len(inparams)

        # systems which are waiting to do work
        freepopens = set( self.popens )
        # systems doing work (mapping popen -> retn index)
        waitingpopens = {}

        # repeat while work to do, or work being done
        while len(inparams) > 0 or len(waitingpopens) > 0:

            # start job if possible
            while len(freepopens) > 0 and len(inparams) > 0:
                idx, params = inparams[0]
                popen = iter(freepopens).next()
                # send the process the parameters
                self.send_parameters(popen, params)
                # move to next parameters and mark popen as busy
                del inparams[0]
                waitingpopens[popen] = idx
                freepopens.remove(popen)

            # poll waiting external commands, waiting at least 1ms
            # if nothing is returned
            for fd, event in self.poll.poll(1):
                popen = self.fdmap[fd]

                # popen got something, so see whether there is a probability
                lnprob = self.get_lnprob(popen)
                if lnprob is not None:
                    # record result
                    idx = waitingpopens[popen]
                    results[idx] = lnprob
                    # open process up for work again
                    del waitingpopens[popen]
                    freepopens.add(popen)

        return results

def main():
    # subprocesses to run
    cmds = [ [ sys.executable, __file__, 'remote' ]
            for i in range(4) ]
    # start subprocesses
    pool = Pool( cmds )

    # two parameter chi2 fit to data (see remote below)
    ndim, nwalkers, nburn, nchain = 2, 100, 100, 1000
    # some wild initial parameters
    p0 = [np.random.rand(ndim)*0.1 for i in xrange(nwalkers)]

    # Start sampler. Note lnprob function is None as it is not used.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, None, pool=pool)

    # Burn in period
    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    # Proper run
    sampler.run_mcmc(pos, nchain, rstate0=state)

    # Print out median parameters (a, b)
    print "a = %g, b = %g" % ( np.median(sampler.chain[:,:,0]),
                               np.median(sampler.chain[:,:,1]) )

def remote():
    """Return chi2 probability of fit to data."""

    # our fake data and error bars
    x = np.arange(9)
    y = np.array([1.97,2.95,4.1,5.04,5.95,6.03,8,8.85,10.1])
    err = 0.2

    while True:
        line = sys.stdin.readline()
        if not line:
            # calling process has closed stdin
            break

        params = [float(v) for v in line.split()]
        mody = params[0] + params[1]*x
        chi2 = np.sum( ((y-mody) / err)**2 )
        lnprob = -0.5*chi2
        sys.stdout.write(str(lnprob)+'\n')
        sys.stdout.flush()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'remote':
        remote()
    else:
        main()
