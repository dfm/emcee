"""EMCEE example which gets probabilties from a set of external
processes, rather than from a Python function.

Jeremy Sanders 2012
"""

import subprocess
import select
import atexit
import collections
import os
import sys

import numpy as np
import emcee

# make sure pools are finished at end
_pools = []
def _finishPools():
    while len(_pools) > 0:
        _pools[0].finish()
atexit.register(_finishPools)

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
            self.initSubprocess(p)
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
            self.closeSubprocess(p)
        # wait until they have closed
        for p in self.popens:
            p.wait()
        del self.popens[:]
        # make sure we don't finish twice
        del _pools[ _pools.index(self) ]

    def initSubprocess(self, popen):
        """Initialise the subprocess given by popen.
        Override this."""

    def closeSubprocess(self, popen):
        """Finish process given by popen."""
        popen.stdin.close()

    def sendParameters(self, popen, params):
        """Send parameters to remote subprocess.
        By default just writes a line with parameters + \n

        Override this for more complex behaviour
        """
        txt = ' '.join([str(x) for x in params])
        popen.stdin.write(txt + '\n')
        popen.stdin.flush()

    def getLnProb(self, popen):
        """Called when the subprocess has written something to stdout.
        If the process has returned a lnprob, return its value.
        If it has not, return None.
        Override this."""

        # Read text available. This is more complex than we expect as
        # we might not get the full line. This probably isn't required
        # as we do line buffering.
        txt = os.read(popen.stdout.fileno(), 4096)
        # add to buffered text
        self.buffer[popen] += txt

        fulltxt = self.buffer[popen]
        eol = fulltxt.find('\n')
        if eol < 0:
            # no line end, so we haven't read the complete text
            return None
        else:
            # convert number
            num = float( fulltxt[:eol] )
            # strip out used text
            self.buffer[popen] = fulltxt[eol+1:]
            return num

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
                self.sendParameters(popen, params)
                # move to next parameters and mark popen as busy
                del inparams[0]
                waitingpopens[popen] = idx
                freepopens.remove(popen)

            # poll waiting external commands, waiting at least 1ms
            # if nothing is returned
            for fd, event in self.poll.poll(1):
                popen = self.fdmap[fd]

                # popen got something, so see whether there is a probability
                lnprob = self.getLnProb(popen)
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
    ndim, nwalkers = 2, 100
    p0 = [np.random.rand(ndim)*0.1 for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, None, pool=pool)
    sampler.run_mcmc(p0, 1000)

    print sampler.chain

def remote():
    # return chi2 probability of fit to data
    x = np.arange(9)
    y = np.array([0.97,1.95,3.1,4.04,4.95,6.03,7,7.85,9.1])
    err = 0.2

    while True:
        line = sys.stdin.readline()
        if not line:
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
