"""An emcee example which gets probabilties from a set of external
processes, rather than from a Python function. We use a Pool-like
object which provides map to pass to emcee.

This example starts the remote() method of itself in different
processes to compute the lnprob. The remote process returns the
probability for a chi2 fit of a+b*x to some data.

Note that by using a command line using the "ssh" command, this
example can be extended to run on many computers simultaneously.

Note that this example will not work on Windows, as Windows does not
allow select.select to be used on pipes from subprocesses.

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
def _finish_pools():
    while _pools:
        _pools[0].finish()
atexit.register(_finish_pools)

class Pool(object):
    """Pool object manages external commands and sends and receives
    values."""

    def __init__(self, commands):
        """Start up remote procesess."""

        # list of open subprocesses
        self.popens = []
        # input text buffers for processes
        self.buffer = collections.defaultdict(str)

        for cmd in commands:
            p = subprocess.Popen(cmd,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            self.init_subprocess(p)
            self.popens.append(p)

        # keep track of open pool objects
        _pools.append(self)

    def finish(self):
        """Finish all processes."""
        # tell processes to finish
        for p in self.popens:
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

    def send_parameters(self, stdin, params):
        """Send parameters to remote subprocess.
        By default just writes a line with parameters + \n

        Override this for more complex behaviour
        """
        txt = ' '.join([str(x) for x in params])
        stdin.write(txt + '\n')
        stdin.flush()

    def identify_lnprob(self, text):
        """Is the log probability in this text from the remote
        process? Return value if yes, or None.

        Override this if process returns more than a single value
        """
        if text[-1] != '\n':
            return None
        try:
            return float(text.strip())
        except ValueError:
            return None

    def get_lnprob(self, stdout):
        """Called when the subprocess has written something to stdout.
        If the process has returned a lnprob, return its value.
        If it has not, return None.
        """

        # Read text available. This is more complex than we expect as
        # we might not get the full text.
        txt = os.read(stdout.fileno(), 4096)
        # add to buffered text
        self.buffer[stdout] += txt

        val = self.identify_lnprob(self.buffer[stdout])
        if val is not None:
            self.buffer[stdout] = ''
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
        # Stdout from systems currently doing work.  Maps stdout ->
        # (output index, Popen object)
        waitingstdout = {}

        # repeat while work to do, or work being done
        while inparams or waitingstdout:

            # start job if possible
            while freepopens and inparams:
                idx, params = inparams[0]
                popen = iter(freepopens).next()
                # send the process the parameters
                self.send_parameters(popen.stdin, params)
                # move to next parameters and mark popen as busy
                del inparams[0]
                waitingstdout[popen.stdout] = (idx, popen)
                freepopens.remove(popen)

            # see whether any stdouts have output
            stdouts = select.select( waitingstdout.keys(), [], [], 0.001 )[0]
            for stdout in stdouts:
                # see whether process has written out probability
                lnprob = self.get_lnprob(stdout)
                if lnprob is not None:
                    # record result
                    idx, popen = waitingstdout[stdout]
                    results[idx] = lnprob
                    # open process up for work again
                    del waitingstdout[stdout]
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
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

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
