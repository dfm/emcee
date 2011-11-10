#!/usr/bin/env python
# encoding: utf-8
"""


"""

from __future__ import division

__all__ = ['GaussianSampler', 'EMSampler']

import numpy as np

from ensemble import EnsembleSampler
import mixtures

class GaussianSampler(EnsembleSampler):
    def _propose_position(self,s0,comp,lnprob):
        """
        Propose a new position given a current position and the complementary set

        Parameters
        ----------
        s0 : numpy.ndarray
            The current positions of the set of walkers that will be advanced

        comp : numpy.ndarray
            The complementary set of walkers

        lnprob : numpy.ndarray
            The current ln-probability of the s0 set

        Returns
        -------
        newposition : numpy.ndarray
            The new positions of the walkers (same shape as s0)

        newlnprob : numpy.ndarray
            The ln-probability of the walkers at newposition

        accept : numpy.ndarray
            Array of bools indicating whether or not a walker's new position
            should be accepted

        """
        s = np.atleast_2d(s0)
        n0 = len(s)
        c = np.atleast_2d(comp)
        ncomp = len(comp)

        mu  = np.mean(c, axis=0)
        cov = np.cov(c, rowvar=0)

        # propose new walker position and calculate the lnprobability
        newposition = self._random.multivariate_normal(mu,cov,size=n0)
        newposition[:,self._fixedinds] = self._fixedvals
        newlnprob = self.ensemble_lnposterior(newposition)

        diff = newposition-mu
        newQ = -0.5*np.dot(diff, np.linalg.solve(cov, diff.T)).diagonal()
        diff = s0-mu
        oldQ = -0.5*np.dot(diff, np.linalg.solve(cov, diff.T)).diagonal()

        lnpdiff = oldQ - newQ + newlnprob - lnprob
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return newposition, newlnprob, accept

class EMSampler(EnsembleSampler):
    def __init__(self, *args, **kwargs):
        self._K = kwargs.pop('K', 3)
        super(EMSampler, self).__init__(*args, **kwargs)

    def _propose_position(self,s0,comp,lnprob):
        """
        Propose a new position given a current position and the complementary set

        Parameters
        ----------
        s0 : numpy.ndarray
            The current positions of the set of walkers that will be advanced

        comp : numpy.ndarray
            The complementary set of walkers

        lnprob : numpy.ndarray
            The current ln-probability of the s0 set

        Returns
        -------
        newposition : numpy.ndarray
            The new positions of the walkers (same shape as s0)

        newlnprob : numpy.ndarray
            The ln-probability of the walkers at newposition

        accept : numpy.ndarray
            Array of bools indicating whether or not a walker's new position
            should be accepted

        """
        s = np.atleast_2d(s0)
        n0 = len(s)
        c = np.atleast_2d(comp)
        ncomp = len(comp)

        self.em = mixtures.MixtureModel(self._K, comp)
        self.em.run_kmeans()
        try:
            self.em.run_em()
        except mixtures.EMSingular:
            print "Singular: reducing K"
            self._K -= 1
            self.em = mixtures.MixtureModel(self._K, comp)
            self.em.run_kmeans()
            self.em.run_em()

        newposition = self.em.sample(n0)
        newposition[:,self._fixedinds] = self._fixedvals
        newlnprob = self.ensemble_lnposterior(newposition)

        newQ = self.em.lnprob(newposition)
        oldQ = self.em.lnprob(s)

        lnpdiff = oldQ - newQ + newlnprob - lnprob
        accept = (lnpdiff > np.log(self._random.rand(len(lnpdiff))))

        return newposition, newlnprob, accept

if __name__ == '__main__':
    pass

