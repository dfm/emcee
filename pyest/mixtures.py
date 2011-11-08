#!/usr/bin/env python
# encoding: utf-8
"""
Gaussian mixture models

"""

from __future__ import division

__all__ = ['MixtureModel']

import numpy as np

class MixtureModel(object):
    """
    Gaussian mixture model

    P data points in D dimensions with K clusters

    Shapes
    ------
    data -> (P, D)
    means -> (D, K)

    """
    def __init__(self, K, data):
        self._K    = K
        self._data = np.atleast_2d(data)
        self._lu   = None

        self._means = data[np.random.randint(data.shape[0],size=self._K),:].T
        self._cov   = [np.cov(data,rowvar=0)]*self._K
        self._as    = np.random.rand(K)
        self._as /= np.sum(self._as)

    @property
    def K(self):
        return self._K

    @K.setter
    def set_K(self, k):
        self._K = k

    @property
    def means(self):
        return self._means.T

    # ================= #
    # K-Means Algorithm #
    # ================= #

    def run_kmeans(self, maxiter=200, tol=1e-8, verbose=True):
        """
        Fit the given data using K-means

        """
        L = None
        for i in xrange(maxiter):
            newL = self._update_kmeans()
            if L is None:
                L = newL
            else:
                dL = np.abs(newL-L)
                if dL < tol:
                    break
                L = newL
        if i < maxiter-1:
            if verbose:
                print "K-Means converged after %d iterations"%(i)
        else:
            print "Warning: K-means didn't converge"

    def _update_kmeans(self):
        # dists.shape == (P,K)
        dists = np.sum((self._data[:,:,None] - self._means[None,:,:])**2, axis=1)

        # rs.shape == (P,K)
        rs = dists == np.min(dists,axis=1)[:,None]
        self._kmeans_rs = rs

        # self._means.shape == (D,K)
        self._means =  np.sum(rs[:,None,:] * self._data[:,:,None], axis=0)
        self._means /= np.sum(rs, axis=0)

        L = np.sum(rs*dists)
        return L

    # ============ #
    # EM Algorithm #
    # ============ #

    def run_em(self, maxiter=400, tol=1e-8, verbose=True):
        """
        Fit the given data using EM

        """
        L = None
        for i in xrange(maxiter):
            newL = self._expectation()
            self._maximization()
            if L is None:
                L = newL
            else:
                dL = np.abs(newL-L)
                if dL < tol:
                    break
                L = newL
        if i < maxiter-1:
            if verbose:
                print "EM converged after %d iterations"%(i)
        else:
            print "Warning: EM didn't converge after %d iterations"%(i+1)

    def _multi_gauss(self, k, X):
        # X.shape == (P,D)
        # self._means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        det = np.linalg.det(self._cov[k])

        # X1.shape == (P,D)
        X1 = X - self._means[None,:,k]

        # X2.shape == (P,D)
        X2 = np.linalg.solve(self._cov[k], X1.T).T

        p = -0.5*np.sum(X1 * X2, axis=1)

        return 1/np.sqrt( (2*np.pi)**(X.shape[1]) * det )*np.exp(p)

    def _expectation(self):
        # self._rs.shape == (P,K)
        L, self._rs = self._calc_prob(self._data)
        return np.sum(L, axis=0)

    def _maximization(self):
        # Nk.shape == (K,)
        Nk = np.sum(self._rs, axis=0)
        # self._means.shape == (D,K)
        self._means = np.sum(self._rs[:,None,:] * self._data[:,:,None], axis=0)
        self._means /= Nk[None,:]
        self._cov = []
        for k in range(self._K):
            # D.shape == (P,D)
            D = self._data - self._means[None,:,k]
            self._cov.append(np.dot(D.T, self._rs[:,k,None]*D)/Nk[k])
        self._as = Nk/self._data.shape[0]

    def _calc_prob(self, x):
        x = np.atleast_2d(x)
        rs = np.concatenate([self._as[k]*self._multi_gauss(k, x)
                for k in range(self._K)]).reshape((-1, self._K), order='F')
        L = np.log(np.sum(rs, axis=1))
        rs /= np.sum(rs, axis=1)[:,None]
        return L, rs

    def lnprob(self, x):
        return self._calc_prob(x)[0]

    def sample(self,N):
        samples = np.vstack(
                [np.random.multivariate_normal(self.means[k], self._cov[k],
                    size=int(self._as[k]*(N+1))) for k in range(self._K)])
        return samples[:N,:]

