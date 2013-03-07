#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling demo based on: http://www.youtube.com/watch?v=Vv3f0QNWvWQ
and: https://github.com/duvenaud/harlemcmc-shake

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import itertools
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as pl
import emcee


class Mixture(object):

    def __init__(self, means, covs, weights=None):
        if weights is None:
            weights = np.ones(len(means), dtype=float)

        assert len(means) == len(weights) and len(means) == len(covs)

        self.K = len(means)
        self.means = [np.array(m) for m in means]
        self.icovs = [linalg.inv(c) for c in covs]
        self.logws = [np.log(0.5 * w / np.pi) - 0.5 * linalg.slogdet(c)[1]
                            for w, c in zip(weights / np.sum(weights), covs)]

    def __call__(self, p):
        p = np.array(p)
        ds = [p - m for m in self.means]
        lp = np.array([w - 0.5 * np.dot(d, np.dot(ic, d))
                for w, d, ic in zip(self.logws, ds, self.icovs)])
        a = np.max(lp)
        return a + np.log(np.sum(np.exp(lp - a)))


def rotate_cov(c, th):
    cth, sth = np.cos(th), np.sin(th)
    r = np.array([[cth, -sth], [sth, cth]])
    return np.dot(r.T, np.dot(c, r))


# Set up the sampler.
ndim, nwalkers = 2, 100

# Build the letters.
mixes = []
samplers = []
ics = []


def build_mixture(means, covs):
    m = Mixture(means, covs)
    mixes.append(m)
    samplers.append(emcee.EnsembleSampler(nwalkers, ndim, m))
    ics.append(np.array([4 * np.random.rand(2) - 2 for n in range(nwalkers)]))


# The letter "H".
vert_cov = [[0.005, 0], [0, 0.6]]
horz_cov = [[0.6, 0], [0, 0.01]]
build_mixture([[-1.5, 0], [0, 0], [1.5, 0]], [vert_cov, horz_cov, vert_cov])

# The letter "A".
c1 = [[0.45, 0], [0, 0.01]]
c2 = [[0.01, 0], [0, 1.2]]
build_mixture([[0, -1], [-0.9, 0], [0.9, 0]],
              [c1, rotate_cov(c2, np.pi / 8.), rotate_cov(c2, -np.pi / 8.)])

# The letter "R".
c1 = [[0.01, 0], [0, 0.75]]
c2 = [[1.2, 0], [0, 0.01]]
build_mixture([[-1.5, 0], [0.25, 1.25], [0.25, 0.5], [0.25, -0.9]],
              [c1, rotate_cov(c2, np.pi / 12), rotate_cov(c2, -np.pi / 12),
               rotate_cov(c2, np.pi / 8)])

# The letter "L".
c1 = [[0.01, 0], [0, 0.6]]
c2 = [[0.6, 0], [0, 0.01]]
build_mixture([[-1.5, 0], [0, -1.5]], [c1, c2])

# The spherical case.
build_mixture([[0, 0]], [[[1, 0], [0, 1]]])

# The letter "M".
c1 = [[0.005, 0], [0, 0.6]]
c2 = [[0.01, 0], [0, 1.0]]
build_mixture([[-1.5, 0], [1.5, 0], [-0.7, 0.2], [0.7, 0.2]],
              [c1, c1, rotate_cov(c2, -np.pi / 6.),
               rotate_cov(c2, np.pi / 6.)])

# The letter "S".
th = np.linspace(-0.5 * np.pi, np.pi, 50)
means = (np.vstack([0.1 - np.cos(th) * 1.75, np.sin(th) * 0.9 + 0.9])).T
means = np.concatenate([means, -means])
covs = [[[0.01, 0], [0, 0.01]]] * 2 * len(th)
build_mixture(means, covs)

# The second letter "H".
vert_cov = [[0.005, 0], [0, 0.6]]
horz_cov = [[1.2, 0], [0, 0.01]]
build_mixture([[-1.5, 0], [0, 0], [1.5, 0]], [vert_cov, horz_cov, vert_cov])

# The letter "K".
c1 = [[0.01, 0], [0, 0.75]]
c2 = [[1.2, 0], [0, 0.01]]
build_mixture([[-1.5, 0], [-0.1, -0.8], [-0.1, 0.8]],
              [c1, rotate_cov(c2, np.pi / 6), rotate_cov(c2, -np.pi / 6)])

# Set up the axes.
nx, ny = 3, 3
fig = pl.figure(figsize=[10 * nx / 3, 10 * ny / 3])
axes = [fig.add_axes((xi / nx, (ny - yi - 1) / ny, 1 / nx, 1 / ny),
                     frameon=True, xticks=[], yticks=[])
                for yi, xi in itertools.product(range(ny), range(nx))]

# Plot the letters and initial coniditions.
x, y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
points = []
for i, mix in enumerate(mixes):
    Z = np.exp([mix([a, b])
                for a, b in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    axes[i].imshow(Z, interpolation="nearest", cmap="gray",
                   extent=[-2, 2, 2, -2])
    p, = axes[i].plot(ics[i][:, 0], ics[i][:, 1], "or")
    points.append(p)
    axes[i].set_xlim(-2, 2)
    axes[i].set_ylim(-2, 2)

# Start the samplers and iterators.
iterations = 200
gens = [s.sample(p, iterations=iterations) for s, p in zip(samplers, ics)]

# Iterate.
try:
    os.makedirs("harlemcmc")
except os.error:
    pass
pl.savefig("harlemcmc/{0:04d}.png".format(0))
for i in range(iterations):
    pos = [g.next()[0] for g in gens]
    [(el.set_xdata(p[:, 0]), el.set_ydata(p[:, 1]))
                for el, p in zip(points, pos)]
    pl.draw()
    pl.savefig("harlemcmc/{0:04d}.png".format(i + 1))

# ffmpeg -i harlmcmc/%4d.png -r 12 -vcodec libx264 harlmcmc.mp4
