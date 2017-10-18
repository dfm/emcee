# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HDFBackend", "TempHDFBackend"]

import os
from tempfile import NamedTemporaryFile

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .. import __version__
from .backend import Backend


class HDFBackend(Backend):

    def __init__(self, filename, name="mcmc"):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name

    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def reset(self, nwalkers, ndim):
        with self.open("w") as f:
            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset("accepted", data=np.zeros(nwalkers, dtype=int))
            g.create_dataset("chain",
                             (0, nwalkers, ndim),
                             maxshape=(None, nwalkers, ndim),
                             dtype=np.float64)
            g.create_dataset("log_prob",
                             (0, nwalkers),
                             maxshape=(None, nwalkers),
                             dtype=np.float64)

    def has_blobs(self):
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("You must run the sampler with "
                                     "'store == True' before accessing the "
                                     "results")

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard+thin-1:self.iteration:thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)
            return v

    @property
    def shape(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def random_state(self):
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, delta_N, blobs):
        """Expand the storage space by ``N``"""
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            N = g.attrs["iteration"] + delta_N
            g["chain"].resize(N, axis=0)
            g["log_prob"].resize(N, axis=0)
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                    g.create_dataset("blobs", (N, nwalkers),
                                     maxshape=(None, nwalkers),
                                     dtype=dt)
                else:
                    g["blobs"].resize(N, axis=0)
                g.attrs["has_blobs"] = True

    def save_step(self, coords, log_prob, blobs, accepted, random_state):
        """Save a step to the backend"""
        self._check(coords, log_prob, blobs, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = coords
            g["log_prob"][iteration, :] = log_prob
            if blobs is not None:
                g["blobs"][iteration, :] = blobs
            g["accepted"][:] += accepted

            for i, v in enumerate(random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1


class TempHDFBackend(object):

    def __enter__(self):
        f = NamedTemporaryFile("w", delete=False)
        f.close()
        self.filename = f.name
        return HDFBackend(f.name, "test")

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
