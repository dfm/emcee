# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HDFBackend", "TempHDFBackend", "does_hdf5_support_longdouble"]

import os
from tempfile import NamedTemporaryFile

import numpy as np

from .. import __version__
from .backend import Backend


try:
    import h5py
except ImportError:
    h5py = None


def does_hdf5_support_longdouble():
    if h5py is None:
        return False
    with NamedTemporaryFile(
        prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
    ) as f:
        f.close()

        with h5py.File(f.name, "w") as hf:
            g = hf.create_group("group")
            g.create_dataset("data", data=np.ones(1, dtype=np.longdouble))
            if g["data"].dtype != np.longdouble:
                return False
        with h5py.File(f.name, "r") as hf:
            if hf["group"]["data"].dtype != np.longdouble:
                return False
    return True


class HDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str; optional): The name of the group where the chain will
            be saved.
        read_only (bool; optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.

    """

    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

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
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        f = h5py.File(self.filename, mode)
        if not self.dtype_set and self.name in f:
            g = f[self.name]
            if "chain" in g:
                self.dtype = g["chain"].dtype
                self.dtype_set = True
        return f

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset(
                "accepted",
                data=np.zeros(nwalkers),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    def has_blobs(self):
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]
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

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    dt = np.dtype((blobs.dtype, blobs.shape[1:]))
                    g.create_dataset(
                        "blobs",
                        (ntot, nwalkers),
                        maxshape=(None, nwalkers),
                        dtype=dt,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                else:
                    g["blobs"].resize(ntot, axis=0)
                    if g["blobs"].dtype.shape != blobs.shape[1:]:
                        raise ValueError(
                            "Existing blobs have shape {} but new blobs "
                            "requested with shape {}"
                            .format(g["blobs"].dtype.shape, blobs.shape[1:]))
                g.attrs["has_blobs"] = True

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = state.coords
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1


class TempHDFBackend(object):
    def __init__(self, dtype=None, compression=None, compression_opts=None):
        self.dtype = dtype
        self.filename = None
        self.compression = compression
        self.compression_opts = compression_opts

    def __enter__(self):
        f = NamedTemporaryFile(
            prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
        )
        f.close()
        self.filename = f.name
        return HDFBackend(
            f.name,
            "test",
            dtype=self.dtype,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
