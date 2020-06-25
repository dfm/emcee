# -*- coding: utf-8 -*-

"""Backend base class."""


from .. import autocorr
from ..state import State

__all__ = ["BackendBase"]


class BackendBase:
    """Backend base class. Not meant to be used directly."""

    # Methods to be implemented by children

    def __init__(self, dtype=None):
        raise NotImplementedError("Method must be implemented by child class.")

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        raise NotImplementedError("Method must be implemented by child class.")

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs."""
        raise NotImplementedError("Method must be implemented by child class.")

    @property
    def iteration(self):
        """Return the iteration number."""
        raise NotImplementedError("Method must be implemented by child class.")

    @property
    def initialized(self):
        """Return true if backend has been initialized."""
        raise NotImplementedError("Method must be implemented by child class.")

    @property
    def shape(self):
        """The dimensions of the ensemble ``(nwalkers, ndim)``"""
        raise NotImplementedError("Method must be implemented by child class.")

    def _get_value(self, name, flat, thin, discard):
        """Get a value from the backend."""
        raise NotImplementedError("Method must be implemented by child class.")

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current list of blobs. This is used to compute the
                dtype for the blobs array.

        """
        raise NotImplementedError("Method must be implemented by child class.")

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        raise NotImplementedError("Method must be implemented by child class.")

    @property
    def random_state(self):
        """Return the random state."""
        raise NotImplementedError("Method must be implemented by child class.")

    # Methods that *can* be overwritten by children

    def __enter__(self):
        """Enter method for context manager."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Exit method for context manager."""
        pass

    # Common methods

    def get_value(self, name, flat=False, thin=1, discard=0):
        """Get a value from the backend."""
        if not self.initialized or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        return self._get_value(name, flat=flat, thin=thin, discard=discard)

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        """Get the chain of blobs for each sample in the chain

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return self.get_value("log_prob", **kwargs)

    def get_last_sample(self):
        """Access the most recent sample in the chain"""
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        it = self.iteration
        blobs = self.get_blobs(discard=it - 1)
        if blobs is not None:
            blobs = blobs[0]
        return State(
            self.get_chain(discard=it - 1)[0],
            log_prob=self.get_log_prob(discard=it - 1)[0],
            blobs=blobs,
            random_state=self.random_state,
        )

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        x = self.get_chain(discard=discard, thin=thin)
        return thin * autocorr.integrated_time(x, **kwargs)

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def _check(self, state, accepted):
        self._check_blobs(state.blobs)
        nwalkers, ndim = self.shape
        has_blobs = self.has_blobs()
        if state.coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format(
                    (nwalkers, ndim)
                )
            )
        if state.log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and len(state.blobs) != nwalkers:
            raise ValueError(
                "invalid blobs size; expected {0}".format(nwalkers)
            )
        if accepted.shape != (nwalkers,):
            raise ValueError(
                "invalid acceptance size; expected {0}".format(nwalkers)
            )
