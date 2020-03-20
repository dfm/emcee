# -*- coding: utf-8 -*-

import warnings
from functools import wraps

import numpy as np

__all__ = ["sample_ball", "deprecated", "deprecation_warning"]


def deprecation_warning(msg):
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def deprecated(alternate):
    def wrapper(func, alternate=alternate):
        msg = "'{0}' is deprecated.".format(func.__name__)
        if alternate is not None:
            msg += " Use '{0}' instead.".format(alternate)

        @wraps(func)
        def f(*args, **kwargs):
            deprecation_warning(msg)
            return func(*args, **kwargs)

        return f

    return wrapper


@deprecated(None)
def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.

    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.

    """
    assert len(p0) == len(std)
    return np.vstack(
        [p0 + std * np.random.normal(size=len(p0)) for i in range(size)]
    )


@deprecated(None)
def sample_ellipsoid(p0, covmat, size=1):
    """
    Produce an ellipsoid of walkers around an initial parameter value,
    according to a covariance matrix.

    :param p0: The initial parameter value.
    :param covmat:
        The covariance matrix.  Must be symmetric-positive definite or
        it will raise the exception numpy.linalg.LinAlgError
    :param size: The number of samples to produce.

    """
    return np.random.multivariate_normal(
        np.atleast_1d(p0), np.atleast_2d(covmat), size=size
    )

try:
    Generator = np.random.Generator
except AttributeError:
    Generator = None

def rng_integers(gen, low, **kwargs):
    """
    Generate integers from a Generator or RandomState.

    :param gen: Generator or RandomState object
    :param low: Passed as first argument to gen.integers() or to
        gen.randint() depending of its type.
    :param kwargs: Passed to gen.integers() or to gen.randint()
        depending on its type.

    """
    if isinstance(gen, Generator):
        return gen.integers(low, **kwargs)
    return gen.randint(low, **kwargs)

def _check_random_state(seed, state):
    """Check seed argument and set RandomState.

    Based on scikit-learn utils/validation.py.
    """
    if isinstance(seed, (int, np.integer)) or seed is None:
        random = np.random.mtrand.RandomState()
        random.set_state(state)
        if seed is not None:
            random.seed(seed)
        return random
    if isinstance(seed, np.random.RandomState):
        return seed
    try:
    # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise TypeError(
        "seed must be an int, np.random.RandomState, np.random.Generator or "
        "None of type {}".format(type(seed))
    )
