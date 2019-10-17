# -*- coding: utf-8 -*-


try:
    from ptemcee import Sampler as PTSampler
except ImportError:

    class PTSampler(object):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The PTSampler from emcee has been forked to "
                "https://github.com/willvousden/ptemcee, "
                "please install that package to continue using the PTSampler"
            )


__all__ = ["PTSampler"]
