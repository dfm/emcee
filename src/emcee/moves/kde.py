# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


__all__ = ["KDEMove"]


class KDEMove(RedBlueMove):
    """A proposal using a KDE of the complementary ensemble

    This is a simplified version of the method used in `kombine
    <https://github.com/bfarr/kombine>`_. If you use this proposal, you should
    use *a lot* of walkers in your ensemble.

    Args:
        bw_method: The bandwidth estimation method. See `the scipy docs
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_
            for allowed values.

    """

    def __init__(self, bw_method=None, **kwargs):
        if gaussian_kde is None:
            raise ImportError(
                "you need scipy.stats.gaussian_kde to use the KDEMove"
            )
        self.bw_method = bw_method
        super(KDEMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        kde = gaussian_kde(c.T, bw_method=self.bw_method)
        q = kde.resample(len(s))
        factor = kde.logpdf(s.T) - kde.logpdf(q)
        return q.T, factor
