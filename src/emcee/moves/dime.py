# -*- coding: utf-8 -*-

import numpy as np

try:
    from scipy.special import logsumexp
    from scipy.stats import multivariate_t
except ImportError:
    multivariate_t = None

from .red_blue import RedBlueMove

__all__ = ["DIMEMove"]


def mvt_sample(df, mean, cov, size, random):
    """Sample from multivariate t distribution

    For reasons beyond my understanding, the results from random.multivariate_normal with non-identity covariance matrix are not reproducibel across architecture. Since scipy.stats.multivariate_t is based on numpy's multivariate_normal, the workaround is to crochet this manually. Advantage is that the scipy dependency drops out.
    """

    dim = len(mean)

    # draw samples
    snorm = random.randn(size, dim)
    chi2 = random.chisquare(df, size) / df

    # calculate sqrt of covariance
    svd_cov = np.linalg.svd(cov * (df - 2) / df)
    sqrt_cov = svd_cov[0] * np.sqrt(svd_cov[1]) @ svd_cov[2]

    return mean + snorm @ sqrt_cov / np.sqrt(chi2)[:, None]


class DIMEMove(RedBlueMove):
    r"""A proposal using adaptive differential-independence mixture ensemble MCMC.

    This is the `Differential-Independence Mixture Ensemble proposal` as developed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/dime_mcmc_boehl.pdf>`_.

    Args:
        sigma (Optional[float]): standard deviation of the Gaussian used to stretch the proposal vector.
        gamma (Optional[float]): mean stretch factor for the proposal vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
        aimh_prob (Optional[float]): probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to :math:`0.1`.
        df_proposal_dist (Optional[float]): degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to :math:`10`.
    """

    def __init__(
        self,
        sigma=1.0e-5,
        gamma=None,
        aimh_prob=0.1,
        df_proposal_dist=10,
        **kwargs
    ):
        if multivariate_t is None:
            raise ImportError(
                "you need scipy.stats.multivariate_t and scipy.special.logsumexp to use the DIMEMove"
            )

        self.sigma = sigma
        self.g0 = gamma
        self.aimh_prob = aimh_prob
        self.dft = df_proposal_dist

        kwargs["nsplits"] = 1
        super(DIMEMove, self).__init__(**kwargs)

    def setup(self, coords):
        # set some sane defaults

        nchain, npar = coords.shape

        if self.g0 is None:
            # pure MAGIC
            self.g0 = 2.38 / np.sqrt(2 * npar)

        if not hasattr(self, "prop_cov"):
            # even more MAGIC
            self.prop_cov = np.eye(npar)
            self.prop_mean = np.zeros(npar)
            self.accepted = np.ones(nchain, dtype=bool)
            self.cumlweight = -np.inf

    def propose(self, model, state):
        """Wrap original propose to get the some info on the current state"""

        self.lprobs = state.log_prob
        state, accepted = super(DIMEMove, self).propose(model, state)
        self.accepted = accepted
        return state, accepted

    def update_proposal_dist(self, x):
        """Update proposal distribution with ensemble `x`"""

        nchain, npar = x.shape

        # log weight of current ensemble
        lweight = (
            logsumexp(self.lprobs)
            + np.log(sum(self.accepted))
            - np.log(nchain)
        )

        # calculate stats for current ensemble
        ncov = np.cov(x.T, ddof=1)
        nmean = np.mean(x, axis=0)

        # update AIMH proposal distribution
        newcumlweight = np.logaddexp(self.cumlweight, lweight)
        self.prop_cov = (
            np.exp(self.cumlweight - newcumlweight) * self.prop_cov
            + np.exp(lweight - newcumlweight) * ncov
        )
        self.prop_mean = (
            np.exp(self.cumlweight - newcumlweight) * self.prop_mean
            + np.exp(lweight - newcumlweight) * nmean
        )
        self.cumlweight = newcumlweight

    def get_proposal(self, x, dummy, random):
        """Actual proposal function"""

        nchain, npar = x.shape

        # update AIMH proposal distribution
        self.update_proposal_dist(x)

        # differential evolution: draw the indices of the complementary chains
        i0 = np.arange(nchain) + random.randint(1, nchain, size=nchain)
        i1 = np.arange(nchain) + random.randint(1, nchain - 1, size=nchain)
        i1 += i1 >= i0
        # add small noise and calculate proposal
        f = self.sigma * random.randn(nchain)
        q = x + self.g0 * (x[i0 % nchain] - x[i1 % nchain]) + f[:, np.newaxis]
        factors = np.zeros(nchain, dtype=np.float64)

        # draw chains for AIMH sampling
        xchnge = random.rand(nchain) <= self.aimh_prob

        # draw alternative candidates and calculate their proposal density
        xcand = mvt_sample(
            df=self.dft,
            mean=self.prop_mean,
            cov=self.prop_cov,
            size=sum(xchnge),
            random=random,
        )
        lpropdold, lpropdnew = multivariate_t.logpdf(
            np.vstack((x[None, xchnge], xcand[None])),
            self.prop_mean,
            self.prop_cov * (self.dft - 2) / self.dft,
            df=self.dft,
        )

        # update proposals and factors
        q[xchnge, :] = np.reshape(xcand, (-1, npar))
        factors[xchnge] = lpropdold - lpropdnew

        return q, factors
