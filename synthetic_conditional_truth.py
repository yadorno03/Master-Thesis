"""Float64 population benchmarks; no model-fitting dependency."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.optimize import brentq
from scipy.stats import norm, kstest

@dataclass
class FactorPopulation:
    B: np.ndarray
    D: np.ndarray
    pi: np.ndarray
    m: np.ndarray
    S: np.ndarray

    @property
    def d(self): return self.B.shape[0]
    @property
    def k(self): return self.B.shape[1]
    @property
    def mu(self): return self.m @ self.B.T
    @property
    def covs(self): return np.array([self.B @ s @ self.B.T + np.diag(self.D) for s in self.S])
    @property
    def w(self): return np.ones(len(self.D)) / len(self.D)
    @property
    def portfolio(self):
        return self.mu @ self.w, np.einsum('i,jik,k->j', self.w, self.covs, self.w)

    def cdf(self, x, c=None):
        a,v = self.portfolio
        x = np.asarray(x)
        if c is not None: x = np.minimum(x, c)
        logp = logsumexp(np.log(self.pi) + log_ndtr((x[...,None]-a)/np.sqrt(v)), axis=-1)
        if c is not None: logp -= self.log_event(c)
        return np.exp(logp)

    def threshold(self, probability):
        if not 0 < probability < 1: raise ValueError('probability must be interior')
        a,v = self.portfolio
        lo,hi = np.min(a-12*np.sqrt(v)), np.max(a+12*np.sqrt(v))
        while self.cdf(lo) > probability: lo -= hi-lo
        while self.cdf(hi) < probability: hi += hi-lo
        return brentq(lambda c: self.cdf(c)-probability, lo, hi, xtol=1e-13)

    def log_event(self,c):
        a,v=self.portfolio
        return logsumexp(np.log(self.pi)+log_ndtr((c-a)/np.sqrt(v)))

    def component_probabilities(self,c):
        a,v=self.portfolio
        return np.exp(np.log(self.pi)+log_ndtr((c-a)/np.sqrt(v))-self.log_event(c))

    def moments(self,c=None):
        means,covs,weights=self.mu.copy(),self.covs.copy(),self.pi
        if c is not None:
            a,v=self.portfolio; z=(c-a)/np.sqrt(v)
            lam=np.exp(norm.logpdf(z)-log_ndtr(z))
            cross=covs @ self.w
            means -= cross * (lam/np.sqrt(v))[:,None]
            covs -= np.einsum('ji,jk,j->jik',cross,cross,lam*(lam+z)/v)
            weights=self.component_probabilities(c)
        mean=weights @ means
        delta=means-mean
        cov=np.einsum('j,jik->ik',weights,covs)+np.einsum('j,ji,jk->ik',weights,delta,delta)
        return mean,cov

    def sample(self,n,rng,c=None,return_labels=False):
        """Original ambient Gaussian draw order, for Gaussian fitting.

        sample_returns supplies explicit latent factors using a separate draw
        construction; both methods have the same law, but different seeded rows.
        """
        weights=self.pi if c is None else self.component_probabilities(c)
        labels=rng.choice(len(weights),size=n,p=weights)
        out=np.empty((n,len(self.D)),dtype=np.float64)
        a,v=self.portfolio
        for j in range(len(weights)):
            ix=np.flatnonzero(labels==j)
            x=rng.multivariate_normal(self.mu[j],self.covs[j],size=len(ix))
            if c is not None:
                # 1-U is in (0,1]; ndtri_exp avoids underflow of U*Phi(z).
                logu=np.log1p(-rng.random(len(ix)))
                p=a[j]+np.sqrt(v[j])*ndtri_exp(logu+log_ndtr((c-a[j])/np.sqrt(v[j])))
                x += (p-x@self.w)[:,None]*(self.covs[j]@self.w/v[j])
            out[ix]=x
        if not np.isfinite(out).all(): raise FloatingPointError('nonfinite truth sample')
        return (out,labels) if return_labels else out

    def sample_returns(self, n_samples, rng):
        """Draw R = B F + epsilon and return (returns, factors, regime labels)."""
        labels = rng.choice(len(self.pi), size=n_samples, p=self.pi)
        factors = np.empty((n_samples, self.k), dtype=np.float64)
        for j in range(len(self.pi)):
            indices = np.flatnonzero(labels == j)
            if len(indices):
                factors[indices] = rng.multivariate_normal(
                    self.m[j], self.S[j], size=len(indices), check_valid='raise')
        eps = rng.normal(0., np.sqrt(self.D), size=(n_samples, self.d))
        returns = factors @ self.B.T + eps
        return returns, factors, labels

    def _return_array(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.d or not np.isfinite(values).all():
            raise ValueError(f'Returns must be a finite (N, {self.d}) array')
        return values

    def exact_conditional_cdf_given_F(self, R_eval, F_eval):
        """Asset PIT given the true factors, before any portfolio-event filtering.

        Estimated factors do not in general yield independent uniform PITs.
        """
        returns = self._return_array(R_eval)
        factors = np.asarray(F_eval, dtype=np.float64)
        if factors.shape != (len(returns), self.k) or not np.isfinite(factors).all():
            raise ValueError(f'Factors must be a finite (N, {self.k}) array matching returns')
        return norm.cdf((returns - factors @ self.B.T) / np.sqrt(self.D))

    def exact_marginal_cdf(self, R_eval):
        """Unconditional asset CDFs; PIT columns may remain mutually dependent."""
        returns = self._return_array(R_eval)
        result = np.zeros_like(returns)
        for weight, mean, covariance in zip(self.pi, self.m, self.S):
            variances = np.einsum('ij,jk,ik->i', self.B, covariance, self.B) + self.D
            result += weight * norm.cdf((returns - self.B @ mean) / np.sqrt(variances))
        return result

    def evaluate_generated_samples(self, R_gen):
        """Mean and population std of per-asset marginal PIT KS statistics.

        These are descriptive marginal diagnostics, not a joint-law test.
        """
        pit = self.exact_marginal_cdf(R_gen)
        if not len(pit):
            raise ValueError('At least one generated sample is required')
        statistics = np.array([kstest(column, 'uniform').statistic for column in pit.T])
        return float(statistics.mean()), float(statistics.std())

    def variance_contributions(self):
        _,cov=self.moments()
        residual=np.sum(self.D*self.w**2)
        total=self.w@cov@self.w
        return dict(factor=total-residual,residual=residual,total=total,
                    factor_exposure=(self.B.T@self.w).tolist())


GAUSSIAN_POPULATION_DESIGN = 'diagonal-regime-factors-v2'
POPULATION_SEED = 912


def validate_dimensions(d, k):
    """Gaussian populations support integer 3 <= k <= d."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (d, k)) or not 3 <= k <= d:
        raise ValueError('Synthetic population requires integer asset/factor counts with 3 <= k <= d')


def population_metadata(pop, case):
    design = GAUSSIAN_POPULATION_DESIGN
    return dict(population_design=design, population_d=pop.d,
                population_k=pop.k, population_case=case)


def population_from_checkpoint(d, k, case, metadata):
    """Reconstruct using saved architecture, never the current experiment config.

    Legacy checkpoints do not attest the population dimension (even if their
    DFM config says k=8). Reject those rather than infer it from model weights.
    """
    design = GAUSSIAN_POPULATION_DESIGN
    expected = dict(population_design=design, population_d=d,
                    population_k=k, population_case=case)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            'Incompatible or unverified synthetic population checkpoint: '
            f'expected d={d}, population k=DFM k={k}, case={case}. '
            'Retrain the DFM and H with the current population design, then regenerate comparisons.')
    return make_population(d, case, k=k)


def check_population_model(pop, model):
    """Accept an unconditional DFM or its frozen reference."""
    score = model._score if hasattr(model, '_score') else model.factor_score_network
    if (pop.d, pop.k) != (score.d, score.k) or tuple(score.V.shape) != pop.B.shape:
        raise ValueError('Population and DFM asset/factor dimensions differ; retrain the DFM and H')


def check_population_h(pop, h):
    check_population_model(pop, h.reference)
    if h.G(0.).shape != (pop.k + 1, pop.d) or h.network[0].in_features != pop.k + 2:
        raise ValueError('H must use k+1 reduced coordinates plus time; retrain H')


def _gaussian_population(d, k):
    """Original single-Gaussian coefficients and loading construction, unchanged."""
    rng = np.random.default_rng(POPULATION_SEED)
    e = np.ones(d) / np.sqrt(d)
    u = rng.normal(size=d)
    u -= e * (e @ u)
    u /= np.linalg.norm(u)
    b0 = .85 * e + np.sqrt(1 - .85**2) * u
    B = np.linalg.qr(np.column_stack([b0, rng.normal(size=(d, 2))]))[0]
    if B[:, 0] @ e < 0:
        B[:, 0] *= -1
    for _ in range(3, k):
        column = rng.normal(size=d)
        for _ in range(2):
            column -= B @ (B.T @ column)
        column /= np.linalg.norm(column)
        B = np.column_stack((B, column))
    m = np.zeros((1, k))
    m[0, 0] = .15
    variances = np.concatenate(([1.8, .8, .4], .4 / (np.arange(3, k) - 1)))
    return FactorPopulation(B, np.linspace(.12, .38, d), np.ones(1), m,
                            np.diag(variances)[None, :, :])


def make_population(d=50, case='gaussian', k=3):
    """Single Gaussian population with the original seeded coefficients."""
    validate_dimensions(d, k)
    if case != 'gaussian':
        raise ValueError('This branch supports only the Gaussian simulation')
    return _gaussian_population(d, k)
