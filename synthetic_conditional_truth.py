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
    construction: dict | None = None

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
        exposure = self.B.T @ self.w
        return (self.m @ exposure,
                np.einsum('i,jik,k->j', exposure, self.S, exposure)
                + np.sum(self.D * self.w**2))

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
        """Exact factor-based Gaussian draws, optionally portfolio-conditioned.

        Avoid a dense d-by-d decomposition, including at the paper's d=2048.
        Conditional correction uses Sigma_R w computed in factor coordinates.
        """
        weights=self.pi if c is None else self.component_probabilities(c)
        labels=rng.choice(len(weights),size=n,p=weights)
        out=np.empty((n,len(self.D)),dtype=np.float64)
        a,v=self.portfolio
        for j in range(len(weights)):
            ix=np.flatnonzero(labels==j)
            factors = rng.multivariate_normal(self.m[j], self.S[j], size=len(ix),
                                             check_valid='raise')
            x = factors @ self.B.T + rng.normal(0., np.sqrt(self.D), size=(len(ix), self.d))
            if c is not None:
                # 1-U is in (0,1]; ndtri_exp avoids underflow of U*Phi(z).
                logu=np.log1p(-rng.random(len(ix)))
                p=a[j]+np.sqrt(v[j])*ndtri_exp(logu+log_ndtr((c-a[j])/np.sqrt(v[j])))
                cross = self.B @ (self.S[j] @ (self.B.T @ self.w)) + self.D * self.w
                x += (p-x@self.w)[:,None]*(cross/v[j])
            out[ix]=x
        if not np.isfinite(out).all(): raise FloatingPointError('nonfinite truth sample')
        return (out,labels) if return_labels else out

    def sample_returns(self, n_samples, rng):
        """Return (returns, factors, labels) with R = B factors + epsilon.

        For the Chen et al. (2025) construction these are transformed factors F_tilde = A F,
        associated with stored orthonormal B; their covariance is generally full.
        """
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


GAUSSIAN_POPULATION_DESIGN = 'xu-gaussian-orthonormal-v1'
POPULATION_SEED = 912  # Project choice, independent of training/evaluation seeds.
QR_SIGN_CONVENTION = 'nonnegative diagonal of A; zero uses +1'


def validate_dimensions(d, k):
    """Gaussian populations support integer 3 <= k <= d."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (d, k)) or not 3 <= k <= d:
        raise ValueError('Synthetic population requires integer asset/factor counts with 3 <= k <= d')


def population_metadata(pop, case):
    if pop.construction is None:
        raise ValueError('Population lacks construction provenance')
    return dict(population_design=GAUSSIAN_POPULATION_DESIGN, population_d=pop.d,
                population_k=pop.k, population_case=case,
                population_seed=pop.construction['population_seed'],
                qr_sign_convention=QR_SIGN_CONVENTION)


def population_details(pop, case):
    """JSON-ready raw and transformed parameters; checkpoint identity stays small."""
    return dict(**population_metadata(pop, case),
                **{key: value.tolist() for key, value in pop.construction.items()
                   if isinstance(value, np.ndarray)},
                **{key: getattr(pop, key).tolist() for key in ('B', 'D', 'pi', 'm', 'S')})


def population_from_checkpoint(d, k, case, metadata):
    """Reconstruct using saved architecture, never the current experiment config.

    Old population designs and missing seed/sign metadata require fresh DFM/H
    training. A new seed specifies a new realization, not the paper's dataset.
    """
    design = GAUSSIAN_POPULATION_DESIGN
    expected = dict(population_design=design, population_d=d,
                    population_k=k, population_case=case,
                    qr_sign_convention=QR_SIGN_CONVENTION)
    seed = metadata.get('population_seed')
    valid_seed = isinstance(seed, Integral) and not isinstance(seed, bool) and seed >= 0
    if not valid_seed or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            'Incompatible or unverified synthetic population checkpoint: '
            f'expected d={d}, population k=DFM k={k}, case={case}. '
            'Retrain the DFM and H with the current population design, then regenerate comparisons.')
    return make_population(d, case, k=k, seed=seed)


def check_population_model(pop, model):
    """Accept an unconditional DFM or its frozen reference."""
    score = model._score if hasattr(model, '_score') else model.factor_score_network
    if (pop.d, pop.k) != (score.d, score.k) or tuple(score.V.shape) != pop.B.shape:
        raise ValueError('Population and DFM asset/factor dimensions differ; retrain the DFM and H')


def check_population_h(pop, h):
    check_population_model(pop, h.reference)
    expected_inputs = pop.k + 2 + int(h.log_remaining_time)
    if h.G(0.).shape != (pop.k + 1, pop.d) or h.network[0].in_features != expected_inputs:
        raise ValueError('H must use k+1 reduced coordinates plus its configured time features; retrain H')


def _gaussian_population(d, k, seed):
    """Chen et al. (2025) synthetic population, in exact orthonormal coordinates.

    Draw order is mu_F, beta, sigma_epsilon using NumPy default_rng (PCG64).
    Raw returns are modeled without the paper's U-Net preprocessing.
    """
    rng = np.random.default_rng(seed)
    mu_F = rng.uniform(0., .1, size=k)
    sigma_F = 1.5 * mu_F
    Sigma_F = np.diag(sigma_F**2)
    beta = rng.normal(size=(d, k))
    sigma_epsilon = rng.uniform(0., .4, size=d)
    B, A = np.linalg.qr(beta, mode='reduced')
    signs = np.where(np.diag(A) < 0., -1., 1.)
    B = B * signs
    A = signs[:, None] * A
    m = (A @ mu_F)[None, :]
    S = (A @ Sigma_F @ A.T)[None, :, :]
    construction = dict(population_seed=int(seed), beta=beta, mu_F=mu_F,
                        sigma_F=sigma_F, Sigma_F=Sigma_F,
                        sigma_epsilon=sigma_epsilon, A=A)
    return FactorPopulation(B, sigma_epsilon**2, np.ones(1), m, S, construction)


def make_population(d=50, case='gaussian', k=3, *, seed=POPULATION_SEED):
    """Fixed Chen et al. (2025) Gaussian population; seed is separate from observation streams."""
    validate_dimensions(d, k)
    if case != 'gaussian':
        raise ValueError('This branch supports only the Gaussian simulation')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('Population seed must be a nonnegative integer')
    return _gaussian_population(d, k, seed)
