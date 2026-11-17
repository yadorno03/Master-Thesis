"""Panel of daily US equity returns: selection of the assets and split of the days.

The prices are the Yahoo adjusted closes saved under data/raw (not tracked). The
panel keeps ``d`` current S&P 500 and S&P 400 members with a complete price
history, so it carries a survivorship bias: it describes the daily returns of
today's large companies, not an investable historical universe.
"""
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

PRICES = Path('data/raw/equities/yahoo_2026-10-06/adjusted_close.csv.gz')
UNIVERSE = Path('data/universes/us_equities_2026-10-06.csv')
PANEL = Path('data/processed/market_panel.npz')
# Index membership decides which complete histories enter first; ties are alphabetical.
INDEX_ORDER = ('S&P 500', 'S&P 400', 'S&P 600')
SPLITS = ('random', 'chronological')


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


@dataclass(frozen=True)
class Panel:
    """Simple daily returns [days, assets], oldest day first, with their dates and tickers."""
    returns: np.ndarray
    dates: np.ndarray
    tickers: tuple

    def __post_init__(self):
        returns = np.asarray(self.returns, dtype=np.float64)
        if returns.ndim != 2 or not np.isfinite(returns).all():
            raise ValueError('returns must be a finite [days, assets] matrix')
        if len(self.dates) != returns.shape[0] or len(self.tickers) != returns.shape[1]:
            raise ValueError('dates and tickers must match the rows and columns of returns')
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError('tickers must be distinct')
        object.__setattr__(self, 'returns', returns)
        object.__setattr__(self, 'dates', np.asarray(self.dates, dtype=str))
        object.__setattr__(self, 'tickers', tuple(str(ticker) for ticker in self.tickers))

    @property
    def d(self):
        return self.returns.shape[1]

    @property
    def weights(self):
        """The equal-weight portfolio."""
        return np.full(self.d, 1. / self.d)

    def first_assets(self, d):
        """The panel of its first d assets (reduced runs)."""
        d = _positive('d', d)
        if d > self.d:
            raise ValueError(f'The panel has {self.d} assets, fewer than {d}')
        return Panel(self.returns[:, :d], self.dates, self.tickers[:d])

    def sha256(self):
        digest = hashlib.sha256(np.ascontiguousarray(self.returns).tobytes())
        digest.update('|'.join((*self.dates, *self.tickers)).encode())
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, returns=self.returns, dates=self.dates, tickers=np.asarray(self.tickers))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            return cls(saved['returns'], saved['dates'], tuple(saved['tickers'].tolist()))


def build_panel(prices=PRICES, universe=UNIVERSE, *, d=500, days=6_250, max_zero_return_share=.05):
    """The last ``days`` daily returns of ``d`` assets with a complete price history.

    Returns are simple returns of adjusted closes. An asset needs a positive price
    on each of the last days + 1 sessions, and at most max_zero_return_share of
    unchanged prices, which removes stale quotes. Assets are taken by index
    (S&P 500, S&P 400, S&P 600), then alphabetically, until d are found.
    """
    d, days = _positive('d', d), _positive('days', days)
    if not 0 <= max_zero_return_share <= 1:
        raise ValueError('max_zero_return_share must lie in [0, 1]')
    prices = pd.read_csv(prices, index_col=0, parse_dates=True)
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Price sessions must be distinct and increasing')
    if len(prices) < days + 1:
        raise ValueError(f'{days} returns need {days + 1} sessions; the file has {len(prices)}')
    prices = prices.iloc[-(days + 1):]
    prices = prices.loc[:, prices.notna().all() & (prices > 0).all()]
    returns = prices.pct_change().iloc[1:]
    returns = returns.loc[:, (returns == 0).mean() <= max_zero_return_share]
    membership = pd.read_csv(universe).drop_duplicates('ticker').set_index('ticker').source_index
    rank = membership.reindex(returns.columns).map({name: i for i, name in enumerate(INDEX_ORDER)})
    tickers = sorted(rank.dropna().index, key=lambda ticker: (rank[ticker], ticker))
    if len(tickers) < d:
        raise ValueError(f'Only {len(tickers)} assets have a complete history, fewer than {d}')
    tickers = tickers[:d]
    return Panel(returns[tickers].to_numpy(), returns.index.strftime('%Y-%m-%d').to_numpy(), tuple(tickers))


def load_or_build_panel(path=PANEL, **settings):
    """The saved panel, built from the raw prices and saved on first use."""
    path = Path(path)
    if path.exists():
        return Panel.load(path)
    panel = build_panel(**settings)
    panel.save(path)
    return panel


@dataclass(frozen=True)
class Split:
    """Training, validation and evaluation days of a panel; the evaluation days are never used for fitting."""
    train: np.ndarray
    validation: np.ndarray
    evaluation: np.ndarray
    weights: np.ndarray
    info: dict

    @property
    def insample(self):
        """Training then validation days."""
        return np.concatenate((self.train, self.validation))


def split_panel(panel, n_train, n_val, n_eval, *, split='random', seed=912):
    """Split the last n_train + n_val + n_eval days of the panel.

    'random' assigns the days by a seeded permutation: the three sets are then
    exchangeable draws of the same period, the setting of the DFM, which models
    daily returns as independent draws of one law. 'chronological' takes the
    oldest days for training, the next ones for validation and the most recent
    ones for evaluation, so the evaluation also reflects changes of the law in time.
    """
    counts = [_positive(name, value) for name, value in (('n_train', n_train), ('n_val', n_val),
                                                         ('n_eval', n_eval))]
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    total = sum(counts)
    if total > len(panel.returns):
        raise ValueError(f'The panel has {len(panel.returns)} days, fewer than the {total} requested')
    days = np.arange(len(panel.returns) - total, len(panel.returns))
    if split == 'random':
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        days = np.random.default_rng(seed).permutation(days)
    parts = [np.sort(part) for part in np.split(days, np.cumsum(counts)[:-1])]
    info = dict(split=split, seed=int(seed) if split == 'random' else None, first_day=str(panel.dates[days.min()]),
                last_day=str(panel.dates[days.max()]), n_train=counts[0], n_val=counts[1], n_eval=counts[2],
                assets=panel.d, panel_sha256=panel.sha256(),
                evaluation_first_day=str(panel.dates[parts[2][0]]), evaluation_last_day=str(panel.dates[parts[2][-1]]))
    return Split(*(panel.returns[part] for part in parts), panel.weights, info)
