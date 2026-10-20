"""Chronological equity split adapter; mathematics lives in return_preprocessing."""
from return_preprocessing import NumpyStandardizer


def standardize_training(returns, *, transform=None):
    """Keep the notebook's five-result API and its 2000–2019 training window.

    A supplied fitted transform is reused, with exact column-order validation.
    Validation/test data can use transform.transform directly without refitting.
    """
    train = returns.loc['2000-01-01':'2019-12-31']
    test = returns.loc['2020-01-01':'2025-12-31']
    if len(train) < 2 or test.empty:
        raise ValueError('Both chronological splits must be populated.')
    if transform is None:
        transform = NumpyStandardizer.fit(train.to_numpy(), assets=train.columns)
    if not isinstance(transform, NumpyStandardizer):
        raise TypeError('Expected NumpyStandardizer')
    standardized = transform.transform(train.to_numpy(), assets=train.columns)
    return train, test, transform.mean, transform.std, standardized
