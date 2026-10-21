# Shared return preprocessing

`return_preprocessing.ReturnTransform` owns the fitted means, sample standard
deviations, and asset order for both models. `author_data` is a compatibility
import, not another implementation. `NumpyStandardizer` is a NumPy facade over
that same object. Tensor operations use its saved dtype and device; vector
inputs support population-moment calculations and matrix inputs support samples.
U-Net image reshaping belongs to the author training adapter.

Fit only on training rows, after selecting the dataset split:

```python
from return_preprocessing import ReturnTransform, NumpyStandardizer

# The released recipe: convert observations to float32 BEFORE torch reductions.
shared = ReturnTransform.fit(training_returns, assets)
factor_coordinates = NumpyStandardizer.from_transform(shared)
train_x = factor_coordinates.transform(training_returns, assets=assets)
validation_x = factor_coordinates.transform(validation_returns, assets=assets)
test_x = shared(test_returns, assets=assets)
raw_generated = shared.inverse(generated_standardized, assets=assets)
event = shared.event(raw_weights, raw_threshold, assets=assets)

# Alternatively fit once using the research pipeline's float64 NumPy recipe.
research = NumpyStandardizer.fit(training_returns, assets)
shared_float64 = research.tensor_transform
```

A fit is never triggered by transform, inverse, event conversion, or loading
parameters. Pass the same fitted object to both pipelines to compare their
coordinates. When explicit column metadata is supplied, any order mismatch is
rejected. Unlabelled arrays/weights must already follow `assets`; no function can
detect an unannounced permutation. Portfolio conversion delegates to
`conditioning.portfolio_event_in_model_coordinates`, retaining the mean offset.

`equity_preprocessing.standardize_training` preserves the existing five-result
notebook API and chronological split, while using the shared NumPy facade.
It also accepts `transform=...` to reuse a fitted NumPy facade. This small adapter
was extracted from the locally present `equity_data.py`; downloading, filtering,
and tail resampling stay in that research module. The local
`synthetic_evaluation.Standardizer` subclasses the facade and retains its
specialized `ou_moments` method. Those larger, unpublished research modules are
not prerequisites and are not bundled into this PR. The factor checkpoint loader
uses the shared validation and preserves its public tuple return API.

Numerical compatibility:

- Default author fitting and transformation stay float32 and match the released
  recipe bitwise. The default author model and its parameters stay float32.
- NumPy fitting uses float64 mean and `std(ddof=1)`, preserving the equity
  pipeline's float64 results exactly. Float32 research inputs now explicitly
  use float64 statistics; callers needing the released float32 convention should
  use the author fit and share it through `from_transform`.
- The former synthetic fit used `sqrt(diag(np.cov(..., ddof=1)))`. The formula is
  equivalent, but the reduction order can change last bits. Tests compare it
  with a tight float64 tolerance; downstream retraining need not be bitwise
  identical. No covariance is now allocated merely to compute scaling.
- NumPy inverse/transform results follow the saved statistics precision. Default
  synthetic fits remain float64, including population OU calculations.
- Inputs must be finite real values, scales positive, and metadata unique and
  nonempty. Invalid boolean/complex values and empty/degenerate training data are
  rejected. Standardizer now supports a single asset; the separate synthetic
  covariance oracle retains its own dimensional restrictions.

Validation: `python -m unittest discover -s tests -v` (102 tests at PR #31,
with the optional author backend installed). Local synthetic oracle tests also
exercise the subclass and unchanged OU moments. See the stack audit for final
integration results and the merge procedure.
