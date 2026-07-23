# pannet 0.1.0

Initial release. Native `torch` backend (replacing the earlier `mlp`-package
delegation).

## Validation

* Nonlinear covariate-effect recovery benchmark (5 seeds x 2 DGPs, see
  `vignette("pannet-validation")` and `README.md`): `pannet` beats an exact
  linear panel FE estimator (`panglm`) by a replicated 17-24% margin. The
  fair panel-model comparator is a GAMM (`gam(... + s(id, bs = "re"))`),
  not a pooled GAM with no individual effects -- an earlier draft of this
  benchmark used the latter and overstated the gap to a purpose-built
  additive model; corrected before release. Against the GAMM, `pannet`
  ties on the harder DGP and trails slightly on the easier one.
  `tests/testthat/test-nonlinear-recovery.R` is a permanent regression
  guard on the `panglm` comparison.

## Dynamic panel bias

* `pannet_debias()`: split-panel jackknife bias correction (Dhaene &
  Jochmans, 2015) for `model = "dynamic"` fits. Any lagged-dependent-
  variable panel estimator is inconsistent at finite T (Nickell, 1981);
  since `pannet` is a neural architecture, no closed-form LSDV-style
  correction applies, so this fits the full panel and two time-split
  half-panels separately and corrects the average marginal effect of each
  covariate as `2*full - 0.5*(half1 + half2)`. A model-agnostic outer
  wrapper around three ordinary `pannet()` calls; it does not change the
  training loop.

## Features

* MLP backbone with additive learnable individual/time embeddings; `model =
  "pooled"`, `"individual"`, `"time"`, `"twoway"`, `"dynamic"`.
* Families: gaussian, binomial, poisson, multiclass, fractional.
* `offset()` support in the formula (e.g. `y ~ x1 + offset(log(exposure))`),
  added to the linear predictor after the MLP+embeddings, the standard
  `glm()` convention. Correctly rescaled for gaussian (where `y` is
  standardized before training). Not supported with `family = "multiclass"`.
  `predict()` requires `newdata` to supply the same offset variable(s) if
  the model was fit with one.
* `pannet_benchmark()` now includes `panglm`'s FE/RE estimators (gaussian,
  poisson) alongside `plm`/`pglm`, in addition to the pannet configuration
  grid.
* `pannet_forecast()`: genuine recursive multi-step-ahead forecasting for
  `model = "dynamic"` fits. `predict()` only ever supported one-step-ahead
  prediction (it needs the *true* lagged outcome, which stops existing more
  than one period past training); this iterates forward, feeding each
  step's own forecast back in as the next step's lag input -- the standard
  scheme for autoregressive forecasting.

## Bug fixes

* `id`/`time` arguments only worked when their *symbol name* matched a
  column literally called `id`/`time`, regardless of what was passed --
  an NSE (`substitute()`) scoping bug in `panel_resolve_name()`. Fixed by
  resolving via `match.call()` at the actual call site.
* `predict()` on held-out (never-seen-during-training) units returned `NA`
  for a substantial fraction of them even under the default `"zero"`
  new-id strategy -- fixed via a proper sentinel embedding row.
* **`predict()` on `model = "dynamic"` fits silently used zero instead of
  the true lagged outcome/covariates for any newly-supplied time period.**
  Two compounding bugs: (1) `model.matrix()`'s default `na.action` (`na.omit`)
  silently dropped any row with an `NA` lag value -- which every row of
  newly-added data has before its lag is computed -- corrupting the
  row-count alignment `predict.pannet()` relies on; (2) the lag-rebuild
  step was skipped whenever the lag *column names* were already present
  (inherited via `rbind()` with training data), even though their *values*
  were still `NA` placeholders for the new rows. Net effect: one-step-ahead
  dynamic predictions were computed from a zeroed-out history rather than
  the real one. Both are fixed; `predict()` and `pannet_forecast()` now
  agree exactly on one-step-ahead forecasts (see `test-forecast.R`).

## Validation findings (see `vignette("pannet-validation")`)

Systematic checks against `panglm`/`plm` on realistic panel sizes (small N,
short T, sparse counts, dynamic panels) surfaced real limitations, not just
confirmation that the package runs:

* On a heavily zero-inflated, sparse count panel (N=75, T=6), `pannet`'s
  marginal effect for the covariate was biased downward by roughly a third
  relative to the true simulated value, while `panglm`'s exact conditional-
  MLE FE estimator recovered it closely. This bias persisted essentially
  unchanged across network sizes (`hidden = c(64,32)` down to a linear net)
  and with/without dropout -- **it is not an overfitting/capacity issue**,
  and tuning `hidden`/`dropout` does not fix it. The likely cause is
  incidental-parameters-type bias from jointly estimating one embedding per
  individual under sparsity, rather than profiling fixed effects out in
  closed form.
* `model = "dynamic"`'s lagged-outcome coefficient is substantially biased
  (consistent with the well-known Nickell 1981 dynamic panel bias, which
  affects any fixed-effects estimator with a lagged dependent variable and
  finite T -- not `pannet`-specific, and not yet corrected here). Covariate
  slopes in the same dynamic fits were comparatively more reliable.
* Marginal-effect estimates can vary meaningfully across random seeds,
  especially for small true effects, where the seed-to-seed range can
  exceed the true effect's magnitude. Refit across a few seeds before
  trusting a marginal effect close to zero.
* No clear overfitting signal was found comparing default vs. reduced
  hidden-layer sizes on a small (N=75, T=6) nonlinear DGP -- the existing
  early-stopping mechanism appears to already guard against this reasonably
  well in that test.

## Known limitations

* No bias correction for dynamic-panel (lagged-dependent-variable) models.
* Individual/time fixed effects are jointly estimated (dummy/embedding
  style), not profiled out in closed form -- see the sparse-count finding
  above.
* Not yet on CRAN (`panglm` and `glmmTMB` comparisons in tests/vignettes are
  Suggests-gated).
