# pannet 0.1.0

Initial release. Native `torch` backend (replacing the earlier `mlp`-package
delegation).

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

## Bug fixes

* `id`/`time` arguments only worked when their *symbol name* matched a
  column literally called `id`/`time`, regardless of what was passed --
  an NSE (`substitute()`) scoping bug in `panel_resolve_name()`. Fixed by
  resolving via `match.call()` at the actual call site.
* `predict()` on held-out (never-seen-during-training) units returned `NA`
  for a substantial fraction of them even under the default `"zero"`
  new-id strategy -- fixed via a proper sentinel embedding row.

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
