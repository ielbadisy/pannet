# pannet

Neural regression for panel data — **for when the covariate-response
relationship is genuinely nonlinear.**

Classical panel estimators (`plm`, `pglm`, [`panglm`](https://github.com/ielbadisy/panglm))
are fast and exact, but structurally assume a linear index `x'β`. When the
true relationship is `sin(x1) + x2²` instead, no amount of correctly-fitting
that linear model closes the gap — it's a specification problem, not an
estimation problem. `pannet` replaces the linear index with an MLP, while
keeping the panel structure (additive learnable individual/time effects)
that a plain neural network would ignore.

## Validated, not just claimed

Independently re-run across 5 simulated panels each (N=60, T=10, held-out
individuals), recovering the *true noiseless signal* — not just fitting
noisy data — for two nonlinear DGPs:

| Estimator | `sin(x1) + 0.5·x2² + x3` | `sin(2πx1)·cos(x2) + tanh(2x2) + exp(-x3²)` |
|---|---|---|
| `panglm` (exact linear panel FE) | 1.579 | 1.543 |
| `pannet` (MLP + panel effects) | **1.213** (−23%) | **1.335** (−13%) |
| `mgcv::gam` (semiparametric baseline) | 1.124 | 1.288 |

(Mean out-of-sample RMSE against the true signal, lower is better, 5
independent replications per DGP.)

`panglm`'s linear index cannot represent `sin(x1)` or `x2²` regardless of
how well it's estimated — that gap is a specification problem, not an
estimation one. `pannet` closes ~80% of the gap between the linear
estimator and a purpose-built GAM, consistently across both DGPs and all 5
seeds each — a replicated result, not a single lucky draw. See
`vignette("pannet-validation")` for the full methodology.

## Installation

```r
# install.packages("remotes")
remotes::install_github("ielbadisy/pannet")
```

## Quick example

```r
library(pannet)

fit <- pannet(
  y ~ x1 + x2 + x3,
  data   = your_panel,
  id     = unit_id,
  time   = period,
  family = "gaussian",
  model  = "individual"   # additive individual effects; "twoway" adds time effects
)

predict(fit, newdata = new_panel)
marginal_effects(fit, variable = "x1")
```

## When to use pannet (and when not to)

**Use it when:** the covariate-response relationship is plausibly
nonlinear and you have a reasonable number of periods per unit. That's the
one case validated above where it beats a linear panel estimator by a
meaningful, replicated margin.

**Reach for `panglm`/`plm`/`pglm` instead when:** the effect is plausibly
linear (they're faster, exact, and more precisely estimated there), the
panel is short (T ≲ 10) and sparse/zero-inflated (a known bias mode, not
fixed by tuning — see the validation vignette), you need a dynamic
(lagged-outcome) *forecast* (a linear AR+FE benchmark currently forecasts
better at realistic panel lengths), or you need inference with a small
true effect on a single fit (marginal effects can vary seed-to-seed by
more than the effect itself — refit across a few seeds).

This isn't hedging for its own sake — the validation vignette is where
these limits were actually found and fixed where possible; see `NEWS.md`
for what changed as a result.

## Learn more

- `vignette("pannet-introduction")` — API tour, model types, `offset()`,
  extractor compatibility by model.
- `vignette("pannet-validation")` — the benchmark above in full, plus the
  sparse-count, dynamic-panel, seed-stability, and multi-step-forecasting
  checks that shaped the guidance above.
- `NEWS.md` — what's fixed, what's still open.
