## Submission

This is the first submission of `pannet` to CRAN.

`pannet` fits panel and longitudinal regression models with a multilayer
perceptron (native `torch` backend) in place of the linear index used by
classical fixed- and random-effects estimators, while keeping additive
learnable individual/time effects. Validated against `panglm`/`plm`/a
GAMM comparator on nonlinear covariate-effect recovery; includes
`pannet_debias()` (split-panel jackknife bias correction for dynamic
panels) and `pannet_forecast()` (recursive multi-step-ahead forecasting).

## Test environments

* local: Ubuntu 24.04, R 4.5.1 (via `R CMD check --as-cran`)
* win-builder / R-hub: to be run before submission

## R CMD check results

See below for the latest local `R CMD check --as-cran` result.

## Downstream dependencies

None (new package).
