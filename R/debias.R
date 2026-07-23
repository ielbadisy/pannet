#' Split-panel jackknife bias correction for dynamic pannet fits
#'
#' Any fixed-effects-style estimator with a lagged dependent variable is
#' inconsistent at finite T (Nickell, 1981); `pannet(model = "dynamic")` is
#' not exempt (see `vignette("pannet-validation")`'s dynamic-panel section).
#' Because `pannet` is a neural architecture, no closed-form bias correction
#' analogous to the LSDV bias formulas exists. This applies a model-agnostic
#' alternative instead: the split-panel jackknife (Dhaene & Jochmans, 2015).
#'
#' The panel is split into two half-panels by time, `model = "dynamic"` is
#' fit separately on each half and on the full panel, and the (average)
#' marginal effect of each variable is bias-corrected as
#'
#' `theta_corrected = 2 * theta_full - 0.5 * (theta_half1 + theta_half2)`
#'
#' This is an outer wrapper around three ordinary [pannet()] calls; it does
#' not modify the training loop itself.
#'
#' @param formula,data,id,time,... passed through to three separate
#'   [pannet()] calls (on the full panel and on each time-split half); `id`
#'   and `time` follow the same bare-column-name convention as `pannet()`.
#'   `model` must be `"dynamic"`; `time` is required (it defines the split).
#' @param variables character vector of covariates to compute bias-corrected
#'   marginal effects for (via [marginal_effects()]). Defaults to every
#'   non-id/time variable on the right-hand side of `formula`.
#' @return a list with `corrected` (named numeric vector, the bias-corrected
#'   average marginal effects), `full`/`half1`/`half2` (the uncorrected
#'   marginal effects each fit produced), and `fit_full`/`fit_half1`/
#'   `fit_half2` (the three underlying `pannet` fits)
#' @export
pannet_debias <- function(formula, data, id, time = NULL, ..., variables = NULL) {
  mc <- match.call()
  id_name <- panel_resolve_name(mc$id)
  time_name <- if (is.null(mc$time)) NULL else panel_resolve_name(mc$time)
  if (is.null(time_name)) {
    stop("pannet_debias() requires `time` (it defines the two half-panels ",
         "the split-panel jackknife averages over).", call. = FALSE)
  }

  extra_args <- list(...)
  if (!is.null(extra_args$model) && !identical(extra_args$model, "dynamic")) {
    stop("pannet_debias() only applies to model = 'dynamic' (the Nickell bias ",
         "it corrects for is specific to a lagged-dependent-variable panel model).",
         call. = FALSE)
  }
  extra_args$model <- "dynamic"

  fit_one <- function(d) {
    args <- c(list(formula = formula, data = d, id = as.name(id_name),
                    time = as.name(time_name)), extra_args)
    do.call(pannet, args)
  }

  times <- sort(unique(data[[time_name]]))
  if (length(times) < 4) {
    stop("pannet_debias() needs at least 4 distinct time points to form two ",
         "usable half-panels.", call. = FALSE)
  }
  mid <- times[ceiling(length(times) / 2)]
  half1 <- data[data[[time_name]] <= mid, , drop = FALSE]
  half2 <- data[data[[time_name]] > mid, , drop = FALSE]

  fit_full  <- fit_one(data)
  fit_half1 <- fit_one(half1)
  fit_half2 <- fit_one(half2)

  if (is.null(variables)) {
    rhs_vars <- all.vars(formula[[3]])
    variables <- setdiff(rhs_vars, c(id_name, time_name))
  }

  ame <- function(fit, v) {
    tryCatch(marginal_effects(fit, v)$marginal_effect, error = function(e) NA_real_)
  }

  full_ame  <- stats::setNames(vapply(variables, function(v) ame(fit_full, v), numeric(1)), variables)
  half1_ame <- stats::setNames(vapply(variables, function(v) ame(fit_half1, v), numeric(1)), variables)
  half2_ame <- stats::setNames(vapply(variables, function(v) ame(fit_half2, v), numeric(1)), variables)

  corrected <- 2 * full_ame - 0.5 * (half1_ame + half2_ame)

  list(corrected = corrected, full = full_ame, half1 = half1_ame, half2 = half2_ame,
       fit_full = fit_full, fit_half1 = fit_half1, fit_half2 = fit_half2)
}
