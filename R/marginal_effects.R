#' Compute marginal effects for a pannet model
#'
#' The average marginal effect of variable `x_j` is approximated by finite
#' differences:
#'
#' `AME_j ≈ mean{ [μ(x_j + ε) - μ(x_j)] / ε }`
#'
#' For multiclass models, marginal effects are returned on the probability
#' scale, one row per class.
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @export
marginal_effects <- function(object, ...) {
  UseMethod("marginal_effects")
}

#' Compute a finite-difference marginal effect for one covariate
#'
#' The effect is approximated by perturbing `x_j` by `eps` and recomputing the
#' fitted mean or class probability.
#'
#' @param object A fitted `"pannet"` object.
#' @param variable Name of the numeric covariate to perturb.
#' @param eps Finite-difference step size.
#' @param newdata Optional data frame used as the baseline for the effect.
#' @param ... Unused.
#' @return A data frame with the estimated marginal effect.
#' @export
marginal_effects.pannet <- function(object, variable, eps = 1e-4, newdata = NULL, ...) {
  if (missing(variable) || !nzchar(variable)) {
    stop("`variable` is required.", call. = FALSE)
  }
  base <- if (is.null(newdata)) object$raw_data else newdata
  if (!variable %in% names(base)) {
    stop("`variable` not found in data.", call. = FALSE)
  }
  if (!is.numeric(base[[variable]])) {
    stop("`variable` must be numeric for finite-difference marginal effects.", call. = FALSE)
  }
  perturbed <- base
  perturbed[[variable]] <- perturbed[[variable]] + eps
  if (object$family == "multiclass") {
    p0 <- predict(object, newdata = base, type = "prob")
    p1 <- predict(object, newdata = perturbed, type = "prob")
    eff <- (p1 - p0) / eps
    out <- data.frame(
      variable = variable,
      class = colnames(eff),
      marginal_effect = colMeans(eff, na.rm = TRUE),
      scale = "probability"
    )
    rownames(out) <- NULL
    return(out)
  }
  x0 <- predict(object, newdata = base, type = "response")
  x1 <- predict(object, newdata = perturbed, type = "response")
  eff <- (x1 - x0) / eps
  data.frame(
    variable = variable,
    marginal_effect = mean(eff, na.rm = TRUE),
    scale = "response"
  )
}
