#' Compute marginal effects for a pannet model
#'
#' The average marginal effect of `x_j` is approximated by finite differences:
#'
#' `AME_j = mean{ [mu(x_j + eps) - mu(x_j)] / eps }` (approximately)
#'
#' For multiclass models, one marginal effect per class is returned.
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @export
marginal_effects <- function(object, ...) UseMethod("marginal_effects")

#' @rdname marginal_effects
#' @param variable Name of the numeric covariate to perturb.
#' @param eps Finite-difference step size.
#' @param newdata Optional baseline data frame. When `NULL` the training data
#'   are used.
#' @return A data frame with columns `variable`, `marginal_effect`, and `scale`
#'   (plus `class` for multiclass models).
#' @export
marginal_effects.pannet <- function(object, variable, eps = 1e-4, newdata = NULL, ...) {
  if (missing(variable) || !nzchar(variable)) {
    stop("`variable` must be a non-empty string naming a numeric covariate.", call. = FALSE)
  }
  base <- if (is.null(newdata)) object$raw_data else newdata
  if (!variable %in% names(base)) {
    stop(paste0("`variable` '", variable, "' not found in data."), call. = FALSE)
  }
  if (!is.numeric(base[[variable]])) {
    stop("`variable` must be numeric for finite-difference marginal effects.", call. = FALSE)
  }

  perturbed            <- base
  perturbed[[variable]] <- perturbed[[variable]] + eps

  if (object$family == "multiclass") {
    p0  <- predict(object, newdata = base,      type = "prob")
    p1  <- predict(object, newdata = perturbed, type = "prob")
    eff <- (p1 - p0) / eps
    out <- data.frame(
      variable       = variable,
      class          = colnames(eff),
      marginal_effect = colMeans(eff, na.rm = TRUE),
      scale          = "probability",
      stringsAsFactors = FALSE
    )
    rownames(out) <- NULL
    return(out)
  }

  x0  <- predict(object, newdata = base,      type = "response")
  x1  <- predict(object, newdata = perturbed, type = "response")
  eff <- (as.numeric(x1) - as.numeric(x0)) / eps

  data.frame(
    variable        = variable,
    marginal_effect = mean(eff, na.rm = TRUE),
    scale           = "response",
    stringsAsFactors = FALSE
  )
}
