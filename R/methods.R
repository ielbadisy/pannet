#' Print a fitted pannet model
#'
#' @param x A `"pannet"` object.
#' @param ... Unused.
#' @export
print.pannet <- function(x, ...) {
  cat("pannet fit\n")
  cat("Call:\n")
  print(x$call)
  cat("\n")
  cat("Family:", x$family, "\n")
  cat("Model:", x$model, "\n")
  cat("Observations:", x$data_info$n, "\n")
  cat("Individuals:", x$data_info$n_id, "\n")
  if (!is.na(x$data_info$n_time)) {
    cat("Time periods:", x$data_info$n_time, "\n")
  }
  if (!is.null(x$training$holdout_loss)) {
    cat("Holdout loss:", format(x$training$holdout_loss), "\n")
  }
  invisible(x)
}

#' Summarize a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return A list of model summary components.
#' @export
summary.pannet <- function(object, ...) {
  out <- list(
    call = object$call,
    family = object$family,
    model = object$model,
    n = object$data_info$n,
    n_id = object$data_info$n_id,
    n_time = object$data_info$n_time,
    balanced = object$data_info$balanced,
    training = object$training,
    parameters = object$parameters,
    x_names = object$x_names
  )
  class(out) <- "summary.pannet"
  out
}

#' @export
print.summary.pannet <- function(x, ...) {
  cat("Summary of pannet fit\n")
  cat("Family:", x$family, "\n")
  cat("Model:", x$model, "\n")
  cat("Observations:", x$n, "\n")
  cat("Individuals:", x$n_id, "\n")
  if (!is.na(x$n_time)) {
    cat("Time periods:", x$n_time, "\n")
  }
  cat("Balanced:", x$balanced, "\n")
  if (length(x$training$loss)) {
    cat("Final training loss:", tail(x$training$loss, 1), "\n")
  }
  if (!is.null(x$training$holdout_loss)) {
    cat("Holdout loss:", x$training$holdout_loss, "\n")
  }
  invisible(x)
}

#' Predict from a fitted pannet model
#'
#' The prediction scales are:
#'
#' - `type = "link"`: latent predictor `η`
#' - `type = "response"`: family-specific response scale
#' - `type = "prob"`: probability matrix for classification models
#' - `type = "class"`: predicted class labels for multiclass models
#'
#' For unseen panel units, `new_id_strategy` controls how the unit effect is
#' handled:
#'
#' - `"zero"`: set the new effect to 0
#' - `"mean"`: use the mean learned effect
#' - `"error"`: stop with an error
#'
#' @param object A `"pannet"` object.
#' @param newdata Optional data frame for prediction.
#' @param type Prediction scale.
#' @param new_id_strategy Strategy for unseen unit identifiers.
#' @param ... Unused.
#' @return A numeric vector, factor, or probability matrix depending on `type`
#'   and the model family.
#' @export
predict.pannet <- function(object, newdata = NULL, type = c("response", "link", "prob", "class"), new_id_strategy = c("zero", "mean", "error"), ...) {
  type <- match.arg(type)
  new_id_strategy <- match.arg(new_id_strategy)
  if (!is.null(newdata) && !is.data.frame(newdata)) {
    stop("`newdata` must be a data.frame.", call. = FALSE)
  }
  if (is.null(newdata)) {
    mm <- panel_build_design(
      data = object$raw_data,
      spec = object$spec,
      new_id_strategy = new_id_strategy
    )
    return(panel_predict_internal(object$fit, mm, object$family, type = type))
  }

  train_data <- object$raw_data
  predict_data <- newdata
  normalize_index <- function(x) {
    if (is.factor(x)) {
      x <- as.character(x)
    }
    num <- suppressWarnings(as.numeric(x))
    if (!anyNA(num) && all(!is.na(x))) {
      return(num)
    }
    as.character(x)
  }
  if (object$panel$id_name %in% names(train_data)) {
    train_data[[object$panel$id_name]] <- normalize_index(train_data[[object$panel$id_name]])
    predict_data[[object$panel$id_name]] <- normalize_index(predict_data[[object$panel$id_name]])
  }
  if (!is.null(object$panel$time_name) && object$panel$time_name %in% names(train_data)) {
    train_data[[object$panel$time_name]] <- normalize_index(train_data[[object$panel$time_name]])
    predict_data[[object$panel$time_name]] <- normalize_index(predict_data[[object$panel$time_name]])
  }
  factor_cols <- unique(c(names(object$panel$factor_levels), object$panel$factor_columns))
  factor_cols <- setdiff(factor_cols, c(object$panel$id_name, object$panel$time_name))
  for (nm in intersect(factor_cols, names(train_data))) {
    train_data[[nm]] <- as.character(train_data[[nm]])
  }
  for (nm in intersect(factor_cols, names(predict_data))) {
    predict_data[[nm]] <- as.character(predict_data[[nm]])
  }
  for (nm in setdiff(names(train_data), names(predict_data))) {
    predict_data[[nm]] <- NA
  }
  for (nm in setdiff(names(predict_data), names(train_data))) {
    train_data[[nm]] <- NA
  }
  predict_data <- predict_data[names(train_data)]
  combined <- rbind(train_data, predict_data)
  marker <- c(rep(FALSE, nrow(train_data)), rep(TRUE, nrow(predict_data)))
  combined$.pannet_predict_row <- marker
  combined <- panel_panel_order(combined, object$panel$id_name, object$panel$time_name)
  marker <- combined$.pannet_predict_row
  mm <- panel_build_design(
    data = combined,
    spec = object$spec,
    new_id_strategy = new_id_strategy
  )
  pred <- panel_predict_internal(object$fit, mm, object$family, type = type)
  if (is.matrix(pred)) {
    pred[marker, , drop = FALSE]
  } else {
    pred[marker]
  }
}

#' Fitted values for a pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return The fitted response values on the response scale.
#' @export
fitted.pannet <- function(object, ...) {
  object$fitted
}

#' Residuals from a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return Residuals on the response scale.
#' @export
residuals.pannet <- function(object, ...) {
  y <- object$spec$y_raw[seq_along(object$fitted)]
  y <- switch(
    object$family,
    gaussian = as.numeric(y),
    binomial = panel_binary_to_numeric(y),
    poisson = as.numeric(y),
    multiclass = stop("Residuals are not defined for multiclass pannet fits.", call. = FALSE),
    fractional = as.numeric(y)
  )
  y - object$fitted
}

#' Plot training history for a pannet model
#'
#' @param x A `"pannet"` object.
#' @param ... Unused.
#' @export
plot.pannet <- function(x, ...) {
  mlp::plot_history(x$fit)
}

#' Extract the training history from a pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return A list containing loss curves and training metadata.
#' @export
training_history <- function(object, ...) {
  UseMethod("training_history")
}

#' @export
training_history.pannet <- function(object, ...) {
  object$training
}
