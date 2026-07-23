#' Print a fitted pannet model
#'
#' @param x A `"pannet"` object.
#' @param ... Unused.
#' @export
print.pannet <- function(x, ...) {
  cat("pannet fit\n")
  cat("Call:\n")
  print(x$call)
  cat("\nFamily:", x$family, "  Model:", x$model, "\n")
  cat("Observations:", x$data_info$n,
      "  Individuals:", x$data_info$n_id, "\n")
  if (!is.na(x$data_info$n_time)) cat("Time periods:", x$data_info$n_time, "\n")
  if (x$panel$use_id)   cat("Individual effects (n_id =", x$panel$n_id, ")\n")
  if (x$panel$use_time) cat("Time effects       (n_time =", x$panel$n_time, ")\n")
  cat("Best epoch:", x$training$best_epoch,
      "  Best val loss:", format(x$training$best_validation_loss, digits = 4), "\n")
  if (!is.null(x$training$holdout_loss)) {
    cat("Holdout loss:", format(x$training$holdout_loss, digits = 4), "\n")
  }
  invisible(x)
}

#' Summarise a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return A list of class `"summary.pannet"`.
#' @export
summary.pannet <- function(object, ...) {
  out <- list(
    call        = object$call,
    family      = object$family,
    model       = object$model,
    n           = object$data_info$n,
    n_id        = object$data_info$n_id,
    n_time      = object$data_info$n_time,
    balanced    = object$data_info$balanced,
    use_id      = object$panel$use_id,
    use_time    = object$panel$use_time,
    n_id_eff    = object$panel$n_id,
    n_time_eff  = object$panel$n_time,
    training    = object$training,
    parameters  = object$parameters,
    x_names     = object$x_names
  )
  class(out) <- "summary.pannet"
  out
}

#' @export
print.summary.pannet <- function(x, ...) {
  cat("Summary of pannet fit\n")
  cat("Family:", x$family, "  Model:", x$model, "\n")
  cat("Observations:", x$n, "  Individuals:", x$n_id, "\n")
  if (!is.na(x$n_time)) cat("Time periods:", x$n_time, "\n")
  cat("Balanced:", x$balanced, "\n")
  if (x$use_id)   cat("Individual effects:", x$n_id_eff, "\n")
  if (x$use_time) cat("Time effects:      ", x$n_time_eff, "\n")
  hist <- x$training$history
  if (!is.null(hist) && nrow(hist) > 0L) {
    cat("Final train loss:", tail(hist$train_loss, 1L), "\n")
    cat("Final valid loss:", tail(hist$valid_loss,  1L), "\n")
  }
  cat("Best epoch:", x$training$best_epoch, "\n")
  if (!is.null(x$training$holdout_loss)) {
    cat("Holdout loss:", x$training$holdout_loss, "\n")
  }
  invisible(x)
}

#' Predict from a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param newdata Optional data frame for prediction. When `NULL` returns
#'   training-set fitted values.
#' @param type Prediction scale: `"response"`, `"link"`, `"prob"`, or `"class"`.
#' @param new_id_strategy How to handle IDs not seen during training:
#'   `"zero"` (set effect to 0), `"mean"` (use mean learned effect), or
#'   `"error"` (stop with an error).
#' @param ... Unused.
#' @return Numeric vector, factor, or probability matrix depending on `type`.
#' @export
predict.pannet <- function(
  object, newdata = NULL,
  type = c("response","link","prob","class"),
  new_id_strategy = c("zero","mean","error"),
  ...
) {
  type            <- match.arg(type)
  new_id_strategy <- match.arg(new_id_strategy)

  if (!is.null(newdata) && !is.data.frame(newdata)) {
    stop("`newdata` must be a data.frame.", call. = FALSE)
  }

  if (is.null(newdata)) {
    design <- panel_build_design(object$raw_data, object$spec, new_id_strategy)
    id_idx   <- if (object$panel$use_id)   design$id_idx   else NULL
    time_idx <- if (object$panel$use_time) design$time_idx else NULL
    .pannet_update_sentinels(object$network, new_id_strategy,
                              object$panel$n_id, object$panel$n_time)
    eta <- pannet_forward(object$network, design$x, id_idx, time_idx, object$device)
    return(pannet_eta_to_response(eta, object$family, object$spec$y_center,
                                   object$spec$y_scale, type, object$levels))
  }

  # Align newdata with training data structure
  train_data   <- object$raw_data
  predict_data <- newdata

  normalize_idx <- function(x) {
    if (is.factor(x)) x <- as.character(x)
    num <- suppressWarnings(as.numeric(x))
    if (!anyNA(num) && !anyNA(x)) return(num)
    as.character(x)
  }

  id_nm <- object$panel$id_name
  if (id_nm %in% names(train_data)) {
    train_data[[id_nm]]   <- normalize_idx(train_data[[id_nm]])
    predict_data[[id_nm]] <- normalize_idx(predict_data[[id_nm]])
  }
  tm_nm <- object$panel$time_name
  if (!is.null(tm_nm) && tm_nm %in% names(train_data)) {
    train_data[[tm_nm]]   <- normalize_idx(train_data[[tm_nm]])
    predict_data[[tm_nm]] <- normalize_idx(predict_data[[tm_nm]])
  }

  for (nm in setdiff(names(train_data),   names(predict_data))) predict_data[[nm]] <- NA
  for (nm in setdiff(names(predict_data), names(train_data)))   train_data[[nm]]   <- NA
  predict_data <- predict_data[names(train_data)]

  combined <- rbind(train_data, predict_data)
  marker   <- c(rep(FALSE, nrow(train_data)), rep(TRUE, nrow(predict_data)))
  combined$.pannet_predict_row <- marker
  combined <- panel_panel_order(combined, id_nm, tm_nm)
  marker   <- combined$.pannet_predict_row

  design <- panel_build_design(combined, object$spec, new_id_strategy)
  id_idx   <- if (object$panel$use_id)   design$id_idx   else NULL
  time_idx <- if (object$panel$use_time) design$time_idx else NULL

  # For "mean" strategy, update the sentinel embedding row to mean of known effects
  .pannet_update_sentinels(object$network, new_id_strategy,
                            object$panel$n_id, object$panel$n_time)

  eta <- pannet_forward(object$network, design$x, id_idx, time_idx, object$device)
  pred_full <- pannet_eta_to_response(eta, object$family, object$spec$y_center,
                                       object$spec$y_scale, type, object$levels)

  if (is.matrix(pred_full)) pred_full[marker, , drop = FALSE] else pred_full[marker]
}

#' Fitted values for a pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return Fitted response values.
#' @export
fitted.pannet <- function(object, ...) object$fitted

#' Residuals from a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return Response-scale residuals.
#' @export
residuals.pannet <- function(object, ...) {
  if (object$family == "multiclass") {
    stop("Residuals are not defined for multiclass pannet fits.", call. = FALSE)
  }
  y_raw <- object$spec$y_raw[object$panel$complete_rows]
  y_num <- switch(object$family,
    gaussian   = as.numeric(y_raw),
    binomial   = panel_binary_to_numeric(y_raw),
    poisson    = as.numeric(y_raw),
    fractional = as.numeric(y_raw)
  )
  fitted_v <- if (is.matrix(object$fitted)) object$fitted[, 1L] else as.numeric(object$fitted)
  y_num - fitted_v
}

#' Plot training history for a pannet model
#'
#' @param x A `"pannet"` object.
#' @param ... Unused.
#' @export
plot.pannet <- function(x, ...) {
  hist <- x$training$history
  if (is.null(hist) || nrow(hist) == 0L) {
    message("No training history available.")
    return(invisible(x))
  }
  old_par <- graphics::par(mfrow = c(1L, 1L))
  on.exit(graphics::par(old_par))
  ylim <- range(c(hist$train_loss, hist$valid_loss), na.rm = TRUE)
  graphics::plot(hist$epoch, hist$train_loss, type = "l", col = "steelblue",
    xlab = "Epoch", ylab = "Loss", main = "Training history", ylim = ylim, lwd = 2)
  graphics::lines(hist$epoch, hist$valid_loss, col = "firebrick", lwd = 2)
  graphics::legend("topright", legend = c("Train", "Validation"),
    col = c("steelblue", "firebrick"), lwd = 2, bty = "n")
  graphics::abline(v = x$training$best_epoch, lty = 2, col = "grey50")
  invisible(x)
}

#' Extract training history from a pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return A list with loss curves and metadata.
#' @export
training_history <- function(object, ...) UseMethod("training_history")

#' @export
training_history.pannet <- function(object, ...) object$training

# Update sentinel embedding rows (index n_id+1, n_time+1) before prediction.
# "zero" strategy: sentinel stays at zero (default initialisation).
# "mean" strategy: sentinel set to mean of all real effect rows.
.pannet_update_sentinels <- function(model, strategy, n_id, n_time) {
  if (identical(strategy, "mean")) {
    if (!is.null(model$id_emb) && n_id > 0L) {
      torch::with_no_grad({
        real_rows  <- model$id_emb$weight[1:n_id, ]
        mean_row   <- real_rows$mean(1, keepdim = TRUE)
        model$id_emb$weight[n_id + 1L, ] <- mean_row$squeeze(1)
      })
    }
    if (!is.null(model$time_emb) && n_time > 0L) {
      torch::with_no_grad({
        real_rows  <- model$time_emb$weight[1:n_time, ]
        mean_row   <- real_rows$mean(1, keepdim = TRUE)
        model$time_emb$weight[n_time + 1L, ] <- mean_row$squeeze(1)
      })
    }
  }
  invisible(NULL)
}

#' Retrieve learned individual effects from a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return Named numeric vector of learned individual effects, or `NULL`
#'   for pooled/dynamic/time models.
#' @export
individual_effects <- function(object, ...) UseMethod("individual_effects")

#' @export
individual_effects.pannet <- function(object, ...) {
  if (!object$panel$use_id) return(NULL)
  n  <- object$panel$n_id
  w  <- as.array(object$network$id_emb$weight$to(device = "cpu")$detach())
  w  <- w[seq_len(n), , drop = FALSE]  # exclude sentinel row
  if (ncol(w) == 1L) {
    stats::setNames(as.numeric(w), object$spec$id_levels)
  } else {
    rownames(w) <- object$spec$id_levels
    w
  }
}

#' Retrieve learned time effects from a fitted pannet model
#'
#' @param object A `"pannet"` object.
#' @param ... Unused.
#' @return Named numeric vector of learned time effects, or `NULL`.
#' @export
time_effects <- function(object, ...) UseMethod("time_effects")

#' @export
time_effects.pannet <- function(object, ...) {
  if (!object$panel$use_time) return(NULL)
  n  <- object$panel$n_time
  w  <- as.array(object$network$time_emb$weight$to(device = "cpu")$detach())
  w  <- w[seq_len(n), , drop = FALSE]  # exclude sentinel row
  if (ncol(w) == 1L) {
    stats::setNames(as.numeric(w), object$spec$time_levels)
  } else {
    rownames(w) <- object$spec$time_levels
    w
  }
}
