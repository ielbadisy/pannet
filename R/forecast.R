#' Recursive multi-step-ahead forecasting for dynamic pannet models
#'
#' `predict.pannet()` only supports one-step-ahead prediction for
#' `model = "dynamic"` fits: it needs the *true* lagged outcome, which
#' doesn't exist more than one period past the training data. Genuine
#' multi-step forecasting requires substituting the model's own prior
#' forecasts back in as the lag input for each subsequent step -- the
#' standard recursive/iterated forecasting scheme for autoregressive
#' models. This function does that: for each future time point (in
#' ascending order), it rebuilds the lag columns from the combined
#' history-plus-forecasts-so-far, predicts, and feeds that prediction
#' forward into the next step.
#'
#' @param object A fitted `"pannet"` object with `model = "dynamic"`.
#' @param newdata A data frame with one row per (id, time) to forecast,
#'   containing the id/time columns and the raw covariates (assumed known
#'   for future periods, the standard forecasting convention -- this
#'   function does not forecast the covariates themselves). The response
#'   column is not required (ignored if present).
#' @return A data frame with columns `id`, `time`, and `forecast`, one row
#'   per requested (id, time) pair, in ascending time order.
#' @export
pannet_forecast <- function(object, newdata) {
  if (!inherits(object, "pannet")) stop("`object` must be a fitted pannet model.", call. = FALSE)
  if (!identical(object$model, "dynamic")) {
    stop("pannet_forecast() is for model = 'dynamic' fits; ",
         "other models don't have a lag structure to iterate.", call. = FALSE)
  }
  if (!is.data.frame(newdata)) stop("`newdata` must be a data.frame.", call. = FALSE)

  id_nm  <- object$panel$id_name
  tm_nm  <- object$panel$time_name
  if (is.null(tm_nm)) stop("This model has no time column; forecasting requires one.", call. = FALSE)
  if (!id_nm %in% names(newdata)) stop("`newdata` must contain the id column '", id_nm, "'.", call. = FALSE)
  if (!tm_nm %in% names(newdata)) stop("`newdata` must contain the time column '", tm_nm, "'.", call. = FALSE)

  resp_nm <- object$spec$response_name
  if (!(resp_nm %in% names(newdata))) newdata[[resp_nm]] <- NA_real_

  hist_data <- object$raw_data

  # object$raw_data has id/time stored as factors whose levels only cover
  # the training period (irrelevant for "dynamic" models, which have no
  # id/time embeddings -- time is only used for lag bookkeeping here), so
  # normalize both to a plain numeric/character representation before
  # combining, or future time values silently become NA on assignment.
  normalize_idx <- function(x) {
    if (is.factor(x)) x <- as.character(x)
    num <- suppressWarnings(as.numeric(x))
    if (!anyNA(num) && !anyNA(x)) return(num)
    as.character(x)
  }
  hist_data[[id_nm]] <- normalize_idx(hist_data[[id_nm]])
  hist_data[[tm_nm]] <- normalize_idx(hist_data[[tm_nm]])
  newdata[[id_nm]]   <- normalize_idx(newdata[[id_nm]])
  newdata[[tm_nm]]   <- normalize_idx(newdata[[tm_nm]])

  for (nm in setdiff(names(hist_data), names(newdata))) newdata[[nm]]   <- NA
  for (nm in setdiff(names(newdata), names(hist_data))) hist_data[[nm]] <- NA
  newdata <- newdata[names(hist_data)]

  known_ids <- unique(hist_data[[id_nm]])
  unknown <- setdiff(unique(newdata[[id_nm]]), known_ids)
  if (length(unknown)) {
    stop("pannet_forecast() only supports units seen during training ",
         "(dynamic models need their history); unknown id(s): ",
         paste(utils::head(unknown, 5), collapse = ", "), ".", call. = FALSE)
  }

  combined <- rbind(hist_data, newdata)
  combined <- panel_panel_order(combined, id_nm, tm_nm)
  future_times <- sort(unique(newdata[[tm_nm]]))

  out <- vector("list", length(future_times))
  for (i in seq_along(future_times)) {
    ft <- future_times[[i]]
    # Rebuild lag columns from everything known so far: true history plus
    # forecasts filled in during earlier iterations of this loop.
    combined <- panel_rebuild_lags(combined, id_nm, tm_nm, object$spec$lag_columns)

    rows <- which(combined[[tm_nm]] == ft)
    # Build the design on the FULL history (not just this step's rows):
    # panel_build_design() unconditionally rebuilds lag columns from
    # whatever's in `data`, and a single time-slice has no prior row per
    # unit to lag from. Subset the resulting design *after*, matching how
    # predict.pannet() itself does it.
    design <- panel_build_design(combined, object$spec, new_id_strategy = "zero")
    id_idx   <- if (object$panel$use_id)   design$id_idx[rows]   else NULL
    time_idx <- if (object$panel$use_time) design$time_idx[rows] else NULL

    eta <- pannet_forward(object$network, design$x[rows, , drop = FALSE], id_idx, time_idx, object$device,
                           offset = design$offset[rows] / object$spec$y_scale)
    pred <- pannet_eta_to_response(eta, object$family, object$spec$y_center,
                                    object$spec$y_scale, "response", object$levels)

    combined[[resp_nm]][rows] <- pred
    out[[i]] <- data.frame(id = combined[[id_nm]][rows], time = ft, forecast = pred)
  }

  result <- do.call(rbind, out)
  rownames(result) <- NULL
  names(result)[1] <- id_nm
  names(result)[2] <- tm_nm
  result
}
