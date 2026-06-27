panel_resolve_name <- function(x) {
  expr <- substitute(x)
  if (is.symbol(expr)) {
    return(deparse(expr))
  }
  if (is.character(expr)) {
    return(expr[[1L]])
  }
  stop("`id` and `time` must be column references or single strings.", call. = FALSE)
}

panel_assert_family <- function(family) {
  family <- match.arg(family, c("gaussian", "binomial", "poisson", "multiclass", "fractional"))
  family
}

panel_assert_model <- function(model) {
  model <- match.arg(model, c("pooled", "individual", "time", "twoway", "dynamic"))
  model
}

panel_assert_split <- function(split) {
  match.arg(split, c("by_id", "by_time", "random"))
}

panel_binary_to_numeric <- function(y) {
  if (is.factor(y)) {
    as.integer(y == levels(y)[nlevels(y)])
  } else {
    as.numeric(y)
  }
}

panel_panel_order <- function(data, id_name, time_name = NULL) {
  if (is.null(time_name)) {
    ord <- order(data[[id_name]], seq_len(nrow(data)))
  } else {
    ord <- order(data[[id_name]], data[[time_name]], seq_len(nrow(data)))
  }
  data[ord, , drop = FALSE]
}

panel_add_index_columns <- function(data, id_name, time_name = NULL) {
  data[[id_name]] <- factor(data[[id_name]])
  if (!is.null(time_name)) {
    data[[time_name]] <- factor(data[[time_name]])
  } else {
    data[[".pannet_time"]] <- factor(seq_len(nrow(data)))
    time_name <- ".pannet_time"
  }
  list(data = data, id_name = id_name, time_name = time_name)
}

panel_make_lags <- function(data, id_name, time_name, vars, lags) {
  if (is.null(lags) || length(lags) == 0L || length(vars) == 0L) {
    return(list(data = data, lag_cols = character()))
  }
  lags <- sort(unique(as.integer(lags)))
  lag_cols <- character()
  groups <- split(seq_len(nrow(data)), data[[id_name]])
  for (var in vars) {
    if (!is.numeric(data[[var]])) {
      next
    }
    for (lag in lags) {
      out <- rep(NA_real_, nrow(data))
      for (idx in groups) {
        if (length(idx) <= lag) {
          next
        }
        out[idx[(lag + 1L):length(idx)]] <- data[[var]][idx[seq_len(length(idx) - lag)]]
      }
      nm <- paste0("lag", lag, "_", var)
      data[[nm]] <- out
      lag_cols <- c(lag_cols, nm)
    }
  }
  list(data = data, lag_cols = lag_cols)
}

panel_rebuild_lags <- function(data, id_name, time_name, lag_columns) {
  if (!length(lag_columns)) {
    return(data)
  }
  lag_info <- regexec("^lag([0-9]+)_(.+)$", lag_columns)
  lag_parts <- regmatches(lag_columns, lag_info)
  lag_parts <- lag_parts[lengths(lag_parts) > 0L]
  if (!length(lag_parts)) {
    return(data)
  }
  vars <- unique(vapply(lag_parts, `[[`, character(1), 3L))
  lags <- unique(as.integer(vapply(lag_parts, `[[`, character(1), 2L)))
  data <- panel_panel_order(data, id_name, time_name)
  rebuilt <- panel_make_lags(data, id_name, time_name, vars = vars, lags = lags)
  rebuilt$data
}

panel_split_indices <- function(data, id_name, time_name, split, validation) {
  n <- nrow(data)
  if (is.null(validation) || validation <= 0) {
    return(list(train = seq_len(n), valid = integer()))
  }
  validation <- max(0, min(0.9, validation))
  if (split == "random") {
    valid <- sort(sample.int(n, size = max(1L, floor(n * validation))))
    train <- setdiff(seq_len(n), valid)
    return(list(train = train, valid = valid))
  }
  if (split == "by_id") {
    ids <- unique(data[[id_name]])
    valid_ids <- sample(ids, size = max(1L, floor(length(ids) * validation)))
    valid <- which(data[[id_name]] %in% valid_ids)
    train <- setdiff(seq_len(n), valid)
    return(list(train = train, valid = valid))
  }
  times <- unique(data[[time_name]])
  valid_times <- sample(times, size = max(1L, floor(length(times) * validation)))
  valid <- which(data[[time_name]] %in% valid_times)
  train <- setdiff(seq_len(n), valid)
  list(train = train, valid = valid)
}

panel_transform_response <- function(y, family) {
  switch(
    family,
    gaussian = as.numeric(y),
    binomial = {
      if (is.factor(y)) {
        if (nlevels(y) != 2L) stop("binomial response must have exactly two levels.", call. = FALSE)
        factor(y)
      } else {
        y_num <- as.integer(as.numeric(y))
        if (!all(y_num %in% c(0L, 1L), na.rm = TRUE)) {
          stop("binomial response must be coded as 0/1 or a two-level factor.", call. = FALSE)
        }
        factor(y_num, levels = c(0L, 1L))
      }
    },
    poisson = log1p(as.numeric(y)),
    multiclass = factor(y),
    fractional = {
      y <- as.numeric(y)
      if (any(y < 0 | y > 1, na.rm = TRUE)) {
        stop("fractional response must be in [0, 1].", call. = FALSE)
      }
      qlogis(pmin(pmax(y, 1e-6), 1 - 1e-6))
    }
  )
}

panel_inverse_response <- function(yhat, family, positive_level = "1") {
  switch(
    family,
    gaussian = yhat,
    binomial = yhat,
    poisson = expm1(yhat),
    multiclass = yhat,
    fractional = plogis(yhat)
  )
}

panel_metric_name <- function(family) {
  switch(family,
    gaussian = "rmse",
    binomial = "accuracy",
    poisson = "rmse",
    multiclass = "accuracy",
    fractional = "rmse"
  )
}

panel_build_design <- function(data, spec, training = FALSE, new_id_strategy = c("zero", "mean", "error")) {
  new_id_strategy <- match.arg(new_id_strategy)
  required <- unique(c(spec$design_vars, spec$factor_columns, spec$id_name, spec$time_name))
  required <- required[required %in% names(data)]
  data <- data[, required, drop = FALSE]

  for (nm in spec$factor_columns) {
    levs <- spec$factor_levels[[nm]]
    if (!nm %in% names(data)) {
      next
    }
    data[[nm]] <- factor(as.character(data[[nm]]), levels = levs)
  }
  if (new_id_strategy == "error" && any(vapply(spec$factor_columns, function(nm) nm %in% names(data) && anyNA(data[[nm]]), logical(1)))) {
    stop("New data contains unseen factor levels and `new_id_strategy = \"error\"`.", call. = FALSE)
  }
  for (nm in spec$numeric_columns) {
    if (nm %in% names(data)) {
      data[[nm]] <- as.numeric(data[[nm]])
    }
  }

  if (length(spec$lag_columns) && !all(spec$lag_columns %in% names(data))) {
    data <- panel_rebuild_lags(data, spec$id_name, spec$time_name, spec$lag_columns)
  }

  if (length(spec$lag_columns)) {
    for (nm in spec$lag_columns) {
      data[[nm]] <- as.numeric(data[[nm]])
    }
  }

  if (length(spec$center_columns)) {
    for (nm in spec$center_columns) {
      if (nm %in% names(data)) {
        center <- spec$center[[nm]]
        scale <- spec$scale[[nm]]
        data[[nm]] <- (as.numeric(data[[nm]]) - center) / scale
      }
    }
  }

  mm_terms <- delete.response(terms(spec$design_formula))
  mm <- model.matrix(mm_terms, data = data)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }
  if (anyNA(mm)) {
    if (new_id_strategy == "mean") {
      for (nm in colnames(mm)) {
        idx <- is.na(mm[, nm])
        if (any(idx)) {
          mm[idx, nm] <- spec$column_means[[nm]]
        }
      }
    } else {
      mm[is.na(mm)] <- 0
    }
  }
  mm
}

panel_prepare_spec <- function(formula, data, id_name, time_name, model, family, lags, standardize) {
  id_levels <- sort(unique(as.character(data[[id_name]])))
  data[[id_name]] <- factor(as.character(data[[id_name]]), levels = id_levels)
  factor_levels <- list()
  factor_columns <- character()
  numeric_columns <- character()

  if (!is.null(time_name)) {
    time_levels <- sort(unique(as.character(data[[time_name]])))
    data[[time_name]] <- factor(as.character(data[[time_name]]), levels = time_levels)
  } else {
    time_levels <- character()
  }

  extra_terms <- character()
  if (model %in% c("individual", "twoway")) {
    extra_terms <- c(extra_terms, id_name)
    factor_levels[[id_name]] <- levels(data[[id_name]])
    factor_columns <- c(factor_columns, id_name)
  }
  if (model %in% c("time", "twoway")) {
    if (is.null(time_name)) {
      stop("`time` is required for `model = \"time\"` or `model = \"twoway\"`.", call. = FALSE)
    }
    extra_terms <- c(extra_terms, time_name)
    factor_levels[[time_name]] <- levels(data[[time_name]])
    factor_columns <- c(factor_columns, time_name)
  }

  if (model == "dynamic" && length(lags)) {
    lag_vars <- all.vars(delete.response(terms(formula)))
    lag_vars <- setdiff(lag_vars, c(id_name, time_name))
    lagged <- panel_make_lags(data, id_name, time_name, lag_vars, lags)
    data <- lagged$data
    extra_terms <- c(extra_terms, lagged$lag_cols)
  } else {
    lagged <- list(lag_cols = character())
  }

  base_terms <- delete.response(terms(formula))
  if (length(extra_terms)) {
    design_formula <- update(formula, paste(". ~ . +", paste(extra_terms, collapse = " + ")))
  } else {
    design_formula <- formula
  }

  mf <- model.frame(design_formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  mm <- model.matrix(delete.response(terms(design_formula)), data = mf)
  if ("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  }

  if (standardize) {
    mm_center <- rep(0, ncol(mm))
    mm_scale <- rep(1, ncol(mm))
    names(mm_center) <- colnames(mm)
    names(mm_scale) <- colnames(mm)
    standardized_cols <- character()
    for (nm in colnames(mm)) {
      col <- mm[, nm]
      if (is.numeric(col) && !all(col %in% c(0, 1), na.rm = TRUE)) {
        mm_center[nm] <- mean(col, na.rm = TRUE)
        mm_scale[nm] <- stats::sd(col, na.rm = TRUE)
        if (is.na(mm_scale[nm]) || mm_scale[nm] == 0) {
          mm_scale[nm] <- 1
        }
        mm[, nm] <- (col - mm_center[nm]) / mm_scale[nm]
        standardized_cols <- c(standardized_cols, nm)
      }
    }
  } else {
    mm_center <- setNames(rep(0, ncol(mm)), colnames(mm))
    mm_scale <- setNames(rep(1, ncol(mm)), colnames(mm))
    standardized_cols <- character()
  }

  factor_cols_in_formula <- names(Filter(is.factor, mf))
  numeric_cols_in_formula <- names(Filter(is.numeric, mf))
  numeric_cols_in_formula <- setdiff(numeric_cols_in_formula, c(id_name, time_name))

  spec <- list(
    formula = formula,
    design_formula = design_formula,
    design_vars = all.vars(delete.response(terms(design_formula))),
    response_name = all.vars(formula[[2L]]),
    task = if (family %in% c("binomial", "multiclass")) "classification" else "regression",
    family = family,
    model = model,
    id_name = id_name,
    time_name = time_name,
    id_levels = id_levels,
    time_levels = time_levels,
    factor_levels = factor_levels,
    factor_columns = unique(c(factor_columns, factor_cols_in_formula)),
    numeric_columns = numeric_cols_in_formula,
    lag_columns = lagged$lag_cols,
    lag_base_vars = if (exists("lag_vars", inherits = FALSE)) lag_vars else character(),
    all_columns = names(mf),
    center_columns = standardized_cols,
    center = as.list(mm_center),
    scale = as.list(mm_scale),
    column_means = setNames(colMeans(mm, na.rm = TRUE), colnames(mm)),
    x_names = colnames(mm),
    x_train = mm,
    y_raw = y,
    y_train = panel_transform_response(y, family),
    prepared_data = data
  )
  class(spec) <- "pannet_spec"
  spec
}

panel_eval_loss <- function(y_true, y_pred, family) {
  switch(
    family,
    gaussian = mean((y_true - y_pred)^2, na.rm = TRUE),
    binomial = mean((panel_binary_to_numeric(y_true) - y_pred)^2, na.rm = TRUE),
    poisson = mean((as.numeric(y_true) - y_pred)^2, na.rm = TRUE),
    multiclass = mean(y_true != y_pred, na.rm = TRUE),
    fractional = mean((as.numeric(y_true) - y_pred)^2, na.rm = TRUE)
  )
}
