#' Resolve a column-name argument passed as a bare symbol or a string
#'
#' Takes the *already-quoted* expression for `id`/`time` (typically
#' `match.call()$id`, captured in the caller's own frame) rather than
#' `substitute()`-ing its own parameter -- `substitute()` only unwraps one
#' level of promise, so calling it here on a forwarded argument would just
#' return this function's own parameter name, not the original expression
#' the user wrote at the `pannet()` call site.
#'
#' @keywords internal
#' @noRd
panel_resolve_name <- function(expr) {
  if (is.symbol(expr)) return(deparse(expr))
  if (is.character(expr)) return(expr[[1L]])
  stop("`id` and `time` must be bare column names or single strings.", call. = FALSE)
}

panel_assert_family <- function(f) match.arg(f, c("gaussian","binomial","poisson","multiclass","fractional"))
panel_assert_model  <- function(m) match.arg(m, c("pooled","individual","time","twoway","dynamic"))
panel_assert_split  <- function(s) match.arg(s, c("by_id","by_time","random"))

panel_binary_to_numeric <- function(y) {
  if (is.factor(y)) as.integer(y == levels(y)[nlevels(y)]) else as.numeric(y)
}

panel_panel_order <- function(data, id_name, time_name = NULL) {
  ord <- if (is.null(time_name)) {
    order(data[[id_name]], seq_len(nrow(data)))
  } else {
    order(data[[id_name]], data[[time_name]], seq_len(nrow(data)))
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
  list(data = data, time_name = time_name)
}

panel_make_lags <- function(data, id_name, time_name, vars, lags) {
  if (!length(lags) || !length(vars)) return(list(data = data, lag_cols = character()))
  lags     <- sort(unique(as.integer(lags)))
  lag_cols <- character()
  groups   <- split(seq_len(nrow(data)), data[[id_name]])
  for (var in vars) {
    if (!is.numeric(data[[var]])) next
    for (lag in lags) {
      out <- rep(NA_real_, nrow(data))
      for (idx in groups) {
        n_i <- length(idx)
        if (n_i > lag) out[idx[(lag + 1L):n_i]] <- data[[var]][idx[seq_len(n_i - lag)]]
      }
      nm <- paste0("lag", lag, "_", var)
      data[[nm]] <- out
      lag_cols   <- c(lag_cols, nm)
    }
  }
  list(data = data, lag_cols = lag_cols)
}

panel_rebuild_lags <- function(data, id_name, time_name, lag_columns) {
  if (!length(lag_columns)) return(data)
  m     <- regexec("^lag([0-9]+)_(.+)$", lag_columns)
  parts <- regmatches(lag_columns, m)
  parts <- parts[lengths(parts) > 0L]
  if (!length(parts)) return(data)
  vars <- unique(vapply(parts, `[[`, character(1L), 3L))
  lags <- unique(as.integer(vapply(parts, `[[`, character(1L), 2L)))
  data <- panel_panel_order(data, id_name, time_name)
  panel_make_lags(data, id_name, time_name, vars = vars, lags = lags)$data
}

panel_split_indices <- function(data, id_name, time_name, split, validation) {
  n <- nrow(data)
  if (is.null(validation) || validation <= 0) return(list(train = seq_len(n), valid = integer()))
  validation <- max(0, min(0.9, validation))
  if (split == "random") {
    valid <- sort(sample.int(n, size = max(1L, floor(n * validation))))
    return(list(train = setdiff(seq_len(n), valid), valid = valid))
  }
  if (split == "by_id") {
    ids       <- unique(data[[id_name]])
    valid_ids <- sample(ids, size = max(1L, floor(length(ids) * validation)))
    valid     <- which(data[[id_name]] %in% valid_ids)
    return(list(train = setdiff(seq_len(n), valid), valid = valid))
  }
  times       <- unique(data[[time_name]])
  valid_times <- sample(times, size = max(1L, floor(length(times) * validation)))
  valid       <- which(data[[time_name]] %in% valid_times)
  list(train = setdiff(seq_len(n), valid), valid = valid)
}

# -------------------------------------------------------------------
# Preprocessing spec
# -------------------------------------------------------------------

panel_prepare_spec <- function(formula, data, id_name, time_name, model,
                                family, lags, standardize) {
  # Panel index encoding
  id_levels <- sort(unique(as.character(data[[id_name]])))
  data[[id_name]] <- factor(as.character(data[[id_name]]), levels = id_levels)
  time_levels <- character()
  if (!is.null(time_name)) {
    time_levels <- sort(unique(as.character(data[[time_name]])))
    data[[time_name]] <- factor(as.character(data[[time_name]]), levels = time_levels)
  }

  use_id   <- model %in% c("individual", "twoway")
  use_time <- model %in% c("time", "twoway") && !is.null(time_name)
  if (model %in% c("time", "twoway") && is.null(time_name)) {
    stop("`time` is required for model = 'time' or 'twoway'.", call. = FALSE)
  }

  # Dynamic lags: include lagged y for continuous families
  lag_cols      <- character()
  lag_base_vars <- character()
  if (model == "dynamic" && length(lags)) {
    pred_vars     <- setdiff(all.vars(delete.response(terms(formula))), c(id_name, time_name))
    y_name        <- all.vars(formula[[2L]])
    lag_base_vars <- if (family %in% c("gaussian", "poisson", "fractional")) {
      unique(c(pred_vars, y_name))
    } else {
      pred_vars
    }
    lagged   <- panel_make_lags(data, id_name, time_name, lag_base_vars, lags)
    data     <- lagged$data
    lag_cols <- lagged$lag_cols
  }

  # Design formula: only covariates, NO panel effect dummy columns
  design_formula <- if (length(lag_cols)) {
    update(formula, paste(". ~ . +", paste(lag_cols, collapse = " + ")))
  } else {
    formula
  }

  mf  <- model.frame(design_formula, data = data, na.action = na.pass)
  y   <- model.response(mf)
  mm  <- model.matrix(delete.response(terms(design_formula)), data = mf)
  if ("(Intercept)" %in% colnames(mm)) mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]

  # offset(...) terms (e.g. offset(log(exposure))) are recognised by base R's
  # formula/terms machinery automatically -- excluded from the design matrix
  # above, retrievable here via model.offset(). Added to eta post-hoc (not
  # fed through the MLP), the standard glm() convention.
  offset_vec <- stats::model.offset(mf)
  if (is.null(offset_vec)) offset_vec <- rep(0, nrow(mf))
  offset_vec <- as.numeric(offset_vec)

  # Standardise numeric predictor columns
  mm_center <- stats::setNames(rep(0, ncol(mm)), colnames(mm))
  mm_scale  <- stats::setNames(rep(1, ncol(mm)), colnames(mm))
  std_cols  <- character()
  if (standardize) {
    for (nm in colnames(mm)) {
      col <- mm[, nm]
      if (is.numeric(col) && !all(col %in% c(0, 1), na.rm = TRUE)) {
        mm_center[nm] <- mean(col, na.rm = TRUE)
        mm_scale[nm]  <- stats::sd(col, na.rm = TRUE)
        if (!is.finite(mm_scale[nm]) || mm_scale[nm] == 0) mm_scale[nm] <- 1
        mm[, nm] <- (col - mm_center[nm]) / mm_scale[nm]
        std_cols <- c(std_cols, nm)
      }
    }
  }

  # Track factor/numeric predictor columns for new-data prediction
  factor_cols  <- names(Filter(is.factor, mf))
  factor_levs  <- stats::setNames(
    lapply(factor_cols, function(nm) if (nm %in% names(mf)) levels(mf[[nm]]) else character()),
    factor_cols
  )
  numeric_cols <- setdiff(names(Filter(is.numeric, mf)), c(id_name, time_name))

  # Multiclass levels and output dimension
  mc_levels  <- if (family == "multiclass") sort(unique(as.character(y))) else NULL
  output_dim <- if (family == "multiclass") length(mc_levels) else 1L

  # Encode y for training
  yenc <- pannet_encode_y(y, family, mc_levels)

  spec <- list(
    formula         = formula,
    design_formula  = design_formula,
    design_vars     = all.vars(delete.response(terms(design_formula))),
    response_name   = all.vars(formula[[2L]]),
    family          = family,
    model           = model,
    id_name         = id_name,
    time_name       = time_name,
    id_levels       = id_levels,
    time_levels     = time_levels,
    use_id          = use_id,
    use_time        = use_time,
    n_id            = if (use_id)   length(id_levels)   else 0L,
    n_time          = if (use_time) length(time_levels) else 0L,
    output_dim      = output_dim,
    levels          = mc_levels,
    factor_levels   = factor_levs,
    factor_columns  = factor_cols,
    numeric_columns = numeric_cols,
    lag_columns     = lag_cols,
    lag_base_vars   = lag_base_vars,
    center_columns  = std_cols,
    center          = as.list(mm_center),
    scale           = as.list(mm_scale),
    column_means    = stats::setNames(colMeans(mm, na.rm = TRUE), colnames(mm)),
    x_names         = colnames(mm),
    x_train         = mm,
    y_raw           = y,
    y_encoded       = yenc$y,
    y_center        = yenc$center,
    y_scale         = yenc$scale,
    offset          = offset_vec,
    has_offset      = !is.null(stats::model.offset(mf)),
    prepared_data   = data
  )
  class(spec) <- "pannet_spec"
  spec
}

# -------------------------------------------------------------------
# Design matrix + panel index extraction for prediction
# -------------------------------------------------------------------

# Returns list(x = matrix, id_idx = integer | NULL, time_idx = integer | NULL)
# id_idx and time_idx are 0-based for torch nn_embedding.
panel_build_design <- function(data, spec, new_id_strategy = c("zero", "mean", "error")) {
  new_id_strategy <- match.arg(new_id_strategy)

  # Always rebuild lag columns from the raw variables when the model has
  # any. Checking only whether the *column names* are already present (the
  # previous condition) is not sufficient: predict.pannet()'s newdata path
  # combines new rows with training data via rbind(), which means the new
  # rows inherit the lag column *names* from training (already-fitted
  # values) while their own lag *values* are NA placeholders -- so the old
  # "already present" check skipped rebuilding and silently fed those NAs
  # (zero-filled downstream) into the network instead of the true lagged
  # history. Rebuilding is idempotent and cheap, so just always do it.
  if (length(spec$lag_columns)) {
    data <- panel_rebuild_lags(data, spec$id_name, spec$time_name, spec$lag_columns)
  }

  # Apply stored factor levels to formula-level factors
  for (nm in names(spec$factor_levels)) {
    if (nm %in% names(data)) {
      data[[nm]] <- factor(as.character(data[[nm]]), levels = spec$factor_levels[[nm]])
    }
  }
  for (nm in spec$numeric_columns) {
    if (nm %in% names(data)) data[[nm]] <- as.numeric(data[[nm]])
  }
  for (nm in spec$lag_columns) {
    if (nm %in% names(data)) data[[nm]] <- as.numeric(data[[nm]])
  }

  mm_terms <- delete.response(terms(spec$design_formula))
  # model.matrix()'s default na.action (na.omit) silently DROPS any row with
  # an NA in any term -- including the lag columns, which are NA for every
  # individual's first `lag` training periods. That corrupts the row count
  # (and hence the boolean-marker row alignment predict.pannet relies on)
  # for any dynamic-model design. Build the model.frame with na.action =
  # na.pass explicitly and pass *that* (already class "model.frame") to
  # model.matrix(), which then uses it as-is rather than re-deriving it
  # with the default na.action.
  mf_pred <- model.frame(mm_terms, data = data, na.action = na.pass)
  mm <- model.matrix(mm_terms, data = mf_pred)
  if ("(Intercept)" %in% colnames(mm)) mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  if (nrow(mm) != nrow(data)) {
    # model.matrix() can still drop rows for reasons na.pass doesn't cover
    # (e.g. a factor level combination it can't represent); re-expand to
    # the full row count with NA so downstream alignment stays correct
    # and the NA-handling below has a chance to act on them.
    full <- matrix(NA_real_, nrow(data), ncol(mm), dimnames = list(NULL, colnames(mm)))
    full[match(rownames(mm), rownames(mf_pred)), ] <- mm
    mm <- full
  }

  # offset(...), if the model was fit with one: required in new data too
  # (matching glm()'s convention), since there's no principled default.
  offset_vec <- rep(0, nrow(data))
  if (isTRUE(spec$has_offset)) {
    off <- stats::model.offset(mf_pred)
    if (is.null(off)) {
      stop("This model was fit with an offset() term; `data`/`newdata` must ",
           "supply the same offset variable(s).", call. = FALSE)
    }
    offset_vec <- as.numeric(off)
  }

  # Apply standardisation
  for (nm in spec$center_columns) {
    if (nm %in% colnames(mm)) {
      mm[, nm] <- (mm[, nm] - spec$center[[nm]]) / spec$scale[[nm]]
    }
  }

  # Handle NAs
  if (anyNA(mm)) {
    if (new_id_strategy == "mean") {
      for (nm in colnames(mm)) {
        bad <- is.na(mm[, nm])
        if (any(bad)) mm[bad, nm] <- spec$column_means[[nm]]
      }
    } else {
      mm[is.na(mm)] <- 0
    }
  }

  # Panel index extraction (1-based for R torch nn_embedding)
  # Row n_id+1 (time: n_time+1) is a sentinel kept at zero (or mean) for unseen units
  id_idx   <- NULL
  time_idx <- NULL

  if (!is.null(spec$id_name) && spec$id_name %in% names(data)) {
    id_raw    <- as.character(data[[spec$id_name]])
    id_mapped <- match(id_raw, spec$id_levels)   # 1-based; NA for unseen
    unseen    <- is.na(id_mapped)
    if (any(unseen)) {
      if (new_id_strategy == "error") {
        stop("New data contains IDs not seen during training and `new_id_strategy = 'error'`.", call. = FALSE)
      }
      # Sentinel index = n_id + 1 (zero-initialised; updated to mean by predict.pannet if requested)
      id_mapped[unseen] <- length(spec$id_levels) + 1L
    }
    id_idx <- as.integer(id_mapped)
  }

  if (!is.null(spec$time_name) && spec$time_name %in% names(data)) {
    time_raw    <- as.character(data[[spec$time_name]])
    time_mapped <- match(time_raw, spec$time_levels)  # 1-based
    time_mapped[is.na(time_mapped)] <- length(spec$time_levels) + 1L  # sentinel
    time_idx <- as.integer(time_mapped)
  }

  list(x = mm, id_idx = id_idx, time_idx = time_idx, offset = offset_vec)
}

.pannet_sentinel_idx <- function(n) as.integer(n) + 1L

panel_mean_id_effect <- function(model, n_id) {
  if (is.null(model$id_emb) || n_id == 0L) return(0)
  w <- as.array(model$id_emb$weight$to(device = "cpu")$detach())
  colMeans(w[seq_len(n_id), , drop = FALSE])
}

panel_mean_time_effect <- function(model, n_time) {
  if (is.null(model$time_emb) || n_time == 0L) return(0)
  w <- as.array(model$time_emb$weight$to(device = "cpu")$detach())
  colMeans(w[seq_len(n_time), , drop = FALSE])
}
