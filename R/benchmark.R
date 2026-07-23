#' Simulate panel data for benchmarking
#'
#' Generates a balanced panel with known individual and time effects plus
#' a covariate effect that can be linear or nonlinear.
#'
#' @param N Number of individuals.
#' @param T Number of time periods per individual.
#' @param n_x Number of continuous covariates.
#' @param dgp One of `"linear"`, `"nonlinear"`, `"nonlinear_strong"`.
#' @param family Outcome family (same options as `pannet()`).
#' @param sigma_id SD of individual effects.
#' @param sigma_time SD of time effects.
#' @param sigma_noise SD of residual noise (gaussian) or scale for other families.
#' @param seed Random seed.
#'
#' @return A list with elements `data`, `true_id_effects`, `true_time_effects`,
#'   `true_beta`.
#' @export
pannet_simulate <- function(
  N           = 50L,
  T           = 10L,
  n_x         = 3L,
  dgp         = c("linear", "nonlinear", "nonlinear_strong"),
  family      = "gaussian",
  sigma_id    = 1,
  sigma_time  = 0.5,
  sigma_noise = 1,
  seed        = 42L
) {
  dgp <- match.arg(dgp)
  set.seed(seed)
  n   <- N * T
  id  <- rep(seq_len(N), each = T)
  tim <- rep(seq_len(T), times = N)

  X <- matrix(stats::rnorm(n * n_x), nrow = n, ncol = n_x)
  colnames(X) <- paste0("x", seq_len(n_x))

  a_i   <- stats::rnorm(N, sd = sigma_id)
  g_t   <- stats::rnorm(T, sd = sigma_time)
  panel_effect <- a_i[id] + g_t[tim]

  if (dgp == "linear") {
    beta <- stats::rnorm(n_x)
    eta  <- X %*% beta + panel_effect
  } else if (dgp == "nonlinear") {
    beta <- rep(1, n_x)
    eta  <- sin(X[, 1L]) + 0.5 * X[, 2L]^2 + X[, 3L] + panel_effect
  } else {
    beta <- rep(1, n_x)
    eta  <- sin(2 * pi * X[, 1L]) * cos(X[, 2L]) +
            tanh(2 * X[, 2L]) + exp(-X[, 3L]^2) + panel_effect
  }

  y <- switch(family,
    gaussian   = eta + stats::rnorm(n, sd = sigma_noise),
    binomial   = stats::rbinom(n, 1L, stats::plogis(eta)),
    poisson    = stats::rpois(n, exp(pmin(eta, 5))),
    fractional = stats::rbeta(n, pmax(stats::plogis(eta) * 5, 0.1),
                                 pmax((1 - stats::plogis(eta)) * 5, 0.1)),
    multiclass = {
      probs <- exp(cbind(eta, eta + 0.5, -eta))
      probs <- probs / rowSums(probs)
      factor(apply(probs, 1, function(p) sample(c("A","B","C"), 1, prob = p)))
    }
  )

  df <- data.frame(id = id, time = tim, as.data.frame(X))
  df$y <- y
  list(data = df, true_id_effects = a_i, true_time_effects = g_t,
       true_beta = if (dgp == "linear") stats::setNames(beta, paste0("x", seq_len(n_x))) else NULL)
}

#' Benchmark pannet against classical panel estimators
#'
#' Fits multiple pannet configurations and, if available, classical estimators
#' (`plm` for Gaussian, `pglm` for binary/count), then reports test-set
#' performance metrics.
#'
#' @param formula Model formula.
#' @param data Training data frame.
#' @param id Bare name of the unit identifier column.
#' @param time Bare name of the time identifier column.
#' @param family Outcome family.
#' @param test_data Test data frame (must include the response). If `NULL`,
#'   `test_fraction` of individuals are held out from `data`.
#' @param test_fraction Fraction of individuals to hold out when `test_data`
#'   is `NULL`.
#' @param pannet_configs A named list of extra argument lists forwarded to
#'   `pannet()`. Defaults to a grid of pooled / individual / twoway models.
#' @param epochs Number of training epochs for each pannet configuration.
#' @param seed Random seed.
#' @param verbose Whether to print pannet training progress.
#'
#' @return A data frame with one row per estimator and columns for test metric
#'   and training time.
#' @export
pannet_benchmark <- function(
  formula,
  data,
  id,
  time,
  family         = "gaussian",
  test_data      = NULL,
  test_fraction  = 0.2,
  pannet_configs = NULL,
  epochs         = 300L,
  seed           = 42L,
  verbose        = FALSE
) {
  mc        <- match.call()
  id_name   <- panel_resolve_name(mc$id)
  time_name <- panel_resolve_name(mc$time)

  # Train / test split by individual
  if (is.null(test_data)) {
    set.seed(seed)
    all_ids   <- unique(data[[id_name]])
    n_test    <- max(1L, floor(length(all_ids) * test_fraction))
    test_ids  <- sample(all_ids, n_test)
    test_data <- data[data[[id_name]] %in% test_ids,  , drop = FALSE]
    data      <- data[!data[[id_name]] %in% test_ids, , drop = FALSE]
  }

  response_name <- all.vars(formula[[2L]])
  y_test_raw    <- test_data[[response_name]]

  metric_fn <- function(y_true, y_pred, fam) {
    switch(fam,
      gaussian   = sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE)),
      binomial   = mean((panel_binary_to_numeric(y_true) == as.numeric(y_pred >= 0.5)), na.rm = TRUE),
      poisson    = sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE)),
      multiclass = mean(as.character(y_true) == as.character(y_pred), na.rm = TRUE),
      fractional = sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE))
    )
  }
  metric_label <- switch(family,
    gaussian = "RMSE", binomial = "Accuracy", poisson = "RMSE",
    multiclass = "Accuracy", fractional = "RMSE"
  )
  higher_is_better <- family %in% c("binomial", "multiclass")

  results <- list()

  # --- Default pannet configurations ---
  if (is.null(pannet_configs)) {
    pannet_configs <- list(
      "pannet(pooled)"      = list(model = "pooled"),
      "pannet(individual)"  = list(model = "individual"),
      "pannet(twoway)"      = list(model = "twoway"),
      "pannet(deep+twoway)" = list(model = "twoway", hidden = c(128L, 64L, 32L),
                                    dropout = 0.1, batch_norm = TRUE,
                                    lr_schedule = "cosine")
    )
  }

  for (cfg_name in names(pannet_configs)) {
    cfg <- pannet_configs[[cfg_name]]
    cfg_args <- c(
      list(formula = formula, data = data, id = data[[id_name]],
           time = data[[time_name]], family = family,
           epochs = as.integer(epochs), verbose = verbose, seed = seed,
           validation = 0.15, split = "by_id"),
      cfg
    )
    # id and time need to be the actual column values for the call; use strings
    cfg_args$id   <- as.name(id_name)
    cfg_args$time <- as.name(time_name)

    t0  <- proc.time()[["elapsed"]]
    fit <- tryCatch(
      do.call(pannet, cfg_args),
      error = function(e) { message("pannet config '", cfg_name, "' failed: ", e$message); NULL }
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    if (is.null(fit)) next

    y_pred <- tryCatch(predict(fit, test_data, type = "response"), error = function(e) NULL)
    if (is.null(y_pred)) next
    m <- metric_fn(y_test_raw, y_pred, family)
    results[[cfg_name]] <- data.frame(
      estimator = cfg_name, metric = m, metric_name = metric_label,
      time_sec = elapsed, stringsAsFactors = FALSE
    )
  }

  # --- Classical: plm (Gaussian) ---
  if (family == "gaussian" && requireNamespace("plm", quietly = TRUE)) {
    plm_data <- plm::pdata.frame(rbind(data, test_data),
                                  index = c(id_name, time_name))
    train_plm <- plm_data[plm_data[[id_name]] %in% data[[id_name]], ]
    test_plm  <- plm_data[plm_data[[id_name]] %in% test_data[[id_name]], ]

    fit_ols <- tryCatch(
      stats::lm(formula, data = data),
      error = function(e) NULL
    )
    if (!is.null(fit_ols)) {
      t0 <- proc.time()[["elapsed"]]
      fit_ols2 <- stats::lm(formula, data = data)
      elapsed  <- proc.time()[["elapsed"]] - t0
      y_pred   <- predict(fit_ols2, newdata = test_data)
      m        <- metric_fn(y_test_raw, y_pred, family)
      results[["Pooled OLS"]] <- data.frame(
        estimator = "Pooled OLS", metric = m, metric_name = metric_label,
        time_sec = elapsed, stringsAsFactors = FALSE
      )
    }

    for (effect in c("individual", "twoway")) {
      nm <- paste0("FE (", effect, ")")
      t0  <- proc.time()[["elapsed"]]
      fit_fe <- tryCatch(
        plm::plm(formula, data = train_plm, model = "within", effect = effect),
        error = function(e) NULL
      )
      elapsed <- proc.time()[["elapsed"]] - t0
      if (is.null(fit_fe)) next
      # FE within estimators cannot predict for new individuals out-of-sample.
      # Use grand-mean fallback: predict from OLS without unit dummies (as lower bound).
      y_pred <- tryCatch({
        p <- predict(fit_fe, newdata = test_plm)
        if (anyNA(p)) {
          # Fall back to demeaned-coefficient prediction from the beta only
          beta <- coef(fit_fe)
          rhs_vars <- names(beta)
          X_test  <- as.matrix(test_data[, rhs_vars, drop = FALSE])
          p <- as.numeric(X_test %*% beta) + mean(plm::fixef(fit_fe))
        }
        p
      }, error = function(e) NULL)
      if (is.null(y_pred)) next
      m <- metric_fn(y_test_raw, y_pred, family)
      results[[nm]] <- data.frame(
        estimator = nm, metric = m, metric_name = metric_label,
        time_sec = elapsed, stringsAsFactors = FALSE
      )
    }

    # Random effects
    t0     <- proc.time()[["elapsed"]]
    fit_re <- tryCatch(
      plm::plm(formula, data = train_plm, model = "random"),
      error = function(e) NULL
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    if (!is.null(fit_re)) {
      y_pred <- tryCatch(predict(fit_re, newdata = test_plm), error = function(e) NULL)
      if (!is.null(y_pred)) {
        m <- metric_fn(y_test_raw, y_pred, family)
        results[["RE (GLS)"]] <- data.frame(
          estimator = "RE (GLS)", metric = m, metric_name = metric_label,
          time_sec = elapsed, stringsAsFactors = FALSE
        )
      }
    }
  }

  # --- Classical: pglm (binary / count) ---
  if (family %in% c("binomial", "poisson") && requireNamespace("pglm", quietly = TRUE)) {
    pglm_fam  <- if (family == "binomial") "binomial" else "poisson"
    pglm_link <- if (family == "binomial") "logit"    else "log"

    t0     <- proc.time()[["elapsed"]]
    fit_pg <- tryCatch(
      pglm::pglm(formula, data = data,
                 index = c(id_name, time_name),
                 family = stats::family(pglm_fam),
                 model  = "within",
                 effect = "individual"),
      error = function(e) NULL
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    if (!is.null(fit_pg)) {
      eta_test <- tryCatch(predict(fit_pg, newdata = test_data), error = function(e) NULL)
      if (!is.null(eta_test)) {
        y_pred <- if (family == "binomial") stats::plogis(eta_test) else exp(eta_test)
        m <- metric_fn(y_test_raw, y_pred, family)
        results[["pglm (FE)"]] <- data.frame(
          estimator = "pglm (FE)", metric = m, metric_name = metric_label,
          time_sec = elapsed, stringsAsFactors = FALSE
        )
      }
    }
  }

  # --- Classical: panglm (fast FE/RE for gaussian/poisson) ---
  if (family %in% c("gaussian", "poisson") && requireNamespace("panglm", quietly = TRUE)) {
    for (pg_model in c("within", "random")) {
      nm <- paste0("panglm (", if (pg_model == "within") "FE" else "RE", ")")
      t0 <- proc.time()[["elapsed"]]
      fit_pl <- tryCatch(
        panglm::panglm(formula, data = data, index = c(id_name, time_name),
                        model = pg_model, family = family),
        error = function(e) NULL
      )
      elapsed <- proc.time()[["elapsed"]] - t0
      if (is.null(fit_pl)) next
      # panglm's predict() is coefficient-only (no unit intercept for
      # genuinely new individuals, the same limitation plm's FE predict
      # has above); fine for this relative-performance comparison.
      y_pred <- tryCatch(
        stats::predict(fit_pl, newdata = test_data, type = "response"),
        error = function(e) NULL
      )
      if (is.null(y_pred)) next
      m <- metric_fn(y_test_raw, y_pred, family)
      results[[nm]] <- data.frame(
        estimator = nm, metric = m, metric_name = metric_label,
        time_sec = elapsed, stringsAsFactors = FALSE
      )
    }
  }

  # --- Assemble and print ---
  out <- do.call(rbind, results)
  rownames(out) <- NULL
  out <- out[order(out$metric * if (higher_is_better) -1 else 1), ]

  cat("\n=== pannet_benchmark ===\n")
  cat("Family:", family, "  Metric:", metric_label,
      if (higher_is_better) "(higher better)" else "(lower better)", "\n\n")
  print(out[, c("estimator", "metric", "time_sec")], row.names = FALSE, digits = 4)
  cat("\n")

  invisible(out)
}
