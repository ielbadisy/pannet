#' Fit a neural panel regression model
#'
#' `pannet()` estimates a neural regression model for panel or longitudinal
#' data by delegating the network fit to the local [`mlp`](https://cran.r-project.org/package=mlp)
#' package after panel-aware preprocessing.
#'
#' The latent predictor is
#'
#' `η_it = f_θ(x_it) + a_i + γ_t + h_θ(H_it)`
#'
#' where `f_θ` is a multilayer perceptron, `a_i` is an individual effect,
#' `γ_t` is a time effect, and `h_θ(H_it)` is the dynamic history component.
#'
#' Outcome families in v0.1 are:
#'
#' - `gaussian`: `y_it ~ Normal(μ_it, σ²)` with `μ_it = η_it`
#' - `binomial`: `y_it ~ Bernoulli(p_it)` with `p_it = sigmoid(η_it)`
#' - `poisson`: `y_it ~ Poisson(λ_it)` with `λ_it = exp(η_it)`
#' - `multiclass`: softmax classification
#' - `fractional`: response constrained to `[0, 1]`
#'
#' The loss is family-specific empirical risk:
#'
#' - Gaussian: `mean((y - η)^2)`
#' - Binomial: binary cross-entropy
#' - Poisson: Poisson negative log-likelihood
#'
#' @param formula A model formula such as `y ~ x1 + x2`.
#' @param data A data frame containing the panel.
#' @param id Unit identifier column.
#' @param time Optional time identifier column.
#' @param family Outcome family. See Details.
#' @param model Panel architecture: `"pooled"`, `"individual"`, `"time"`,
#'   `"twoway"`, or `"dynamic"`.
#' @param hidden Hidden layer sizes passed to [`mlp::mlp()`].
#' @param activation Activation function for the hidden layers.
#' @param epochs Number of training epochs.
#' @param batch_size Mini-batch size.
#' @param lr Learning rate.
#' @param optimizer Optimizer passed to [`mlp::mlp()`].
#' @param lags Integer vector of lags for dynamic models.
#' @param lambda_id L2 penalty for individual effects.
#' @param lambda_time L2 penalty for time effects.
#' @param lambda_weights L2 penalty for network weights.
#' @param validation Validation split fraction used inside the learner.
#' @param split Panel-aware split rule used for the held-out performance check.
#' @param standardize Whether to standardize numeric predictors before fitting.
#' @param seed Optional RNG seed.
#' @param verbose Whether to print training progress.
#' @param ... Additional arguments forwarded to [`mlp::mlp()`].
#'
#' @return An object of class `"pannet"`.
#'
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   id = rep(1:5, each = 4),
#'   time = rep(1:4, times = 5),
#'   x1 = rnorm(20),
#'   x2 = runif(20),
#'   y = rnorm(20)
#' )
#' fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time,
#'   family = "gaussian", model = "pooled", epochs = 1, verbose = FALSE)
#' predict(fit, df)[1:3]
#'
#' @export
pannet <- function(
  formula,
  data,
  id,
  time = NULL,
  family = c("gaussian", "binomial", "poisson", "multiclass", "fractional"),
  model = c("pooled", "individual", "time", "twoway", "dynamic"),
  hidden = c(32, 16),
  activation = "relu",
  epochs = 100,
  batch_size = 32,
  lr = 0.001,
  optimizer = c("adam", "sgd"),
  lags = NULL,
  lambda_id = 0,
  lambda_time = 0,
  lambda_weights = 0,
  validation = 0.2,
  split = c("by_id", "by_time", "random"),
  standardize = TRUE,
  seed = NULL,
  verbose = TRUE,
  ...
) {
  family <- panel_assert_family(family)
  model <- panel_assert_model(model)
  split <- panel_assert_split(split)
  optimizer <- match.arg(optimizer, c("adam", "sgd"))
  id_name <- panel_resolve_name(id)
  time_name <- if (is.null(time)) NULL else panel_resolve_name(time)

  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame.", call. = FALSE)
  }
  if (!id_name %in% names(data)) {
    stop("`id` column not found in `data`.", call. = FALSE)
  }
  if (!is.null(time_name) && !time_name %in% names(data)) {
    stop("`time` column not found in `data`.", call. = FALSE)
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  data <- panel_panel_order(data, id_name, time_name)
  idx <- panel_add_index_columns(data, id_name, time_name)
  data <- idx$data
  time_name <- idx$time_name

  spec <- panel_prepare_spec(
    formula = formula,
    data = data,
    id_name = id_name,
    time_name = time_name,
    model = model,
    family = family,
    lags = lags,
    standardize = standardize
  )
  if (is.null(time)) {
    spec$time_name <- NULL
  }

  data <- spec$prepared_data
  design <- spec$x_train
  y_train <- spec$y_train

  complete_rows <- stats::complete.cases(design, y_train)
  design <- design[complete_rows, , drop = FALSE]
  y_train <- y_train[complete_rows]
  data <- data[complete_rows, , drop = FALSE]

  split_idx <- panel_split_indices(data, id_name, time_name, split, validation)
  train_rows <- split_idx$train
  valid_rows <- split_idx$valid
  if (length(train_rows) == 0L) {
    stop("Training split is empty.", call. = FALSE)
  }

  x_train <- design[train_rows, , drop = FALSE]
  y_fit <- y_train[train_rows]

  task <- if (family %in% c("binomial", "multiclass")) "classification" else "regression"
  mlp_fit <- mlp::mlp(
    x = x_train,
    y = y_fit,
    task = task,
    hidden_units = hidden,
    activation = activation,
    epochs = epochs,
    batch_size = batch_size,
    lr = lr,
    optimizer = optimizer,
    validation = 0.05,
    early_stopping = FALSE,
    min_epochs = 1,
    weight_decay = lambda_weights,
    verbose = verbose,
    ...
  )

  training_pred <- panel_predict_internal(
    object = mlp_fit,
    design = design,
    family = family,
    type = "response"
  )

  train_loss <- mlp_fit$training_history
  valid_loss <- NA_real_
  if (length(valid_rows)) {
    valid_pred <- panel_predict_internal(
      object = mlp_fit,
      design = design[valid_rows, , drop = FALSE],
      family = family,
      type = "response"
    )
    valid_loss <- panel_eval_loss(
      y_true = spec$y_raw[valid_rows],
      y_pred = valid_pred,
      family = family
    )
  }

  out <- structure(
    list(
      call = match.call(),
      formula = formula,
      family = family,
      model = model,
      id = id_name,
      time = if (is.null(time)) NULL else time_name,
      terms = terms(formula),
      x_names = colnames(design),
      levels = list(id = levels(data[[id_name]]), time = if (is.null(time)) NULL else levels(data[[time_name]])),
      standardization = list(
        center = spec$center,
        scale = spec$scale,
        standardized_columns = spec$center_columns
      ),
      panel = list(
        id_name = id_name,
        time_name = if (is.null(time)) NULL else time_name,
        id_levels = spec$id_levels,
        time_levels = spec$time_levels,
        factor_levels = spec$factor_levels,
        lag_columns = spec$lag_columns,
        lag_base_vars = spec$lag_base_vars,
        design_formula = spec$design_formula,
        design_vars = spec$design_vars,
        numeric_columns = spec$numeric_columns,
        factor_columns = spec$factor_columns,
        column_means = spec$column_means,
        complete_rows = complete_rows,
        train_rows = train_rows,
        valid_rows = valid_rows
      ),
      network = mlp_fit$network,
      fit = mlp_fit,
      parameters = list(
        hidden = hidden,
        activation = activation,
        epochs = epochs,
        batch_size = batch_size,
        lr = lr,
        optimizer = optimizer,
        lags = lags,
        lambda_id = lambda_id,
        lambda_time = lambda_time,
        lambda_weights = lambda_weights,
        validation = validation,
        split = split,
        standardize = standardize,
        seed = seed
      ),
      training = list(
        loss = train_loss$train_loss,
        validation_loss = train_loss$valid_loss,
        epochs = train_loss$epoch,
        optimizer = optimizer,
        lr = lr,
        converged = isTRUE(mlp_fit$best_epoch <= epochs),
        best_epoch = mlp_fit$best_epoch,
        best_validation_loss = mlp_fit$best_validation_loss,
        holdout_loss = valid_loss
      ),
      data_info = list(
        n = nrow(data),
        n_id = length(unique(data[[id_name]])),
        n_time = if (is.null(time)) NA_integer_ else length(unique(data[[time_name]])),
        balanced = if (is.null(time)) NA else length(unique(table(data[[id_name]]))) == 1L
      ),
      fitted = training_pred,
      raw_data = data,
      spec = spec
    ),
    class = "pannet"
  )
  out
}

#' Internal prediction helper for pannet fits
#'
#' This function converts the fitted `mlp` object back to the family-specific
#' response scale. For example, Gaussian predictions satisfy `μ = η`,
#' binomial predictions satisfy `p = sigmoid(η)`, and Poisson predictions
#' satisfy `λ = exp(η)`.
#'
#' @keywords internal
panel_predict_internal <- function(object, design, family, type = c("response", "link", "prob", "class")) {
  type <- match.arg(type)
  if (family %in% c("binomial", "multiclass")) {
    if (type %in% c("response", "class")) {
      pred <- stats::predict(object, new_data = design, type = if (type == "class") "class" else "prob")
      if (family == "binomial") {
        if (is.matrix(pred)) {
          pos <- colnames(pred)[ncol(pred)]
          return(as.numeric(pred[, pos]))
        }
        return(as.numeric(pred == levels(pred)[2L]))
      }
      if (type == "class") {
        return(pred)
      }
      if (is.matrix(pred)) {
        return(factor(colnames(pred)[max.col(pred)], levels = colnames(pred)))
      }
      return(pred)
    }
    pred <- stats::predict(object, new_data = design, type = "prob")
    if (family == "binomial") {
      return(qlogis(pmin(pmax(pred[, ncol(pred)], 1e-8), 1 - 1e-8)))
    }
    return(log(pmin(pmax(pred, 1e-8), 1)))
  }

  pred <- stats::predict(object, new_data = design, type = "response")
  if (type == "link") {
    if (family == "poisson") {
      return(log1p(pmax(pred, -1 + 1e-8)))
    }
    if (family == "fractional") {
      return(qlogis(pmin(pmax(pred, 1e-8), 1 - 1e-8)))
    }
    return(pred)
  }
  if (family == "poisson") {
    return(expm1(pred))
  }
  if (family == "fractional") {
    return(plogis(pred))
  }
  pred
}
