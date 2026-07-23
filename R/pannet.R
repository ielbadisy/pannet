#' Fit a neural panel regression model
#'
#' `pannet()` estimates a neural regression model for panel or longitudinal
#' data using a multilayer perceptron backbone augmented with additive learnable
#' individual and time effects.
#'
#' The latent predictor is
#'
#' `eta_it = f_theta(x_it) + a_i + gamma_t`
#'
#' where `a_i` and `gamma_t` are scalar (or class-specific for multiclass)
#' learnable embeddings initialised at zero, and `f_theta` is a multi-layer
#' perceptron. For dynamic models, lags of the outcome (continuous families)
#' and covariates are appended to `x_it`.
#'
#' Outcome families:
#'
#' - `gaussian`:   MSE loss; `mu_it = eta_it`
#' - `binomial`:   BCE with logits; `p_it = sigmoid(eta_it)`
#' - `poisson`:    Poisson NLL; `lambda_it = exp(eta_it)`
#' - `multiclass`: categorical CE; `p_itk = softmax(eta_itk)`
#' - `fractional`: quasi-binomial BCE; `mu_it = sigmoid(eta_it)`
#'
#' @param formula A model formula such as `y ~ x1 + x2`. May include an
#'   `offset()` term (e.g. `y ~ x1 + offset(log(exposure))`), added to the
#'   linear predictor after the MLP+embeddings, the standard `glm()`
#'   convention. Not supported with `family = "multiclass"`. `newdata`
#'   passed to `predict()` must supply the same offset variable(s) if the
#'   model was fit with one.
#' @param data A data frame containing the panel.
#' @param id Bare name of the unit identifier column.
#' @param time Bare name of the time identifier column (optional).
#' @param family Outcome family. One of `"gaussian"`, `"binomial"`,
#'   `"poisson"`, `"multiclass"`, `"fractional"`.
#' @param model Panel architecture: `"pooled"`, `"individual"`, `"time"`,
#'   `"twoway"`, or `"dynamic"`.
#' @param hidden Hidden layer sizes, e.g. `c(64, 32)`.
#' @param activation Hidden-layer activation: `"relu"`, `"tanh"`, `"gelu"`.
#' @param dropout Dropout rate(s); scalar or one per hidden layer.
#' @param batch_norm Whether to apply batch normalisation after each hidden layer.
#' @param residual Whether to add skip connections between blocks of equal width.
#' @param lr_schedule Learning rate schedule: `"none"`, `"cosine"`, or `"step"`.
#' @param epochs Number of training epochs.
#' @param batch_size Mini-batch size.
#' @param lr Learning rate.
#' @param optimizer `"adam"` or `"sgd"`.
#' @param lags Integer vector of lag orders for dynamic models.
#' @param lambda_id L2 penalty on individual effects.
#' @param lambda_time L2 penalty on time effects.
#' @param lambda_weights L2 weight decay on MLP parameters.
#' @param validation Fraction of units (or time points) held out for early stopping.
#' @param split Panel-aware split rule: `"by_id"`, `"by_time"`, or `"random"`.
#' @param standardize Whether to standardise numeric predictors.
#' @param seed Optional RNG seed.
#' @param verbose Whether to print per-epoch progress.
#' @param device `"auto"`, `"cpu"`, or `"cuda"`.
#' @param ... Unused.
#'
#' @return An object of class `"pannet"`.
#'
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   id   = rep(1:5, each = 4),
#'   time = rep(1:4, times = 5),
#'   x1   = rnorm(20),
#'   x2   = runif(20),
#'   y    = rnorm(20)
#' )
#' fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time,
#'   family = "gaussian", model = "pooled", epochs = 5, verbose = FALSE)
#' predict(fit, df)[1:3]
#'
#' @export
pannet <- function(
  formula,
  data,
  id,
  time       = NULL,
  family     = c("gaussian","binomial","poisson","multiclass","fractional"),
  model      = c("pooled","individual","time","twoway","dynamic"),
  hidden     = c(64L, 32L),
  activation = c("relu","tanh","gelu"),
  dropout    = 0,
  batch_norm = TRUE,
  residual   = FALSE,
  lr_schedule = c("none","cosine","step"),
  epochs     = 200L,
  batch_size = 32L,
  lr         = 1e-3,
  optimizer  = c("adam","sgd"),
  lags       = NULL,
  lambda_id      = 0,
  lambda_time    = 0,
  lambda_weights = 0,
  validation = 0.2,
  split      = c("by_id","by_time","random"),
  standardize = TRUE,
  seed       = NULL,
  verbose    = TRUE,
  device     = c("auto","cpu","cuda"),
  ...
) {
  family     <- panel_assert_family(family)
  model      <- panel_assert_model(model)
  split      <- panel_assert_split(split)
  activation <- match.arg(activation, c("relu","tanh","gelu"))
  optimizer  <- match.arg(optimizer,  c("adam","sgd"))
  lr_schedule <- match.arg(lr_schedule, c("none","cosine","step"))
  device     <- match.arg(device,     c("auto","cpu","cuda"))
  device     <- if (identical(device, "auto")) {
    if (isTRUE(torch::cuda_is_available())) "cuda" else "cpu"
  } else device

  mc        <- match.call()
  id_name   <- panel_resolve_name(mc$id)
  time_name <- if (is.null(mc$time)) NULL else panel_resolve_name(mc$time)

  if (!is.data.frame(data))                       stop("`data` must be a data.frame.", call. = FALSE)
  if (!id_name %in% names(data))                  stop("`id` column not found in `data`.", call. = FALSE)
  if (!is.null(time_name) && !time_name %in% names(data))
    stop("`time` column not found in `data`.", call. = FALSE)

  if (!is.null(seed)) set.seed(seed)
  if (family == "multiclass" && !is.null(attr(terms(formula), "offset"))) {
    stop("offset() is not supported with family = 'multiclass' (a scalar offset ",
         "can't broadcast across a per-class linear predictor).", call. = FALSE)
  }

  idx_result <- panel_add_index_columns(
    panel_panel_order(data, id_name, time_name), id_name, time_name
  )
  data      <- idx_result$data
  time_name <- idx_result$time_name

  spec <- panel_prepare_spec(
    formula     = formula,
    data        = data,
    id_name     = id_name,
    time_name   = time_name,
    model       = model,
    family      = family,
    lags        = lags,
    standardize = standardize
  )
  if (is.null(mc$time)) spec$time_name <- NULL

  # Remove rows with missing design or response values
  complete_rows <- stats::complete.cases(spec$x_train, spec$y_encoded)
  x_full  <- spec$x_train[complete_rows, , drop = FALSE]
  y_full  <- spec$y_encoded[complete_rows]
  data_ok <- spec$prepared_data[complete_rows, , drop = FALSE]

  design_full <- panel_build_design(data_ok, spec)
  id_idx_full   <- if (spec$use_id)   design_full$id_idx   else NULL
  time_idx_full <- if (spec$use_time) design_full$time_idx else NULL

  split_idx <- panel_split_indices(data_ok, id_name, spec$time_name %||% ".pannet_time", split, validation)
  train_idx <- split_idx$train
  valid_idx <- split_idx$valid

  if (length(train_idx) == 0L) stop("Training split is empty.", call. = FALSE)
  if (length(valid_idx) == 0L) {
    valid_idx <- train_idx
  }

  hidden_int <- as.integer(hidden)
  dropout_v  <- if (length(dropout) == 1L) rep(dropout, length(hidden_int)) else dropout

  result <- pannet_train(
    x            = x_full,
    y_encoded    = y_full,
    id_idx       = id_idx_full,
    time_idx     = time_idx_full,
    family       = family,
    output_dim   = spec$output_dim,
    hidden_units = hidden_int,
    activation   = activation,
    dropout      = dropout_v,
    batch_norm   = batch_norm,
    residual     = residual,
    n_id         = spec$n_id,
    n_time       = spec$n_time,
    epochs       = as.integer(epochs),
    batch_size   = as.integer(batch_size),
    lr           = lr,
    optimizer    = optimizer,
    lr_schedule  = lr_schedule,
    weight_decay = lambda_weights,
    lambda_id    = lambda_id,
    lambda_time  = lambda_time,
    train_idx    = train_idx,
    valid_idx    = valid_idx,
    verbose      = verbose,
    device       = device,
    seed         = seed,
    # divided by y_scale: for gaussian, y (hence eta) is trained on a
    # standardized scale, so a raw-scale offset needs the same rescaling
    # to land correctly; for other families y_scale == 1, a no-op.
    offset       = design_full$offset / spec$y_scale
  )

  # Fitted values (full training data)
  eta_fit <- pannet_forward(result$model, x_full, id_idx_full, time_idx_full, device,
                             offset = design_full$offset / spec$y_scale)
  fitted_vals <- pannet_eta_to_response(
    eta_fit, family, spec$y_center, spec$y_scale, "response", spec$levels
  )

  # Holdout loss on the validation set
  eta_val  <- eta_fit  # already computed above; slice valid rows
  y_val_raw <- spec$y_raw[complete_rows][valid_idx]
  holdout_loss <- tryCatch({
    y_pred_val <- if (is.matrix(fitted_vals)) fitted_vals[valid_idx, , drop = FALSE] else fitted_vals[valid_idx]
    switch(family,
      gaussian   = sqrt(mean((as.numeric(y_val_raw) - as.numeric(y_pred_val))^2, na.rm = TRUE)),
      binomial   = mean((panel_binary_to_numeric(y_val_raw) - as.numeric(y_pred_val))^2, na.rm = TRUE),
      poisson    = sqrt(mean((as.numeric(y_val_raw) - as.numeric(y_pred_val))^2, na.rm = TRUE)),
      multiclass = mean(as.character(y_val_raw) != as.character(y_pred_val), na.rm = TRUE),
      fractional = sqrt(mean((as.numeric(y_val_raw) - as.numeric(y_pred_val))^2, na.rm = TRUE))
    )
  }, error = function(e) NA_real_)

  structure(
    list(
      call    = mc,
      formula = formula,
      family  = family,
      model   = model,
      id      = id_name,
      time    = if (is.null(mc$time)) NULL else spec$time_name,
      terms   = terms(formula),
      x_names = spec$x_names,
      levels  = spec$levels,
      standardization = list(
        center             = spec$center,
        scale              = spec$scale,
        standardized_columns = spec$center_columns
      ),
      panel = list(
        id_name    = id_name,
        time_name  = spec$time_name,
        id_levels  = spec$id_levels,
        time_levels = spec$time_levels,
        use_id     = spec$use_id,
        use_time   = spec$use_time,
        n_id       = spec$n_id,
        n_time     = spec$n_time,
        lag_columns    = spec$lag_columns,
        lag_base_vars  = spec$lag_base_vars,
        complete_rows  = complete_rows,
        train_rows     = train_idx,
        valid_rows     = valid_idx
      ),
      network    = result$model,
      device     = device,
      parameters = list(
        hidden     = hidden_int,
        activation = activation,
        dropout    = dropout_v,
        batch_norm = batch_norm,
        residual   = residual,
        epochs     = as.integer(epochs),
        batch_size = as.integer(batch_size),
        lr         = lr,
        optimizer  = optimizer,
        lr_schedule = lr_schedule,
        lags       = lags,
        lambda_id  = lambda_id,
        lambda_time = lambda_time,
        lambda_weights = lambda_weights,
        validation = validation,
        split      = split,
        standardize = standardize,
        seed       = seed
      ),
      training = list(
        history            = result$history,
        loss               = result$history$train_loss,
        validation_loss    = result$history$valid_loss,
        epochs             = result$history$epoch,
        optimizer          = optimizer,
        lr                 = lr,
        best_epoch         = result$best_epoch,
        best_validation_loss = result$best_loss,
        holdout_loss       = holdout_loss,
        converged          = result$best_epoch < as.integer(epochs)
      ),
      data_info = list(
        n        = nrow(data_ok),
        n_id     = length(unique(data_ok[[id_name]])),
        n_time   = if (is.null(mc$time)) NA_integer_
                   else length(unique(data_ok[[spec$time_name]])),
        balanced = if (is.null(mc$time)) NA
                   else length(unique(table(data_ok[[id_name]]))) == 1L
      ),
      fitted   = fitted_vals,
      raw_data = data_ok,
      spec     = spec
    ),
    class = "pannet"
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a
