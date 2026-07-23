pannet_build_optimizer <- function(optimizer, params, lr, weight_decay) {
  if (identical(optimizer, "adam")) {
    torch::optim_adam(params, lr = lr, weight_decay = weight_decay)
  } else {
    torch::optim_sgd(params, lr = lr, momentum = 0.9, weight_decay = weight_decay)
  }
}

pannet_build_scheduler <- function(schedule, opt, epochs) {
  if (is.null(schedule) || identical(schedule, "none")) return(NULL)
  if (identical(schedule, "cosine")) {
    return(torch::lr_cosine_annealing(opt, T_max = max(2L, as.integer(epochs))))
  }
  torch::lr_step(opt, step_size = max(5L, floor(epochs / 3L)), gamma = 0.5)
}

pannet_clone_state <- function(model) {
  state <- model$state_dict()
  lapply(state, function(p) p$clone())
}

# Core training loop.
# x         - (n, p) numeric matrix (already preprocessed)
# y_encoded - numeric or integer vector (output of pannet_encode_y)
# id_idx    - 0-based integer vector (NULL if model has no individual effects)
# time_idx  - 0-based integer vector (NULL if model has no time effects)
# train_idx / valid_idx - row indices into x/y_encoded
pannet_train <- function(
  x, y_encoded, id_idx, time_idx,
  family, output_dim,
  hidden_units, activation, dropout, batch_norm, residual,
  n_id, n_time,
  epochs, batch_size, lr, optimizer, lr_schedule,
  weight_decay, lambda_id, lambda_time,
  train_idx, valid_idx,
  verbose, device, seed,
  patience_frac = 0.15
) {
  if (!is.null(seed)) torch::torch_manual_seed(as.integer(seed))

  hidden_units <- as.integer(hidden_units)
  dropout <- if (length(dropout) == 1L) rep(as.numeric(dropout), length(hidden_units)) else as.numeric(dropout)

  criterion <- pannet_criterion(family)

  model <- pannet_module(
    input_dim   = ncol(x),
    hidden_units = hidden_units,
    output_dim  = output_dim,
    activation  = activation,
    dropout     = dropout,
    batch_norm  = batch_norm,
    residual    = residual,
    n_id        = n_id,
    n_time      = n_time
  )
  model$to(device = device)

  opt       <- pannet_build_optimizer(optimizer, model$parameters, lr, weight_decay)
  scheduler <- pannet_build_scheduler(lr_schedule, opt, epochs)

  # multiclass targets are 1-based long ints for R torch nnf_cross_entropy
  is_long_y <- identical(family, "multiclass")
  make_y_t  <- function(idx) {
    if (is_long_y) {
      torch::torch_tensor(y_encoded[idx], dtype = torch::torch_long(), device = device)
    } else {
      torch::torch_tensor(as.numeric(y_encoded[idx]), dtype = torch::torch_float(), device = device)
    }
  }
  make_x_t  <- function(idx) torch::torch_tensor(x[idx, , drop = FALSE], dtype = torch::torch_float(), device = device)
  make_id_t <- function(idx) if (!is.null(id_idx))   torch::torch_tensor(id_idx[idx],   dtype = torch::torch_long(), device = device) else NULL
  make_tm_t <- function(idx) if (!is.null(time_idx)) torch::torch_tensor(time_idx[idx], dtype = torch::torch_long(), device = device) else NULL

  x_vl   <- make_x_t(valid_idx)
  y_vl   <- make_y_t(valid_idx)
  id_vl  <- make_id_t(valid_idx)
  tm_vl  <- make_tm_t(valid_idx)
  y_enc_vl <- y_encoded[valid_idx]

  n_train    <- length(train_idx)
  patience   <- max(5L, floor(epochs * patience_frac))
  min_epochs <- max(5L, floor(epochs * 0.05))

  history    <- vector("list", epochs)
  best_loss  <- Inf
  best_metric <- -Inf
  best_epoch <- 1L
  best_state <- NULL
  wait       <- 0L

  for (epoch in seq_len(epochs)) {
    model$train()
    order_   <- sample.int(n_train)
    batches  <- split(order_, ceiling(seq_along(order_) / batch_size))
    tr_losses <- numeric(length(batches))

    for (bi in seq_along(batches)) {
      rows  <- train_idx[batches[[bi]]]
      bx    <- make_x_t(rows)
      by    <- make_y_t(rows)
      bid   <- make_id_t(rows)
      btm   <- make_tm_t(rows)

      opt$zero_grad()
      eta  <- model(bx, bid, btm)
      loss <- criterion(eta, by) + pannet_panel_penalty(model, lambda_id, lambda_time)
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
      opt$step()
      tr_losses[[bi]] <- as.numeric(loss$item())
    }

    model$eval()
    torch::with_no_grad({
      eta_vl   <- model(x_vl, id_vl, tm_vl)
      v_loss   <- as.numeric(criterion(eta_vl, y_vl)$item())
      eta_arr  <- if (family == "multiclass") as.array(eta_vl$to(device = "cpu")) else as.numeric(eta_vl$to(device = "cpu"))
      v_metric <- pannet_valid_metric(family, y_enc_vl, eta_arr)
      tr_loss  <- mean(tr_losses)
      cur_lr   <- opt$param_groups[[1]]$lr

      history[[epoch]] <- data.frame(
        epoch        = epoch,
        train_loss   = tr_loss,
        valid_loss   = v_loss,
        valid_metric = v_metric,
        learning_rate = cur_lr
      )

      if (v_loss < best_loss) {
        best_loss   <- v_loss
        best_metric <- v_metric
        best_epoch  <- epoch
        wait        <- 0L
        best_state  <- pannet_clone_state(model)
      } else if (epoch >= min_epochs) {
        wait <- wait + 1L
      }

      if (isTRUE(verbose)) {
        message(sprintf(
          "Epoch %d/%d  train: %.4f  val: %.4f  metric: %.4f  lr: %.5f",
          epoch, epochs, tr_loss, v_loss, v_metric, cur_lr
        ))
      }
    })

    if (!is.null(scheduler)) scheduler$step()
    if (epoch >= min_epochs && wait >= patience) break
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  history <- do.call(rbind, Filter(Negate(is.null), history))

  list(
    model       = model,
    history     = history,
    best_epoch  = best_epoch,
    best_loss   = best_loss,
    best_metric = best_metric
  )
}

# Run model forward on a design matrix and return raw eta array (CPU).
pannet_forward <- function(model, x, id_idx, time_idx, device) {
  model$eval()
  x_t  <- torch::torch_tensor(x, dtype = torch::torch_float(), device = device)
  id_t <- if (!is.null(id_idx))   torch::torch_tensor(as.integer(id_idx),   dtype = torch::torch_long(), device = device) else NULL
  tm_t <- if (!is.null(time_idx)) torch::torch_tensor(as.integer(time_idx), dtype = torch::torch_long(), device = device) else NULL
  eta  <- torch::with_no_grad(model(x_t, id_t, tm_t))
  as.array(eta$to(device = "cpu"))
}
