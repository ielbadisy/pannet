# Encode y into the format required by each family's loss.
# Gaussian y is standardised; all others are kept on their natural scale.
pannet_encode_y <- function(y, family, mc_levels = NULL) {
  switch(family,
    gaussian = {
      y_n  <- as.numeric(y)
      ctr  <- mean(y_n, na.rm = TRUE)
      scl  <- max(stats::sd(y_n, na.rm = TRUE), 1e-8)
      list(y = (y_n - ctr) / scl, center = ctr, scale = scl)
    },
    binomial = {
      list(y = as.numeric(panel_binary_to_numeric(y)), center = 0, scale = 1)
    },
    poisson = {
      list(y = as.numeric(y), center = 0, scale = 1)
    },
    multiclass = {
      y_fac <- if (is.factor(y)) y else factor(y)
      if (!is.null(mc_levels)) {
        y_fac <- factor(as.character(y_fac), levels = mc_levels)
      }
      # R torch nnf_cross_entropy expects 1-based class indices
      list(y = as.integer(y_fac), center = 0, scale = 1)
    },
    fractional = {
      y_n <- as.numeric(y)
      if (any(y_n < 0 | y_n > 1, na.rm = TRUE)) {
        stop("fractional response must be in [0, 1].", call. = FALSE)
      }
      list(y = y_n, center = 0, scale = 1)
    }
  )
}

# Family-specific loss function operating directly on torch tensors.
# eta has shape (n, output_dim); for non-multiclass it is squeezed to (n,).
pannet_criterion <- function(family) {
  switch(family,
    gaussian = function(eta, y_t) {
      torch::nnf_mse_loss(eta$squeeze(2L), y_t)
    },
    binomial = function(eta, y_t) {
      torch::nnf_binary_cross_entropy_with_logits(eta$squeeze(2L), y_t)
    },
    poisson = function(eta, y_t) {
      e <- eta$squeeze(2L)
      # Poisson NLL: E[exp(eta) - y * eta].  Clamp eta to avoid overflow.
      e_c <- torch::torch_clamp(e, max = 20)
      torch::torch_mean(torch::torch_exp(e_c) - y_t * e)
    },
    multiclass = function(eta, y_t) {
      torch::nnf_cross_entropy(eta, y_t)
    },
    fractional = function(eta, y_t) {
      # Quasi-binomial / fractional logit loss
      torch::nnf_binary_cross_entropy_with_logits(eta$squeeze(2L), y_t)
    }
  )
}

# L2 regularisation on panel effects.
pannet_panel_penalty <- function(model, lambda_id, lambda_time) {
  pen <- torch::torch_tensor(0.0, requires_grad = FALSE)
  if (lambda_id > 0 && model$n_id > 0L) {
    pen <- pen + lambda_id * model$id_emb$weight$pow(2)$mean()
  }
  if (lambda_time > 0 && model$n_time > 0L) {
    pen <- pen + lambda_time * model$time_emb$weight$pow(2)$mean()
  }
  pen
}

# Validation metric computed from the raw eta array (before any inverse link).
# Higher values are always better (accuracy or negative RMSE).
pannet_valid_metric <- function(family, y_enc_valid, eta_arr) {
  switch(family,
    gaussian = {
      pred <- as.numeric(eta_arr)
      -sqrt(mean((y_enc_valid - pred)^2, na.rm = TRUE))
    },
    binomial = {
      prob <- stats::plogis(as.numeric(eta_arr))
      mean((prob >= 0.5) == (y_enc_valid > 0.5), na.rm = TRUE)
    },
    poisson = {
      lambda <- exp(pmin(as.numeric(eta_arr), 20))
      -sqrt(mean((y_enc_valid - lambda)^2, na.rm = TRUE))
    },
    multiclass = {
      cls <- max.col(eta_arr)  # 1-based, matches y_encoded
      mean(cls == y_enc_valid, na.rm = TRUE)
    },
    fractional = {
      pred <- stats::plogis(as.numeric(eta_arr))
      -sqrt(mean((y_enc_valid - pred)^2, na.rm = TRUE))
    }
  )
}

# Convert raw eta (from torch model) to the response-scale prediction.
pannet_eta_to_response <- function(eta_arr, family, y_center, y_scale, type, levels) {
  if (family == "gaussian") {
    mu <- as.numeric(eta_arr) * y_scale + y_center
    return(mu)
  }
  if (family == "binomial") {
    logit <- as.numeric(eta_arr)
    if (type == "link") return(logit)
    return(stats::plogis(logit))
  }
  if (family == "poisson") {
    log_lam <- as.numeric(eta_arr)
    if (type == "link") return(log_lam)
    return(exp(pmin(log_lam, 30)))
  }
  if (family == "fractional") {
    logit <- as.numeric(eta_arr)
    if (type == "link") return(logit)
    return(stats::plogis(logit))
  }
  if (family == "multiclass") {
    if (type == "link") {
      if (!is.null(levels)) colnames(eta_arr) <- levels
      return(eta_arr)
    }
    shifted   <- sweep(eta_arr, 1, apply(eta_arr, 1, max), "-")
    exp_eta   <- exp(shifted)
    prob      <- sweep(exp_eta, 1, rowSums(exp_eta), "/")
    if (!is.null(levels)) colnames(prob) <- levels
    if (type == "prob") return(prob)
    cls_idx <- max.col(prob)
    return(factor(if (!is.null(levels)) levels[cls_idx] else cls_idx, levels = levels))
  }
}
