pannet_activation <- function(activation) {
  switch(
    match.arg(activation, c("relu", "tanh", "gelu", "linear")),
    relu   = torch::nn_relu(),
    tanh   = torch::nn_tanh(),
    gelu   = torch::nn_gelu(),
    linear = torch::nn_identity()
  )
}

pannet_init_linear <- function(layer, activation = "relu") {
  if (identical(activation, "tanh")) {
    torch::nn_init_xavier_uniform_(layer$weight)
  } else {
    torch::nn_init_kaiming_uniform_(layer$weight, nonlinearity = "relu")
  }
  if (!is.null(layer$bias)) torch::nn_init_zeros_(layer$bias)
  invisible(layer)
}

pannet_block <- torch::nn_module(
  "pannet_block",
  initialize = function(input_dim, output_dim, activation,
                        dropout_rate, batch_norm, residual = FALSE) {
    self$linear <- torch::nn_linear(input_dim, output_dim)
    pannet_init_linear(self$linear, activation)
    self$bn   <- if (batch_norm) torch::nn_batch_norm1d(output_dim) else torch::nn_identity()
    self$act  <- pannet_activation(activation)
    self$drop <- if (dropout_rate > 0) torch::nn_dropout(dropout_rate) else torch::nn_identity()
    self$residual <- isTRUE(residual)
    self$proj <- if (self$residual && input_dim != output_dim) {
      p <- torch::nn_linear(input_dim, output_dim, bias = FALSE)
      torch::nn_init_zeros_(p$weight)
      p
    } else NULL
  },
  forward = function(x) {
    out <- self$drop(self$act(self$bn(self$linear(x))))
    if (self$residual) {
      skip <- if (is.null(self$proj)) x else self$proj(x)
      out  <- out + skip
    }
    out
  }
)

# Main pannet neural module.
# MLP backbone + additive scalar panel effects via nn_embedding.
# Panel effects are initialised at zero so training starts like a pooled model.
# For multiclass (output_dim = K), embeddings are K-dimensional (class-specific effects).
pannet_module <- torch::nn_module(
  "pannet_module",
  initialize = function(input_dim, hidden_units, output_dim, activation,
                        dropout, batch_norm, residual, n_id, n_time) {
    current <- as.integer(input_dim)
    self$blocks <- torch::nn_module_list()
    for (i in seq_along(hidden_units)) {
      self$blocks$append(
        pannet_block(current, hidden_units[[i]], activation,
                     dropout[[i]], batch_norm, residual)
      )
      current <- hidden_units[[i]]
    }
    self$head <- torch::nn_linear(current, as.integer(output_dim))
    torch::nn_init_zeros_(self$head$weight)
    torch::nn_init_zeros_(self$head$bias)

    self$n_id   <- as.integer(n_id)
    self$n_time <- as.integer(n_time)

    # +1 row is the sentinel used for unseen IDs/times at prediction (kept at 0 or mean)
    if (n_id > 0L) {
      self$id_emb <- torch::nn_embedding(as.integer(n_id) + 1L, as.integer(output_dim))
      torch::nn_init_zeros_(self$id_emb$weight)
    }
    if (n_time > 0L) {
      self$time_emb <- torch::nn_embedding(as.integer(n_time) + 1L, as.integer(output_dim))
      torch::nn_init_zeros_(self$time_emb$weight)
    }
  },
  forward = function(x, id_idx = NULL, time_idx = NULL) {
    h <- x
    for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
    eta <- self$head(h)
    if (!is.null(id_idx)   && self$n_id   > 0L) eta <- eta + self$id_emb(id_idx)
    if (!is.null(time_idx) && self$n_time > 0L) eta <- eta + self$time_emb(time_idx)
    eta
  }
)
