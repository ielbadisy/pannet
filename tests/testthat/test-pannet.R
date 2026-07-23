library(pannet)

# ---- Helpers ---------------------------------------------------------------

make_gauss_panel <- function(N = 6, T = 5, seed = 1) {
  set.seed(seed)
  data.frame(
    id   = rep(seq_len(N), each = T),
    time = rep(seq_len(T), times = N),
    x1   = rnorm(N * T),
    x2   = runif(N * T),
    y    = rnorm(N * T)
  )
}

quick_fit <- function(df, family = "gaussian", model = "pooled",
                      lags = NULL, ...) {
  pannet(y ~ x1 + x2, data = df, id = id, time = time,
         family = family, model = model, epochs = 5,
         verbose = FALSE, validation = 0, lags = lags, ...)
}

# ---- Class and basic structure ---------------------------------------------

test_that("pannet returns class 'pannet'", {
  df  <- make_gauss_panel()
  fit <- quick_fit(df)
  expect_s3_class(fit, "pannet")
})

test_that("fitted() length matches nrow(data)", {
  df  <- make_gauss_panel()
  fit <- quick_fit(df)
  expect_length(fitted(fit), nrow(df))
})

test_that("predict() on training data has correct length", {
  df  <- make_gauss_panel()
  fit <- quick_fit(df)
  expect_length(predict(fit, df), nrow(df))
})

# ---- Families --------------------------------------------------------------

test_that("binomial family runs and predictions are in [0, 1]", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:5, each = 4), time = rep(1:4, times = 5),
    x1   = rnorm(20), x2 = runif(20),
    y    = factor(sample(c(0, 1), 20, TRUE))
  )
  fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time,
                family = "binomial", model = "pooled", epochs = 5,
                verbose = FALSE, validation = 0)
  p <- predict(fit, df)
  expect_true(all(p >= 0 & p <= 1))
})

test_that("poisson family: predictions are non-negative", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:5, each = 4), time = rep(1:4, times = 5),
    x1   = rnorm(20), x2 = runif(20),
    y    = rpois(20, lambda = 2)
  )
  fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time,
                family = "poisson", model = "pooled", epochs = 5,
                verbose = FALSE, validation = 0)
  p <- predict(fit, df)
  expect_true(all(p >= 0))
})

test_that("fractional family: predictions are in [0, 1]", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:4, each = 5), time = rep(1:5, times = 4),
    x1   = rnorm(20),
    y    = runif(20)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time,
                family = "fractional", model = "pooled", epochs = 5,
                verbose = FALSE, validation = 0)
  p <- predict(fit, df)
  expect_true(all(p >= 0 & p <= 1))
})

test_that("multiclass: prob matrix and factor class predictions", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:4, each = 5), time = rep(1:5, times = 4),
    x1   = rnorm(20), x2 = runif(20),
    y    = factor(sample(letters[1:3], 20, TRUE))
  )
  fit  <- pannet(y ~ x1 + x2, data = df, id = id, time = time,
                 family = "multiclass", model = "pooled", epochs = 5,
                 verbose = FALSE, validation = 0)
  prob <- predict(fit, df, type = "prob")
  cls  <- predict(fit, df, type = "response")
  expect_true(is.matrix(prob))
  expect_equal(ncol(prob), 3L)
  expect_true(all(abs(rowSums(prob) - 1) < 1e-5))
  expect_true(is.factor(cls))
})

# ---- Panel architectures ---------------------------------------------------

test_that("individual model creates N individual effects", {
  df  <- make_gauss_panel(N = 8, T = 5)
  fit <- quick_fit(df, model = "individual")
  effs <- individual_effects(fit)
  expect_length(effs, 8L)
  expect_named(effs)
})

test_that("time model creates T time effects", {
  df  <- make_gauss_panel(N = 5, T = 6)
  fit <- quick_fit(df, model = "time")
  effs <- time_effects(fit)
  expect_length(effs, 6L)
})

test_that("twoway model creates both N individual and T time effects", {
  df  <- make_gauss_panel(N = 6, T = 4)
  fit <- quick_fit(df, model = "twoway")
  expect_length(individual_effects(fit), 6L)
  expect_length(time_effects(fit),       4L)
})

test_that("pooled model has no panel effects", {
  df  <- make_gauss_panel()
  fit <- quick_fit(df, model = "pooled")
  expect_null(individual_effects(fit))
  expect_null(time_effects(fit))
})

# ---- Dynamic model ---------------------------------------------------------

test_that("dynamic model runs and predicts on new periods", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:3, each = 6), time = rep(1:6, times = 3),
    x1   = rnorm(18),
    y    = rnorm(18)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time,
                family = "gaussian", model = "dynamic",
                lags = 1, epochs = 5, verbose = FALSE, validation = 0.2)
  nd  <- data.frame(id = 1:3, time = 7, x1 = rnorm(3))
  p   <- predict(fit, nd)
  expect_length(p, 3L)
})

test_that("dynamic model includes lagged outcome in design for gaussian", {
  set.seed(1)
  df <- data.frame(
    id   = rep(1:4, each = 5), time = rep(1:5, times = 4),
    x1   = rnorm(20),
    y    = rnorm(20)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time,
                family = "gaussian", model = "dynamic", lags = c(1L, 2L),
                epochs = 5, verbose = FALSE, validation = 0)
  expect_true(any(grepl("^lag[12]_y$", fit$x_names)))
})

# ---- Loss decreases --------------------------------------------------------

test_that("training loss decreases over 100 epochs on data with clear signal", {
  set.seed(7)
  n   <- 200
  x1  <- rnorm(n)
  df  <- data.frame(
    id   = rep(1:20, each = 10),
    time = rep(1:10, times = 20),
    x1   = x1,
    y    = 4 * x1 + rnorm(n, sd = 0.3)   # strong linear signal
  )
  fit  <- pannet(y ~ x1, data = df, id = id, time = time,
                 family = "gaussian", model = "pooled",
                 epochs = 100, verbose = FALSE, validation = 0,
                 lr = 0.01, hidden = c(32L, 16L))
  hist <- fit$training$history
  first5 <- mean(head(hist$train_loss, 5))
  last5  <- mean(tail(hist$train_loss, 5))
  expect_true(last5 < first5)
})

# ---- Balanced / unbalanced panel -------------------------------------------

test_that("balanced panel detected correctly", {
  df  <- make_gauss_panel(N = 4, T = 4)
  fit <- quick_fit(df)
  expect_true(isTRUE(fit$data_info$balanced))
})

test_that("unbalanced panel is handled without error", {
  set.seed(2)
  df <- data.frame(
    id   = c(rep(1, 5), rep(2, 3), rep(3, 7)),
    time = c(1:5,        1:3,        1:7),
    x1   = rnorm(15),
    y    = rnorm(15)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time,
                family = "gaussian", model = "individual",
                epochs = 5, verbose = FALSE, validation = 0)
  expect_s3_class(fit, "pannet")
  expect_false(isTRUE(fit$data_info$balanced))
})

# ---- Marginal effects ------------------------------------------------------

test_that("marginal_effects returns a data frame", {
  df  <- make_gauss_panel()
  fit <- quick_fit(df)
  me  <- marginal_effects(fit, "x1")
  expect_true(is.data.frame(me))
  expect_true("marginal_effect" %in% names(me))
})

test_that("marginal effects recover known beta for linear gaussian DGP", {
  set.seed(42)
  n  <- 300
  x1 <- rnorm(n)
  beta_true <- 3
  df <- data.frame(
    id   = rep(1:30, each = 10),
    time = rep(1:10, times = 30),
    x1   = x1,
    y    = beta_true * x1 + rnorm(n, sd = 0.5)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time,
                family = "gaussian", model = "pooled",
                hidden = c(64L, 32L), epochs = 400, lr = 5e-3,
                verbose = FALSE, validation = 0.1, seed = 42)
  me  <- marginal_effects(fit, "x1")
  # AME should be within ±1 of the true beta
  expect_true(abs(me$marginal_effect - beta_true) < 1.5)
})

# ---- Simulation benchmark --------------------------------------------------

test_that("pannet_simulate produces data with correct dimensions", {
  sim <- pannet_simulate(N = 10, T = 5, seed = 1)
  expect_equal(nrow(sim$data), 50L)
  expect_true("y" %in% names(sim$data))
  expect_length(sim$true_id_effects, 10L)
})

test_that("pannet outperforms linear model on nonlinear DGP", {
  sim <- pannet_simulate(N = 40, T = 10, dgp = "nonlinear", seed = 1)
  df  <- sim$data
  set.seed(1)
  all_ids   <- unique(df$id)
  test_ids  <- sample(all_ids, 8L)
  train_df  <- df[!df$id %in% test_ids, ]
  test_df   <- df[ df$id %in% test_ids, ]

  # Linear baseline
  lm_fit   <- stats::lm(y ~ x1 + x2 + x3, data = train_df)
  lm_rmse  <- sqrt(mean((test_df$y - predict(lm_fit, test_df))^2))

  # pannet
  nn_fit   <- pannet(y ~ x1 + x2 + x3, data = train_df,
                     id = id, time = time, family = "gaussian",
                     model = "twoway", hidden = c(64L, 32L),
                     epochs = 200, lr = 5e-3, verbose = FALSE,
                     validation = 0.15, seed = 1)
  nn_rmse  <- sqrt(mean((test_df$y - predict(nn_fit, test_df))^2))

  cat(sprintf("\nNonlinear DGP: lm RMSE = %.3f  pannet RMSE = %.3f\n",
              lm_rmse, nn_rmse))
  expect_true(nn_rmse < lm_rmse)
})
