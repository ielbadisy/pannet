# The validated headline claim (see README.md and vignette("pannet-validation")):
# pannet recovers genuinely nonlinear covariate effects meaningfully better
# than a linear panel estimator. This is a permanent regression guard for
# that claim, not just an ad hoc benchmark -- if a future change erodes it,
# this should fail.

test_that("pannet beats a linear panel estimator on a genuinely nonlinear DGP", {
  testthat::skip_if_not_installed("panglm")
  set.seed(1)
  N <- 60; Tt <- 10
  d <- data.frame(id = rep(1:N, each = Tt), time = rep(1:Tt, N))
  x1 <- rnorm(N * Tt); x2 <- rnorm(N * Tt); x3 <- rnorm(N * Tt)
  a_i <- rep(rnorm(N, sd = 1), each = Tt)
  g_t <- rep(rnorm(Tt, sd = 0.5), N)
  eta_true <- sin(x1) + 0.5 * x2^2 + x3 + a_i + g_t
  d$x1 <- x1; d$x2 <- x2; d$x3 <- x3
  d$y <- eta_true + rnorm(N * Tt, sd = 1)

  test_ids <- sample(1:N, 15)
  train <- d[!d$id %in% test_ids, ]
  test  <- d[d$id %in% test_ids, ]
  eta_test_true <- eta_true[d$id %in% test_ids]

  f_panglm <- panglm::panglm(y ~ x1 + x2 + x3, data = train, index = c("id", "time"),
                              model = "within", family = "gaussian")
  pred_panglm <- as.matrix(test[c("x1", "x2", "x3")]) %*% coef(f_panglm)

  f_pannet <- pannet(y ~ x1 + x2 + x3, data = train, id = id, time = time,
                      family = "gaussian", model = "individual",
                      epochs = 300, verbose = FALSE, seed = 1)
  pred_pannet <- predict(f_pannet, newdata = test)

  rmse_panglm <- sqrt(mean((eta_test_true - pred_panglm)^2))
  rmse_pannet <- sqrt(mean((eta_test_true - pred_pannet)^2))

  # Validated across 5 independent seeds this margin is comfortably ~15-30%;
  # use a conservative single-seed threshold to avoid a flaky test while
  # still catching a real regression.
  expect_true(rmse_pannet < 0.95 * rmse_panglm)
})
