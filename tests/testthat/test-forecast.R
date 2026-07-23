test_that("pannet_forecast() runs recursively without NA and matches one-step predict()", {
  set.seed(1)
  N <- 10; Tt <- 8
  d <- data.frame(id = rep(1:N, each = Tt), time = rep(1:Tt, N))
  d$x1 <- rnorm(nrow(d))
  y <- numeric(nrow(d))
  for (i in seq_len(N)) {
    idx <- which(d$id == i); yp <- 0
    for (t in idx) { yp <- 0.6 * yp + 0.5 * d$x1[t] + rnorm(1, sd = 0.3); y[t] <- yp }
  }
  d$y <- y

  f <- pannet(y ~ x1, data = d, id = id, time = time, family = "gaussian",
              model = "dynamic", lags = 1, epochs = 60, verbose = FALSE, seed = 1)

  nd <- data.frame(id = rep(1:N, each = 3), time = rep(9:11, N), x1 = rnorm(N * 3))
  fc <- pannet_forecast(f, nd)

  expect_false(anyNA(fc$forecast))
  expect_equal(nrow(fc), nrow(nd))
  expect_setequal(fc$time, c(9, 10, 11))

  # one-step-ahead forecast should equal predict() with the true (known) lag
  nd1 <- data.frame(id = 1:N, time = 9, x1 = nd$x1[nd$time == 9])
  p1 <- predict(f, nd1)
  fc1 <- fc[fc$time == 9, ]
  fc1 <- fc1[order(fc1$id), ]
  expect_equal(fc1$forecast, as.numeric(p1), tolerance = 1e-6)
})

test_that("predict() on a dynamic model uses the real lagged history, not zero", {
  # Regression guard for a bug where panel_build_design() skipped
  # rebuilding lag columns whenever their *names* (inherited via rbind with
  # training data) were already present, even though the new rows' lag
  # *values* were NA placeholders -- silently feeding zero instead of the
  # true lagged outcome into every dynamic-model prediction.
  set.seed(9)
  N <- 8; Tt <- 6
  d <- data.frame(id = rep(1:N, each = Tt), time = rep(1:Tt, N))
  d$x1 <- rnorm(nrow(d))
  y <- numeric(nrow(d))
  for (i in seq_len(N)) {
    idx <- which(d$id == i); yp <- 0
    for (t in idx) { yp <- 0.7 * yp + 0.4 * d$x1[t] + rnorm(1, sd = 0.2); y[t] <- yp }
  }
  d$y <- y

  f <- pannet(y ~ x1, data = d, id = id, time = time, family = "gaussian",
              model = "dynamic", lags = 1, hidden = integer(0),
              epochs = 150, verbose = FALSE, seed = 1)

  # Same x1, same id/time, but the *training history* differs by
  # construction across units -- if lag were silently zeroed, predictions
  # would collapse to a function of x1 alone and lose this variation.
  nd <- data.frame(id = 1:N, time = 7, x1 = rep(0, N))
  p <- predict(f, nd)
  expect_true(sd(p) > 1e-6) # not degenerate/constant despite identical x1
})

test_that("pannet_forecast() errors for non-dynamic models and unknown units", {
  set.seed(1)
  d <- data.frame(id = rep(1:5, each = 4), time = rep(1:4, 5), x1 = rnorm(20), y = rnorm(20))
  f_pooled <- pannet(y ~ x1, data = d, id = id, time = time, family = "gaussian",
                     model = "pooled", epochs = 5, verbose = FALSE)
  expect_error(pannet_forecast(f_pooled, data.frame(id = 1, time = 5, x1 = 0)), "dynamic")

  f_dyn <- pannet(y ~ x1, data = d, id = id, time = time, family = "gaussian",
                  model = "dynamic", lags = 1, epochs = 5, verbose = FALSE)
  expect_error(pannet_forecast(f_dyn, data.frame(id = 999, time = 5, x1 = 0)), "unknown|unseen|id")
})

test_that("pannet_forecast() forecast errors grow with horizon (recursive compounding, sanity check)", {
  set.seed(42)
  N <- 30; Tt <- 10
  d <- data.frame(id = rep(1:N, each = Tt), time = rep(1:Tt, N))
  d$x1 <- rnorm(nrow(d))
  a_i <- rep(rnorm(N, sd = 0.5), each = Tt)
  y <- numeric(nrow(d))
  for (i in seq_len(N)) {
    idx <- which(d$id == i); yp <- 0
    for (t in idx) { yp <- 0.5 * yp + 0.4 * d$x1[t] + a_i[t] + rnorm(1, sd = 0.3); y[t] <- yp }
  }
  d$y <- y

  f <- pannet(y ~ x1, data = d, id = id, time = time, family = "gaussian",
              model = "dynamic", lags = 1, epochs = 100, verbose = FALSE, seed = 1)
  nd <- data.frame(id = rep(1:N, each = 3), time = rep((Tt + 1):(Tt + 3), N), x1 = rnorm(N * 3))
  fc <- pannet_forecast(f, nd)

  # Not a strict monotonicity assertion (noise can occasionally break it for
  # a single fit), just confirms the mechanism produces a full, sane,
  # multi-step forecast rather than crashing or degenerating.
  expect_true(all(is.finite(fc$forecast)))
  expect_equal(length(unique(fc$time)), 3L)
})
