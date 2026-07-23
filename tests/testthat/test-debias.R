test_that("pannet_debias() applies the split-panel jackknife formula correctly", {
  set.seed(1)
  N <- 30; Tt <- 8
  d <- data.frame(id = rep(1:N, each = Tt), time = rep(1:Tt, N))
  d$x1 <- rnorm(N * Tt)
  a_i <- rep(rnorm(N, sd = 1), each = Tt)
  y <- numeric(N * Tt)
  for (i in seq_len(N)) {
    idx <- ((i - 1) * Tt + 1):(i * Tt)
    y[idx[1]] <- a_i[idx[1]] + rnorm(1)
    for (t in 2:Tt) y[idx[t]] <- 0.3 * d$x1[idx[t]] + 0.5 * y[idx[t - 1]] + a_i[idx[t]] + rnorm(1, sd = 0.5)
  }
  d$y <- y

  res <- pannet_debias(y ~ x1, data = d, id = id, time = time, lags = 1,
                        epochs = 20, verbose = FALSE, seed = 1)

  expect_named(res, c("corrected", "full", "half1", "half2", "fit_full", "fit_half1", "fit_half2"))
  expect_s3_class(res$fit_full, "pannet")
  expect_s3_class(res$fit_half1, "pannet")
  expect_s3_class(res$fit_half2, "pannet")
  # the correction itself is pure arithmetic on already-computed marginal
  # effects, so this should hold exactly regardless of the underlying fits
  expect_equal(unname(res$corrected["x1"]),
               unname(2 * res$full["x1"] - 0.5 * (res$half1["x1"] + res$half2["x1"])))
})

test_that("pannet_debias() requires model = 'dynamic' and a time variable", {
  set.seed(2)
  d <- data.frame(id = rep(1:20, each = 5), time = rep(1:5, 20))
  d$x1 <- rnorm(100); d$y <- rnorm(100)

  expect_error(pannet_debias(y ~ x1, data = d, id = id, lags = 1, epochs = 5, verbose = FALSE),
               "requires `time`")
  expect_error(pannet_debias(y ~ x1, data = d, id = id, time = time, model = "pooled",
                              lags = 1, epochs = 5, verbose = FALSE),
               "only applies to model = 'dynamic'")
})
