library(pannet)

test_that("pannet fits a gaussian pooled model", {
  set.seed(1)
  df <- data.frame(
    id = rep(1:5, each = 4),
    time = rep(1:4, times = 5),
    x1 = rnorm(20),
    x2 = runif(20),
    y = rnorm(20)
  )
  fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time, family = "gaussian", model = "pooled", epochs = 1, verbose = FALSE, validation = 0)
  expect_s3_class(fit, "pannet")
  expect_length(fitted(fit), nrow(df))
})

test_that("predict works on training data", {
  set.seed(1)
  df <- data.frame(
    id = rep(1:5, each = 4),
    time = rep(1:4, times = 5),
    x1 = rnorm(20),
    x2 = runif(20),
    y = rnorm(20)
  )
  fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time, family = "gaussian", model = "pooled", epochs = 1, verbose = FALSE, validation = 0)
  p <- predict(fit, df)
  expect_length(p, nrow(df))
})

test_that("binomial and poisson families are supported", {
  set.seed(1)
  df <- data.frame(
    id = rep(1:5, each = 4),
    time = rep(1:4, times = 5),
    x1 = rnorm(20),
    x2 = runif(20),
    yb = factor(sample(c(0, 1), 20, TRUE)),
    yp = rpois(20, lambda = 2)
  )
  fb <- pannet(yb ~ x1 + x2, data = df, id = id, time = time, family = "binomial", model = "pooled", epochs = 1, verbose = FALSE, validation = 0)
  fp <- pannet(yp ~ x1 + x2, data = df, id = id, time = time, family = "poisson", model = "pooled", epochs = 1, verbose = FALSE, validation = 0)
  expect_s3_class(fb, "pannet")
  expect_s3_class(fp, "pannet")
})

test_that("marginal effects return a data frame", {
  set.seed(1)
  df <- data.frame(
    id = rep(1:5, each = 4),
    time = rep(1:4, times = 5),
    x1 = rnorm(20),
    x2 = runif(20),
    y = rnorm(20)
  )
  fit <- pannet(y ~ x1 + x2, data = df, id = id, time = time, family = "gaussian", model = "pooled", epochs = 1, verbose = FALSE, validation = 0)
  me <- marginal_effects(fit, "x1")
  expect_true(is.data.frame(me))
})

test_that("dynamic prediction uses panel history", {
  set.seed(1)
  df <- data.frame(
    id = rep(1:3, each = 5),
    time = rep(1:5, times = 3),
    x1 = rnorm(15),
    y = rnorm(15)
  )
  fit <- pannet(y ~ x1, data = df, id = id, time = time, family = "gaussian", model = "dynamic", lags = 1, epochs = 1, verbose = FALSE, validation = 0.2)
  nd <- data.frame(
    id = c(1, 2, 3),
    time = c(6, 6, 6),
    x1 = rnorm(3)
  )
  p <- predict(fit, nd)
  expect_length(p, nrow(nd))
})

test_that("multiclass and fractional reporting works", {
  set.seed(1)
  df_multi <- data.frame(
    id = rep(1:4, each = 5),
    time = rep(1:5, times = 4),
    x1 = rnorm(20),
    x2 = runif(20),
    y = factor(sample(letters[1:3], 20, TRUE))
  )
  fit_multi <- pannet(y ~ x1 + x2, data = df_multi, id = id, time = time, family = "multiclass", model = "pooled", epochs = 1, verbose = FALSE, validation = 0.2)
  prob <- predict(fit_multi, df_multi, type = "prob")
  cls <- predict(fit_multi, df_multi, type = "response")
  me_multi <- marginal_effects(fit_multi, "x1")
  expect_true(is.matrix(prob))
  expect_true(is.factor(cls))
  expect_true(all(c("class", "scale") %in% names(me_multi)))

  df_frac <- data.frame(
    id = rep(1:4, each = 5),
    time = rep(1:5, times = 4),
    x1 = rnorm(20),
    y = pmin(pmax(runif(20), 0), 1)
  )
  fit_frac <- pannet(y ~ x1, data = df_frac, id = id, time = time, family = "fractional", model = "pooled", epochs = 1, verbose = FALSE, validation = 0.2)
  p_frac <- predict(fit_frac, df_frac)
  me_frac <- marginal_effects(fit_frac, "x1")
  expect_true(all(p_frac >= 0 & p_frac <= 1))
  expect_true(all(c("marginal_effect", "scale") %in% names(me_frac)))
})
