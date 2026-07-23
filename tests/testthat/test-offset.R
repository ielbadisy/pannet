test_that("offset() in poisson matches glm() closely (pooled, linear net)", {
  set.seed(1)
  n_id <- 50; n_t <- 6
  d <- data.frame(id = rep(1:n_id, each = n_t), time = rep(1:n_t, n_id))
  d$x1 <- rnorm(nrow(d))
  d$exposure <- runif(nrow(d), 1, 10)
  d$y <- rpois(nrow(d), lambda = d$exposure * exp(0.5 * d$x1))

  g <- glm(y ~ x1 + offset(log(exposure)), data = d, family = poisson())

  f <- pannet(y ~ x1 + offset(log(exposure)), data = d, id = id, time = time,
              family = "poisson", model = "pooled", hidden = integer(0),
              epochs = 300, verbose = FALSE, seed = 1)

  pred <- predict(f, type = "response")
  expect_length(pred, nrow(d))
  # Not an exact-match test (MLP head vs GLM even with hidden=c() isn't
  # bit-identical optimization), but should track glm's fitted values
  # closely if the offset landed on the right scale.
  expect_gt(cor(pred, fitted(g)), 0.95)
  expect_equal(mean(pred) / mean(fitted(g)), 1, tolerance = 0.05)
})

test_that("offset() with gaussian family is scaled correctly despite y standardization", {
  set.seed(2)
  n_id <- 30; n_t <- 5
  d <- data.frame(id = rep(1:n_id, each = n_t), time = rep(1:n_t, n_id))
  d$x1 <- rnorm(nrow(d))
  d$base <- rnorm(nrow(d), sd = 3) # acts as the offset
  d$y <- 2 * d$x1 + d$base + rnorm(nrow(d), sd = 0.5)

  g <- glm(y ~ x1 + offset(base), data = d, family = gaussian())

  f <- pannet(y ~ x1 + offset(base), data = d, id = id, time = time,
              family = "gaussian", model = "pooled", hidden = integer(0),
              epochs = 300, verbose = FALSE, seed = 2)
  pred <- predict(f, type = "response")

  expect_gt(cor(pred, fitted(g)), 0.9)
  # if the offset/y_scale rescaling were wrong, predictions would be off by
  # roughly a constant multiplicative factor (y_scale) rather than tracking
  # glm's fitted values at a ~1:1 slope
  fit_slope <- coef(lm(pred ~ fitted(g)))[2]
  expect_equal(unname(fit_slope), 1, tolerance = 0.2)
})

test_that("offset() is rejected for family = 'multiclass'", {
  set.seed(3)
  d <- data.frame(id = rep(1:20, each = 4), time = rep(1:4, 20))
  d$x1 <- rnorm(nrow(d))
  d$expo <- runif(nrow(d), 1, 5)
  d$y <- sample(c("a", "b", "c"), nrow(d), replace = TRUE)

  expect_error(
    pannet(y ~ x1 + offset(log(expo)), data = d, id = id, time = time,
           family = "multiclass", model = "pooled", epochs = 5, verbose = FALSE),
    "multiclass"
  )
})

test_that("predict() with newdata missing a required offset variable errors clearly", {
  set.seed(1)
  n_id <- 20; n_t <- 4
  d <- data.frame(id = rep(1:n_id, each = n_t), time = rep(1:n_t, n_id))
  d$x1 <- rnorm(nrow(d))
  d$exposure <- runif(nrow(d), 1, 10)
  d$y <- rpois(nrow(d), lambda = d$exposure * exp(0.3 * d$x1))

  f <- pannet(y ~ x1 + offset(log(exposure)), data = d, id = id, time = time,
              family = "poisson", model = "pooled", hidden = integer(0),
              epochs = 20, verbose = FALSE, seed = 1)

  expect_error(
    predict(f, newdata = d[1:5, c("id", "time", "x1")]),
    "offset"
  )
  expect_no_error(predict(f, newdata = d[1:5, ]))
})
