# Run with Rscript --vanilla tests/test-ch11-causalarima.R after installing R dependencies.
script_arg <- commandArgs()[startsWith(commandArgs(), "--file=")]
repo <- dirname(dirname(normalizePath(sub("--file=", "", script_arg))))
chapter <- file.path(repo, "11. Time series methods Measuring impact without a control group.R")
expressions <- parse(chapter)
definitions <- Filter(function(expr) {
  is.call(expr) && identical(expr[[1]], as.name("<-")) &&
    identical(expr[[2]], as.name("impact_CA"))
}, as.list(expressions))
if (length(definitions) != 1L) stop("impact_CA helper is missing from the Chapter 11 script")
env <- new.env(parent = .GlobalEnv)
eval(definitions[[1]], envir = env)
impact_CA <- env$impact_CA
suppressPackageStartupMessages(library(CausalArima))

assert_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-10)))
}
check_boot <- function(model, result, horizon = NULL) {
  post <- model$dates >= model$int.date
  selected <- if (is.null(horizon)) rep(TRUE, sum(post)) else model$dates[post] <= horizon
  observed <- model$y[post][selected]
  forecast <- model$forecast[selected]
  draws <- model$boot$boot.distrib[selected, , drop = FALSE]
  probs <- c(model$alpha / 2, 1 - model$alpha / 2)
  for (cumulative in c(FALSE, TRUE)) {
    obs <- if (cumulative) sum(observed) else mean(observed)
    predicted <- if (cumulative) sum(forecast) else mean(forecast)
    distribution <- if (cumulative) colSums(draws) else colMeans(draws)
    effects <- obs - distribution
    table <- if (cumulative) result$effect_cum else result$average
    expected_forecast <- c(predicted, unname(quantile(distribution, probs)), sd(distribution))
    expected_effect <- c(obs - predicted, unname(quantile(effects, probs)), sd(effects))
    assert_equal(as.numeric(table["forecasted", ]), expected_forecast)
    assert_equal(as.numeric(table["absolute_effect", ]), expected_effect)
    assert_equal(as.numeric(table["relative_effect", ]), expected_effect / predicted)
    assert_equal(as.numeric(table["observed", ]), c(obs, NA, NA, NA))
  }
  p <- min(mean(colSums(draws) >= sum(observed)), mean(colSums(draws) <= sum(observed)))
  assert_equal(unname(result$p_values["p"]), p)
}

data <- read.csv(file.path(repo, "data/time_series_campaign.csv"), colClasses = c(date = "Date"))
set.seed(42)
fit <- function(values, dates, intervention, xreg = NULL) {
  CausalArima(y = ts(values, frequency = 7), dates = dates,
              int.date = intervention, xreg = xreg, nboot = 1000)
}
models <- list(
  main = fit(data$y, data$date, data$date[85]),
  placebo = fit(data$y[1:80], data$date[1:80], data$date[60]),
  auxiliary = fit(data$x1, data$date, data$date[85]),
  without_x = fit(data$y, data$date, data$date[85]),
  with_x = fit(data$y, data$date, data$date[85], as.matrix(data$x1))
)
ns <- asNamespace("CausalArima")
namespaces <- c("impact", ".impact_summary", ".impact_summary_horizons")
original_functions <- lapply(namespaces, get, envir = ns)
stopifnot(!exists("ce", envir = .GlobalEnv, inherits = FALSE),
          nrow(models$main$boot$boot.distrib) == 36L,
          nrow(models$placebo$boot$boot.distrib) == 21L)
baseline <- list()
for (name in names(models)) {
  model <- models[[name]]
  original_model <- serialize(model, NULL)
  rng <- .Random.seed
  warnings <- character()
  result <- withCallingHandlers(impact_CA(model, format = "numeric"), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  stopifnot(length(warnings) == 0L, identical(original_model, serialize(model, NULL)),
            identical(rng, .Random.seed), !exists("ce", envir = .GlobalEnv, inherits = FALSE),
            ncol(model$boot$boot.distrib) == 1000L)
  normal_only <- model
  normal_only$boot <- NULL
  expected <- CausalArima::impact(normal_only)
  assert_equal(result$impact_norm, expected$impact_norm)
  assert_equal(result$arima, expected$arima)
  check_boot(model, result$impact_boot)
  horizons <- model$dates[c(which(model$dates >= model$int.date)[8], length(model$dates))]
  by_horizon <- impact_CA(model, format = "numeric", horizon = horizons)
  assert_equal(by_horizon$impact_norm, CausalArima::impact(normal_only, horizon = horizons)$impact_norm)
  for (h in seq_along(horizons)) check_boot(model, by_horizon$impact_boot[[h]], horizons[h])
  stopifnot(identical(original_model, serialize(model, NULL)), identical(rng, .Random.seed))
  baseline[[name]] <- result
  cat("PASS", name, "own-draw summaries, normal tables, horizons, and isolation\n")
}

ce <- models$main
ce$boot$boot.distrib <- ce$boot$boot.distrib + 1000
global_model <- serialize(ce, NULL)
rng <- .Random.seed
for (name in names(models)) assert_equal(impact_CA(models[[name]]), baseline[[name]])
error <- tryCatch(impact_CA(models$placebo, format = "invalid"), error = identity)
stopifnot(inherits(error, "error"),
          grepl("allowed 'format' values", conditionMessage(error), fixed = TRUE),
          identical(global_model, serialize(ce, NULL)), identical(rng, .Random.seed))
cat("PASS unrelated global ce and invalid-format error preserve state\n")

missing <- models$auxiliary
post <- which(missing$dates >= missing$int.date)
missing$y[post[5]] <- NA_real_
missing$forecast[5] <- NA_real_
original_missing <- serialize(missing, NULL)
invisible(impact_CA(missing))
stopifnot(identical(original_missing, serialize(missing, NULL)),
          identical(global_model, serialize(ce, NULL)), identical(rng, .Random.seed))
for (i in seq_along(namespaces)) stopifnot(identical(original_functions[[i]], get(namespaces[i], envir = ns)))
cat("PASS missing-entry input preservation and installed namespace isolation\n")
cat("CausalArima chapter workaround: all checks passed\n")
