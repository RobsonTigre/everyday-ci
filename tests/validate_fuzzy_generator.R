# Frozen held-out validation. Run from the official student task worktree:
# Rscript tests/validate_fuzzy_generator.R /absolute/audit/output/directory
# Every seed evaluates the canonical Chapter 8 R generator, not a copied DGP.
source("tests/fuzzy_generator_helpers.R")
root <- normalizePath(".")
out <- normalizePath(commandArgs(TRUE)[1], mustWork = TRUE)
seeds <- 1001:1500
n_workers <- 2L
Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
stopifnot(requireNamespace("rdrobust", quietly = TRUE))
writeLines(c(capture.output(sessionInfo()), paste("rdrobust", packageVersion("rdrobust")),
    paste("workers", n_workers), "seed sequence: 1001:1500; canonical Mersenne-Twister set.seed(seed) per iteration",
    "No seed-specific tuning or optional stopping; thresholds frozen before the run.",
    system2("sha256sum", shQuote(file.path(root, fuzzy_source)), stdout = TRUE)),
    file.path(out, "monte-carlo-environment.txt"))
run_one <- function(seed) {
    e <- run_fuzzy_generator(root, seed)
    d <- e$data_fuzzy
    stopifnot(identical(names(d), c("x", "y", "D", "Z", "business_age", "annual_revenue", "employees")),
        all(e$D1 >= e$D0), all(e$D == ifelse(e$Z == 1, e$D1, e$D0)),
        all(e$y0 >= 0), max(abs(e$y - e$y0 - 5000 * e$D)) < 1e-10)
    main <- rdrobust::rdrobust(d$y, d$x, c = 650, fuzzy = d$D)
    adjusted <- rdrobust::rdrobust(d$y, d$x, c = 650, fuzzy = d$D,
        covs = d[, c("business_age", "employees", "annual_revenue")])
    naive <- unname(coef(lm(y ~ D, data = d))["D"])
    ols <- unname(coef(lm(y ~ D + x + business_age + employees + annual_revenue, data = d))["D"])
    data.frame(seed = seed, naive = naive, ols = ols,
        main = main$coef[1], main_ci_lower = main$ci[3, 1], main_ci_upper = main$ci[3, 2],
        adjusted = adjusted$coef[1], adjusted_ci_lower = adjusted$ci[3, 1], adjusted_ci_upper = adjusted$ci[3, 2],
        main_h_left = main$bws[1, 1], main_h_right = main$bws[1, 2],
        main_b_left = main$bws[2, 1], main_b_right = main$bws[2, 2],
        main_stage = main$tau_T[1], main_stage_robust_z = main$z_T[3],
        adjusted_stage = adjusted$tau_T[1], adjusted_stage_robust_z = adjusted$z_T[3],
        untreated_floor_count = sum(e$y0 == 0), defiers = sum(e$D1 < e$D0),
        max_constant_effect_error = max(abs(e$y - e$y0 - 5000 * e$D)))
}
main <- function() {
    # Explicit set.seed(seed) inside every task preserves the frozen R generator's
    # RNG sequence independently of worker scheduling; workers share no RNG state.
    workers <- parallel::makePSOCKcluster(n_workers)
    on.exit(parallel::stopCluster(workers), add = TRUE)
    parallel::clusterExport(workers, c("root", "fuzzy_source", "fuzzy_expressions", "run_fuzzy_generator", "run_one"), envir = globalenv())
    results <- list()
    started <- Sys.time()
    for (batch in split(seeds, ceiling(seq_along(seeds) / 10))) {
        rows <- do.call(rbind, parallel::parLapply(workers, batch, run_one))
        results[[length(results) + 1L]] <- rows
        first <- length(results) == 1L
        write.table(rows, file.path(out, "monte-carlo-per-seed.csv"), sep = ",", row.names = FALSE,
            col.names = first, append = !first)
        cat(sprintf("%s | %d/500 seeds complete | %.1f seconds elapsed\n", Sys.time(), sum(lengths(lapply(results, function(x) x$seed))), as.numeric(difftime(Sys.time(), started, units = "secs"))))
        flush.console()
    }
    d <- do.call(rbind, results)
    stopifnot(identical(d$seed, seeds), all(is.finite(as.matrix(d))))
    summary <- data.frame(metric = character(), value = numeric(), lower = numeric(), upper = numeric(), accepted = logical())
    add <- function(metric, value, lower = -Inf, upper = Inf) {
        summary[nrow(summary) + 1L, ] <<- list(metric, value, lower, upper, value >= lower && value <= upper)
    }
    for (method in c("main", "adjusted")) {
        add(paste0(method, "_mean_bias"), mean(d[[method]] - 5000), -250, 250)
        add(paste0(method, "_robust_95_coverage"), mean(d[[paste0(method, "_ci_lower")]] <= 5000 & d[[paste0(method, "_ci_upper")]] >= 5000), .92, .98)
        add(paste0(method, "_closer_than_full_ols"), mean(abs(d[[method]] - 5000) < abs(d$ols - 5000)), .95, 1)
    }
    add("full_ols_mean_upward_bias", mean(d$ols - 5000), 2000)
    add("naive_worst_absolute_error_frequency", mean(abs(d$naive - 5000) > pmax(abs(d$ols - 5000), abs(d$main - 5000), abs(d$adjusted - 5000))))
    add("main_first_stage_minimum", min(d$main_stage), 0)
    add("main_first_stage_minimum_robust_z", min(d$main_stage_robust_z), qnorm(.975))
    add("adjusted_first_stage_minimum", min(d$adjusted_stage), 0)
    add("adjusted_first_stage_minimum_robust_z", min(d$adjusted_stage_robust_z), qnorm(.975))
    add("total_defiers", sum(d$defiers), 0, 0)
    add("maximum_constant_effect_error", max(d$max_constant_effect_error), 0, 1e-10)
    write.csv(summary, file.path(out, "monte-carlo-summary.csv"), row.names = FALSE)
    print(summary, digits = 10, row.names = FALSE)
    cat("Elapsed seconds:", as.numeric(difftime(Sys.time(), started, units = "secs")), "\n")
    if (!all(summary$accepted)) stop("Frozen acceptance criteria failed; do not tune or publish", call. = FALSE)
    cat("Frozen acceptance criteria PASSED. Workers stopped on exit.\n")
}
main()
