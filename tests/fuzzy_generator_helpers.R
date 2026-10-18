# Reuse the official Chapter 8 generator; never maintain a second DGP here.
fuzzy_source <- "8. Regression discontinuity design Cutoffs as natural experiments.R"
fuzzy_expressions <- function(root) {
    lines <- readLines(file.path(root, fuzzy_source))
    start <- which(lines == "set.seed(2025)")
    end <- which(lines == "# 4.1 Demonstrating the Selection Problem")
    stopifnot(length(start) == 1L, length(end) == 1L)
    expressions <- parse(text = lines[start:(end - 1L)])
    # Keep CSV publication out of isolated tests and repeated simulations.
    expressions[!vapply(expressions, function(expr) is.call(expr) && identical(expr[[1]], as.name("write.csv")), logical(1))]
}
run_fuzzy_generator <- function(root, seed = 2025L) {
    expressions <- fuzzy_expressions(root)
    expressions[[1]] <- substitute(set.seed(seed), list(seed = seed))
    environment <- new.env(parent = globalenv())
    eval(expressions, envir = environment)
    environment
}
