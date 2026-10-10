# Run with Rscript tests/test-install.R after installing remotes.
# Downloads are replaced with controlled outcomes; the installer runs in a child
# process so assertions check its exit status, not just an error caught by source().
args <- commandArgs(trailingOnly = TRUE)
if (length(args) && args[1] == "--worker") {
  scenario <- args[2]
  installer <- args[3]
  loadNamespace("remotes")
  installed <- "remotes"
  github_request <- NULL
  env <- new.env()
  env$install.packages <- function(pkgs, ...) {
    failed <- switch(scenario, cran_failure = "fixest",
                     tidybayes_failure = "tidybayes", character())
    installed <<- union(installed, setdiff(pkgs, failed))
    if (any(pkgs %in% failed)) warning("installation had non-zero exit status")
  }
  env$requireNamespace <- function(package, quietly = FALSE, ...) {
    package %in% installed
  }
  assignInNamespace("install_github", function(repo, dependencies = NA, ...) {
    github_request <<- list(repo = repo, dependencies = dependencies)
    if (scenario == "github_failure" || !"tidybayes" %in% installed) {
      warning("CausalArima installation had non-zero exit status")
    } else {
      installed <<- union(installed, "CausalArima")
    }
  }, ns = "remotes")
  source(installer, local = env)
  if (scenario == "pinned_source") {
    stopifnot(
      identical(github_request$repo, "FMenchetti/CausalArima@890854054f7f9e64c3304b6f7ea0cb19d9143769"),
      identical(github_request$dependencies, TRUE)
    )
  }
  # A zero exit must mean CausalArima's undeclared dependency was installed first.
  if (scenario == "success") {
    stopifnot(all(c("tidybayes", "CausalArima") %in% installed))
  }
} else {
  script_arg <- commandArgs()[startsWith(commandArgs(), "--file=")]
  test_file <- normalizePath(sub("--file=", "", script_arg))
  installer <- normalizePath(if (length(args)) args[1] else "install.R")
  scenarios <- c("success", "pinned_source", "github_failure", "cran_failure", "tidybayes_failure")
  passed <- logical(length(scenarios))
  for (i in seq_along(scenarios)) {
    scenario <- scenarios[i]
    log <- tempfile()
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      c("--vanilla", shQuote(test_file), "--worker", scenario,
                        shQuote(installer)), stdout = log, stderr = log)
    output <- readLines(log, warn = FALSE)
    unlink(log)
    missing_package <- switch(scenario, github_failure = "CausalArima",
                              cran_failure = "fixest", tidybayes_failure = "tidybayes",
                              "")
    passed[i] <- if (scenario %in% c("success", "pinned_source")) status == 0L else {
      status != 0L && any(grepl(missing_package, output, fixed = TRUE))
    }
    cat(if (passed[i]) "PASS" else "FAIL", scenario, "exit", status, "\n")
    if (!passed[i]) cat(paste(output, collapse = "\n"), "\n")
  }
  stopifnot(all(passed))
}
