library(tidyverse)

# ..............................................................................
# Build params.csv for the k5model_mixture experiment.
#
#   grid = (200 datasets from the manifest) x sampler{nuts,hmc} x n_chains{1,2}
#        = 800 rows, each = one overspecified K_MODEL=5 fit.
#
# Run locally with the project's R (it reads the generated manifest, which lives
# under the gitignored data/in/). This OVERWRITES params.csv, including any small
# test grid used for local validation.
# ..............................................................................

# ---- locate this script's directory (works via Rscript and RStudio) ----
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    return(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  getwd()
}

script_dir <- get_script_dir()

manifest_path <- fs::path(
  script_dir, "..", "..", "data", "in", "k5model_mixture", "manifest.csv"
)
manifest <- read_csv(manifest_path, show_col_types = FALSE)

# ---- fit grid ----
sampler  <- c("nuts", "hmc")
n_chains <- c(1, 2)

params <- manifest |>
  select(dataset_key, scenario, k_true, data_seed) |>
  crossing(sampler = sampler, n_chains = n_chains) |>
  mutate(
    k_model               = 5L,
    warmup                = 2000L,
    posterior             = 10000L,
    seed                  = 42L,
    a_mu                  = 0.01,
    a_delta               = 0.01,
    dirichlet_a           = 1.0,
    num_integration_steps = 10L,
  ) |>
  arrange(k_true, data_seed, sampler, n_chains)

write_csv(params, fs::path(script_dir, "params.csv"))
cat("Wrote", nrow(params), "rows to", fs::path(script_dir, "params.csv"), "\n")
