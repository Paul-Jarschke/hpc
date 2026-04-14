# This script will gather data from all job directories in JOBS_DIRECTORY
# It will aim to produce one .csv file for each subdirectory of out/ for each job.
# Say, your output directories are structured like this:

# jobs/
#   001-demo-knitr/
#     out/
#       results/
#         results-row0000.csv
#         results-row0001.csv
#   002-demo-jupyter/
#     out/
#       results/
#         results-row0000.csv
#         results-row0001.csv

# Then this script will collect all data in the /out/results/ directories
# into job-wise dataframes that are placed here:

# data/out/jobs/results/
#    results-001-demo-knitr.csv
#    results-002-demo-jupyter.csv

# So there will be one output .csv for each output type and each job. The output type
# is given by the name of the subdirectory of the out/ directory in each job.
# So in this case it is "results".

library(tidyverse)
library(fs)

JOBS_DIRECTORY <- "jobs"

jobsdirs <- path(path_wd(), JOBS_DIRECTORY) |>
  dir_ls()
data_dir <- path(path_wd(), "data", "out", JOBS_DIRECTORY)
data_dir |> dir_create()


# ..............................................................................
# ---- Gather output directories ----
# ..............................................................................

exclude <- NULL # vector of strings (job directory starts) or NULL
include <- NULL # vector of strings (job directory starts) or NULL

out_dirs <- list()
i <- 1
for (dir in jobsdirs) {
  if (str_starts(fs::path_file(dir), "_")) {
    next
  }

  if (!is.null(exclude)) {
    if (any(str_starts(fs::path_file(dir), exclude))) {
      next
    }
  }

  if (!is.null(include)) {
    if (any(str_starts(fs::path_file(dir), include[, 1]))) {
      j <- str_which(fs::path_file(dir), include[, 1])
      out_dirs[[i]] <- path(dir, include[j, 2])
      i <- i + 1
    }
  }

  this_out_dir <- path(dir, "out")

  if (dir_exists(this_out_dir)) {
    out_dirs[[i]] <- this_out_dir
    i <- i + 1
  }
}

jobsdirs <- map_chr(out_dirs, function(x) fs::path(x, "..") |> fs::path_real())

# ..............................................................................
# ---- Get unique output directory names ----
# ..............................................................................

# get list of unique output directory names
out_dir_names <- map(out_dirs, function(x) dir_ls(x) |> path_file()) |>
  unlist() |>
  unique()

# ..............................................................................
# ---- Load all output directories ----
# ..............................................................................

out <- list()

i <- 1
for (dir in out_dirs) {
  for (subdir in out_dir_names) {
    subdir_path <- path(dir, subdir)

    if (!dir_exists(subdir_path)) {
      cat("Does not exist: ", subdir_path, "\n")
      next
    }

    cat("Reading: ", subdir_path, "\n")
    df <- tryCatch(
      {
        tmp <- subdir_path |>
          dir_ls() |>
          # filter out anything that is not .csv
          as_tibble() |>
          mutate(ext = fs::path_ext(value)) |>
          filter(ext == "csv") |>
          pull(value) |>
          read_csv(show_col_types = FALSE)

        tmp
      },
      error = function(e) {
        cat("Error. Trying more robust, but slower approach.", "\n")
        subdir_path |>
          dir_ls() |>
          # filter out anything that is not .csv
          as_tibble() |>
          mutate(ext = fs::path_ext(value)) |>
          filter(ext == "csv") |>
          pull(value) |>
          map(function(x) {
            tmp <- read_csv(x, show_col_types = FALSE)
            tmp
          }) |>
          bind_rows()
      }
    )

    jobname <- jobsdirs[i] |> fs::path_file()
    out[[subdir]][[jobname]] <- df

    i <- i + 1
  }
}

out |> names()

# ..............................................................................
# ---- Save outputs ----
# ..............................................................................

choice <- utils::menu(
  c("Yes", "No"),
  title = str_glue("Do you want to OVERWRITE existing data, if it exists?")
)

cat("\n\nStarting export", "\n")
for (i in seq_along(out)) {
  out_dfs_subdir <- out[[i]]
  this_data_type_dir <- path(data_dir, names(out)[i])
  this_data_type_dir |> dir_create()

  for (j in seq_along(out_dfs_subdir)) {
    df_this_job <- out_dfs_subdir[[j]]
    this_job_name <- names(out_dfs_subdir)[j]
    this_df_file_name <- paste0(names(out)[[i]], "-", this_job_name, ".csv")

    cat("Job: \t", this_job_name, "\ndf: \t", this_df_file_name, "\n")

    this_df_file_name <- path(this_data_type_dir, this_df_file_name)

    if (fs::file_exists(this_df_file_name)) {
      if (choice == 1) {
        cat("\t Overwriting data.")
        write_csv(df_this_job, this_df_file_name)
      } else {
        cat("\t File exists. Did not write data.")
      }
    } else {
      cat("\t Writing data.")
      write_csv(df_this_job, this_df_file_name)
    }

    cat("\n\n")
  }
}
