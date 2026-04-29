# Workflow for Experimentation using a High Performance Compute (HPC) Cluster

This guide assumes that setup has been completed successfully. See `hpc_setup.md`
for a detailed guide on how to set up the project repository locally and on the remote 
HPC.

Once setup is complete, the high-level experimentation workflow looks like this, taking
the demo job `jobs/001-demo-knitr` as an example.

1. **Write code** locally in `jobs/001-demo-knitr/run.qmd`.
    - Define parameters that you want to vary over experiment runs in 
      `jobs/001-demo-knitr/params.csv`. I find it convenient to create these `.csv`
      files using `jobs/001-demo-knitr/params.R`, but it does not matter how the 
      parameter files are created.
    - Your code should go in and after the section "Model Code". 
    - Make use of the logger in `run.qmd` that is already set up. Doing extensive
       logging is invaluable for debugging.
    - Job `jobs/001-demo-knitr` is a demo for working with a `run.qmd` quarto notebook
       that mixes R and Python. Even if you really only want to use R, you can use this.
       But you should leave the boilerplate Python code intact, it handles setup and
       finalization.
2. **Test code** locally by executing `run.qmd` locally cell-by-cell in interactive mode or
   by rendering it using quarto.
    - Switch the `JOB_TESTING` document parameter on and off to toggle simplified
       behavior in your `run.qmd` for testing whether code runs without errors. This
       can save a lot of time.
    - Make absolutely sure that all data that you need from one experiment run is 
       saved to `jobs/001-demo-knitr/out/` as intended.
    - Before submission of jobs to HPC, you will want to have reasonable estimations
       about how much time and memory your jobs need. You will want to request enough of 
       these resources so your jobs run without problems, but as little as possible.
3. **Configure resource requests** for HPC
    - The basics for requesting resources are defined in each job's `resources.json`.
       It contains the following fields:
    - `SBATCH_PARTITION`: The HPC partition to submit to. I commonly use `scc-cpu` or `medium`.
    - `SBATCH_CPUS_PER_TASK`: How many CPUs for each run. I often use one.
    - `SBATCH_MEM`: How much memory to be allocated for your job. If you ask for too
        much, your job may take a while to start. If you ask for too little, your job
        will fail.
    - `SBATCH_TIME`: Time limit for your job, format `"01:00:00"` for `dd:hh:mm`. 
        Maximum is two days on most HPC partitions of the GWDG. If you ask for too
        much, your job may take a while to start. If you ask for too little, your job
        will fail.
    - `SBATCH_ARRAY_MAX_CONCURRENT`: How many runs (rows of `params.csv`) can be
       executed by the HPC in parallel. Be civil and don't swamp the server with 
       thousands of parallel runs. I often use values between 50 and 100.
    - `HPC_NUMBER_OF_JOBS_TO_SUBMIT`: How many jobs to actually submit. The GWGD HPC
       by default allows a total of 1000 job submissions at a time, but your params.csv
       may contain more rows than that. So choosing a value <= 1000 here ensures you are
       in the allowed range.
    - `JOBROWS_PER_ARRAY_SUBJOB`: Can be used to define how many rows of params.csv 
       should be run by each job in the job array. For example, if you have 5000 rows
       in params.csv and each job takes only 3 hours, you may choose 
       HPC_NUMBER_OF_JOBS_TO_SUBMIT = 1000 and JOBROWS_PER_ARRAY_SUBJOB=5. Then you
       will submit 1000 jobs, each of which runs 5 rows of params.csv.
4. **Test submission** to HPC
    - Upload the input data to the HPC. By default, the input data are ignored via `.gitignore` to avoid committing large files to a Git repository. If your data is not large, you can simply remove this line from the `.gitignore`. Otherwise, you should upload your data manually to the HPC. This is often possible simply via drag-and-drop in the VS Code / Positron remote explorer.
    - Use a minimal `params.csv` with just one row that you are quite certain will run
       quickly.
    - Submit your job by executing `python jobs/001-demo-knitr/hpc/submit.py`.
    - It may take a while until your job starts. You can check the status of your job
       on the server using `squeue --me`. The default settings in this repository should
       also ensure that you receive emails when jobs start and finish.
    - Inspect the content of `jobs/001-demo-knitr/out/` and the logs in 
       `jobs/001-demo-knitr/log/`. Is everything running as expected? Only if yes, 
       proceed to the next step.
5. **Submit all jobs** to HPC
    - Create the full `jobs/001-demo-knitr/params.R`.
    - Run `python jobs/001-demo-knitr/hpc/submit.py`.
    - Alternatively, to submit all jobs for all directories in `jobs/`, run 
       `python scripts/submit_all.py`.
6. **Download data** from HPC to your local machine
    - Run `bash jobs/001-demo-knitr/hpc/download.sh` to download the data for job `001`.
    - Alternatively, run `python scripts/download_all.py` to download the data for 
       all directories in `jobs`.
7. **Gather data**
    - After downloading, the data is still scattered in multiple super-short `.csv`
       files in `jobs/001-demo-knitr/out`, and possibly over multiple jobs. We want to
       collect them into a few larger files.
    - Run the code in `scripts/gather_out_greedy.R` or `scripts/gather_out_lazy.R` 
       in an interactive R session. This will
       collect the data from all jobs and runs and place them in `data/out/jobs/` as 
       aggregated .csv files.
8. **Analyse data**
    - This is where this guide ends. I like to do my data analysis in R and keep the
       analysis scripts in the same repository. I would add a directory `analysis`,
       where I place my analysis scripts.

## When to create new directories in `jobs/`?

Usually, I try to create a new subdirectory in `jobs/` for larger, independent pieces
of code. For example, if I want to run a simulation study on my model and some 
competitor models, my jobs directory my look like this:

```text
jobs/
    001-my_model/
        run.qmd
        params.csv
    002-competitor1/
        run.qmd
        params.csv
    003-competitor2/
        run.qmd
        params.csv
```

However, I also often create new directories during model development to try out stuff:

```text
jobs/
    001-my_model-prototype/
        run.qmd
        params.csv
    002-my_model-with_changes/
        run.qmd
        params.csv
    003-my_model-trying_out_things/
        run.qmd
        params.csv
```

When I create a new directory in `jobs/`, I always copy an existing one and just
change the things that need changing.


## Other useful scripts in this repository

- `jobs/001-demo-knitr/hpc/clear.sh` Each directory in `jobs/` contains this script.
   It will delete all log-files on the remote HPC server. Use this before you re-submit
   a job. It can be executed via `bash jobs/001-demo-knitr/hpc/clear.sh`.
   
   a. The `clear.sh` script will not delete the `jobs/001-demo-knitr/out/` directory
      on the remote server by default. But it can do so, if you pass the corresponding 
      flag: `bash jobs/001-demo-knitr/hpc/clear.sh --delete-out`

- `scripts/clear_all.py` calls the clearing script for all jobs on the HPC.
- `scripts/clear_locally.py` does the same as the clearing script, but for all local
   job directories.
- `scripts/run_jobs_locally.py` executes jobs locally in sequence. It contains 
   a list of jobs to be executed that you can (and should) adapt to list all jobs
   that you want to be executed.

## How to customize the submit script

Each jobs has its own submit script in `jobs/001-demo-knitr/hpc/template.sh.j2`.
If you need to customize it for your job, this is the place to do it.
