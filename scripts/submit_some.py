from pathlib import Path
from subprocess import run

# Submit the new 2-chain mixture jobs (100-103), preliminary 5-seed grid (every run
# keeps its full posterior: data_seed <= POSTERIOR_SUBSET_MAX, see the run.qmds).
# 4 jobs x 20 rows = 80 tasks, under the scc-medium 1000 submit cap, so no packing needed.
# WARNING: do NOT re-add a job here while its array is still running (squeue --me), or you
# create duplicate concurrent arrays.
SUBMIT = [
    "100",
    "101",
    "102",
    "103",
]

JOBS_DIRECTORY: str = "jobs"

if __name__ == "__main__":
    wd = Path.cwd()
    jobs = wd / JOBS_DIRECTORY

    for dir in jobs.iterdir():
        if dir.name[:3] in SUBMIT:
            run(["python", str(dir / "hpc" / "submit.py")])
