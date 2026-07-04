from pathlib import Path
from subprocess import run

# Full 100-seed grid (400 rows/job; the 5-seed rows are already marked finished via
# scripts/mark_finished_from_out.sh, so ~380 tasks remain per job). 4 x 380 = 1520 tasks
# exceeds the scc-medium ~1000 queued-task cap -> submit in TWO WAVES:
#   wave 1: "100", "101"   (the fast Gibbs arms, ~760 tasks)
#   wave 2: "102", "103"   (run after wave 1 has largely drained)
# WARNING: do NOT re-add a job here while its array is still running (squeue --me), or you
# create duplicate concurrent arrays.
SUBMIT = [
    "100",
    "101",
]

JOBS_DIRECTORY: str = "jobs"

if __name__ == "__main__":
    wd = Path.cwd()
    jobs = wd / JOBS_DIRECTORY

    for dir in jobs.iterdir():
        if dir.name[:3] in SUBMIT:
            run(["python", str(dir / "hpc" / "submit.py")])
