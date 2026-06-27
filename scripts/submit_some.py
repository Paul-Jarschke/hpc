from pathlib import Path
from subprocess import run

# Submit only the 2-chain (c2) jobs this round: 005 nuts-c2, 007 hmc-c2, 009 bayesm-c2.
# 3 jobs x 200 rows = 600 tasks, under the scc-medium 1000 submit cap, so no packing needed.
# WARNING: do NOT re-add a job here while its array is still running (squeue --me), or you
# create duplicate concurrent arrays.
SUBMIT = [
    "005",
    "007",
    "009",
]

JOBS_DIRECTORY: str = "jobs"

if __name__ == "__main__":
    wd = Path.cwd()
    jobs = wd / JOBS_DIRECTORY

    for dir in jobs.iterdir():
        if dir.name[:3] in SUBMIT:
            run(["python", str(dir / "hpc" / "submit.py")])
