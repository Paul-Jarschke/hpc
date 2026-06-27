from pathlib import Path
from subprocess import run

# Submit jobs 004-009 (matched on the leading 3 digits of the job dir name). Safe to list
# them all now that the HPC out/ and finished/ markers are cleared - there are no running
# arrays to duplicate. Each entry submits ALL 200 rows of that job's params.csv as one SLURM
# array. WARNING: do NOT re-add a job here while its array is still running (squeue --me), or
# you create duplicate concurrent arrays.
SUBMIT = [
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
]

JOBS_DIRECTORY: str = "jobs"

if __name__ == "__main__":
    wd = Path.cwd()
    jobs = wd / JOBS_DIRECTORY

    for dir in jobs.iterdir():
        if dir.name[:3] in SUBMIT:
            run(["python", str(dir / "hpc" / "submit.py")])
