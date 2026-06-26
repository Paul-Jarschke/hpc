from pathlib import Path
from subprocess import run

# bayesm arm only — do NOT re-add 004-007 while their arrays are still running
# (re-submitting unfinished rows creates duplicate concurrent arrays). Add them
# back here only once squeue --me is calm and you want to mop up stragglers.
SUBMIT = [
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
