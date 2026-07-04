from pathlib import Path
from subprocess import run

# 50-seed grid (200 rows/job; already-completed rows are marked finished via
# scripts/mark_finished_from_out.sh, so <= 180 tasks remain per job). Submitted in LIST
# ORDER: NUTS first (it defines the total makespan), then HMC, then the fast Gibbs arms.
# 4 x ~180 = ~720 tasks fits under the scc-medium QOSMaxSubmitJobPerUserLimit (~1000) in
# ONE submission round. If sbatch is ever rejected anyway, re-run this script once the
# queue has drained - finished/ markers make resubmission idempotent (only missing rows
# are submitted, nothing double-runs).
# WARNING: do NOT re-add a job here while its array is still running (squeue --me), or you
# create duplicate concurrent arrays.
SUBMIT = [
    "103",
    "102",
    "100",
    "101",
]

JOBS_DIRECTORY: str = "jobs"

if __name__ == "__main__":
    wd = Path.cwd()
    jobs = wd / JOBS_DIRECTORY

    for prefix in SUBMIT:                       # submit in LIST order (longest job first)
        matches = [d for d in sorted(jobs.iterdir()) if d.name.startswith(prefix)]
        if not matches:
            print(f"WARNING: no job dir matches prefix '{prefix}' - skipped")
            continue
        for dir in matches:
            run(["python", str(dir / "hpc" / "submit.py")])
