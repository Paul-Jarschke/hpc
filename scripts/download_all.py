"""
Download job `out/` directories from the HPC to the LOCAL machine, so all gathering,
analysis and plotting can happen locally (the HPC only runs jobs + produces out/).

Run this LOCALLY (not on the cluster). It:
  - loads the HPC connection settings from `.env` (HPC_SSH_ALIAS, HPC_PROJECT_DIR), so you
    do not have to `source .env` first;
  - streams each job's remote `out/` as a gzip tar over ONE ssh connection and extracts it
    locally with Python's tarfile -- no bash, no scp, no remote temp file, and no fragile
    Windows<->MSYS path conversion (the reason hpc/download.sh fails on Windows);
  - downloads every job in jobs/ by default, or only the jobs whose directory name
    contains one of the substrings you pass.

    .venv/Scripts/python.exe scripts/download_all.py                 # all jobs
    .venv/Scripts/python.exe scripts/download_all.py 004-k5-nuts-c1  # just this job
    .venv/Scripts/python.exe scripts/download_all.py 004 008         # any matching jobs

Each ssh prompts once for your key passphrase; run `ssh-add` first (ssh-agent) to enter it
only once for all jobs. Afterwards: `python analysis/post_process.py` to gather, then plot.
"""

import os
import sys
import tarfile
from pathlib import Path
from subprocess import run

JOBS_DIRECTORY = "jobs"
REQUIRED_ENV = ("HPC_SSH_ALIAS", "HPC_PROJECT_DIR")


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE .env loader (no third-party dependency). Fills in vars that are
    missing or empty in the current environment; leaves already-set values untouched."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def download_job(job_dir: Path, alias: str, project_dir: str, with_posteriors: bool = False) -> bool:
    """Stream <project_dir>/jobs/<name>/out from the HPC and extract into jobs/<name>/out.

    By default the heavy out/posterior_raw is EXCLUDED - the per-run summary CSVs are all you
    need for the recovery analysis (Tier 1), so the download is KB-to-MB scale. Pass
    with_posteriors=True to also pull the Tier-2 subset posteriors (data_seed<=5, for the
    cross-sampler marginal comparison)."""
    remote_parent = f"{project_dir}/jobs/{job_dir.name}"   # home-relative POSIX path on the HPC
    archive = job_dir / "output.tar.gz"
    exclude = "" if with_posteriors else "--exclude='out/posterior_raw' "
    what = "out (incl. posteriors)" if with_posteriors else "out summaries (no posterior_raw)"
    print(f"\n=== {job_dir.name}: streaming {remote_parent}/{what} from {alias} ===")

    # One ssh connection: tar+gzip the remote out/ to stdout, captured into a local file.
    # `-C <parent> out` keeps the archive rooted at "out/" so it extracts to jobs/<name>/out.
    with open(archive, "wb") as fh:
        result = run(["ssh", alias, f"tar -czf - -C '{remote_parent}' {exclude}out"], stdout=fh)
    # tar exit 1 = "file changed as we read it": the job is still writing out/, so a file changed
    # mid-read. The archive is a valid snapshot, so accept it. exit >= 2 (or ssh failure 255) is real.
    if result.returncode > 1:
        archive.unlink(missing_ok=True)
        print(f"  FAILED: ssh/tar exit {result.returncode} "
              f"(out/ missing on the cluster, or the SSH auth failed - retry)")
        return False
    if result.returncode == 1:
        print("  note: tar exit 1 (a file changed mid-read - job still running); snapshot is fine")

    try:
        with tarfile.open(archive) as tar:
            tar.extractall(job_dir, filter="data")         # filter='data': safe extraction
    except Exception as exc:  # noqa: BLE001 - report and continue with other jobs
        print(f"  FAILED to extract: {exc}")
        return False
    finally:
        archive.unlink(missing_ok=True)

    out_dir = job_dir / "out"
    n_files = sum(1 for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() else 0
    print(f"  OK: extracted into {job_dir.name}/out/ ({n_files} files)")
    return True


if __name__ == "__main__":
    wd = Path.cwd()
    load_dotenv(wd / ".env")

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)} (checked the environment and ./.env). "
                 f"Run from the repo root where .env lives, or export them first.")
    alias = os.environ["HPC_SSH_ALIAS"]
    project_dir = os.environ["HPC_PROJECT_DIR"].rstrip("/")

    jobs_root = wd / JOBS_DIRECTORY
    if not jobs_root.is_dir():
        sys.exit(f"No '{JOBS_DIRECTORY}/' directory under {wd}. Run from the repo root.")

    args = sys.argv[1:]
    with_posteriors = "--with-posteriors" in args
    filters = [a for a in args if not a.startswith("-")]
    jobs = sorted(p for p in jobs_root.iterdir() if p.is_dir())
    if filters:
        jobs = [j for j in jobs if any(f in j.name for f in filters)]
    if not jobs:
        sys.exit(f"No jobs matched {filters or 'jobs/'}.")

    print(f"Downloading {len(jobs)} job(s) from {alias} (remote root: {project_dir})"
          + ("" if with_posteriors else "  [summaries only - excluding posterior_raw]"))
    failures = [j.name for j in jobs if not download_job(j, alias, project_dir, with_posteriors)]

    if failures:
        sys.exit(f"\nDownload failed for: {', '.join(failures)}")
    print("\nAll requested downloads finished.")
