"""
This get executed locally by the user to submit this job.
"""

import os
from pathlib import Path
from subprocess import run

submit_script_template = """set -eo pipefail
source ~/.bashrc
source ~/.dotenv

cd {hpc_project_dir}
git pull

micromamba activate r-4.5
source .venv/bin/activate
Rscript -e 'renv::status()'
Rscript -e 'renv::restore()'

export HPC_PROJECT_DIR={hpc_project_dir}
python {jobdir}/hpc/render.py

sbatch --job-name={jobname} {jobdir}/hpc/sbatch.sh
"""

if __name__ == "__main__":
    jobdir = (Path(__file__).parent / "..").resolve()
    basedir = (jobdir / ".." / "..").resolve()
    if basedir != Path.cwd():
        raise RuntimeError(
            f"The working directory ({Path.cwd()}) is different from the guessed "
            f"base directory ({basedir}). This is unexpected."
        )

    print(jobdir.relative_to(basedir))

    submit = submit_script_template.format(
        hpc_project_dir=os.environ.get("HPC_PROJECT_DIR"),
        jobdir=jobdir.relative_to(basedir).as_posix(),
        jobname=jobdir.name,
    )

    run(
        ["C:/Program Files/Git/bin/bash.exe", str(basedir / "scripts/check_git_status.sh")],
        check=True,
    )

    run(
        ["ssh", "-q", os.environ.get("HPC_SSH_ALIAS"), "bash", "-s"],
        input=submit.replace("\r\n", "\n").encode("utf-8"),
    )
