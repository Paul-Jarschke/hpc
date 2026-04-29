"""
This get executed on the remote server. Usually it does not need
to be touched manually.
"""

import json
import os
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HPC_PROJECT_DIR = os.environ.get("HPC_PROJECT_DIR")


def render_template(
    template_dir: Path,
    template_name: str,
    output_path: Path,
    context: dict,
) -> None:
    """
    Render a Jinja2 template to a file.

    Parameters
    ----------
    template_dir : Path
        Directory containing the Jinja template.
    template_name : str
        Filename of the template (e.g. 'slurm_job.sh.j2').
    output_path : Path
        Where to write the rendered script.
    context : dict
        Mapping of template variables to values.
    """
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,  # fail loudly if something is missing
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_name)
    rendered = template.render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)

    # Make script executable (optional but usually useful)
    output_path.chmod(0o755)


TEMPLATE_NAME = "template.sh.j2"
SCRIPT_NAME = "sbatch.sh"


def render_submit_script(jobdir: Path):
    hpc_dir = jobdir / "hpc"
    logdir = jobdir / "log"
    finished_dir = jobdir / "finished"
    finished_dir.mkdir(exist_ok=True)
    logdir.mkdir(exist_ok=True)

    params = pd.read_csv(jobdir / "params.csv")

    all_rows = list(range(params.shape[0]))
    finished_rows = [int(f.name) for f in finished_dir.iterdir()]
    remaining_rows = [str(i) for i in all_rows if i not in finished_rows]

    if len(remaining_rows) == 0:
        raise RuntimeError("Nothing to submit: 0 remaining rows.")

    with open(jobdir / "resources.json", "r", encoding="utf-8") as f:
        resources = json.load(f)

    print(f"Remaining: {len(remaining_rows)} jobs.")

    n_per_array = resources.pop("HPC_NUMBER_OF_JOBS_TO_SUBMIT")

    this_batch = remaining_rows[:n_per_array]

    context = resources | {
        # ---------------------------------------------
        # constant or automatically inferred settings
        "N_REMAINING": len(this_batch),
        "CONDITION_INDICES": "\n".join(this_batch),
        "JOBDIR": str(jobdir),
        "LOG_DIR": str(logdir),
        "FINISHED_DIR": str(finished_dir),
        "SBATCH_ACCOUNT": os.environ.get("HPC_PROJECT_ACCOUNT_NAME"),
        "SLURM_STDOUT_DIR": str(jobdir / "slurm-out"),
        "SLURM_STDERR_DIR": str(jobdir / "slurm-err"),
        # assumes this script is executed on the server in the right working directory
        "HPC_PROJECT_DIR": os.environ.get("HPC_PROJECT_DIR"),
    }

    render_template(
        template_dir=hpc_dir,
        template_name=TEMPLATE_NAME,
        output_path=hpc_dir / SCRIPT_NAME,
        context=context,
    )


if __name__ == "__main__":
    jobdir = (Path(__file__).parent / "..").resolve()
    basedir = (jobdir / ".." / "..").resolve()
    if basedir != Path.cwd():
        raise RuntimeError(
            f"The working directory ({Path.cwd()}) is different from the guessed "
            f"base directory ({basedir}). This is unexpected."
        )

    print(jobdir.relative_to(basedir))
    render_submit_script(jobdir.relative_to(basedir))
