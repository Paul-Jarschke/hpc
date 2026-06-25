# Windows Setup

This repository (the "Template for Reproducible Experimentation") was written with a
Linux/macOS local machine in mind. This note documents what had to change to **develop and
run the knitr demo job (`jobs/001-demo-knitr`) on Windows**.

Most changes were made in commit `9fcc60e` *("fix knitr demo: windows: recreate python
environment, adapt for windows")*. A few items are environment prerequisites that live
outside git, plus one known bug — both are listed below.

> Scope: this covers **local development and local rendering** on Windows. The HPC itself is
> Linux, so the remote-side scripts (`template.sh.j2`, `render.py`) are unaffected. The
> `submit.py` / `download.sh` / `clear.sh` scripts run *locally*, so they are the parts that
> care about Windows.

---

## Prerequisites (install these first)

| Tool | Notes |
|------|-------|
| **R 4.5.x** | Installed at `C:\Program Files\R\R-4.5.1`. Its `bin\x64` must be on `PATH` (the knitr engine is R-driven; `quarto` finds R via `PATH`/`R_HOME`). |
| **Quarto** | e.g. `C:\Users\<you>\AppData\Local\Apps\Quarto\bin`. Verify with `quarto --version`. |
| **Git for Windows** | Provides `C:\Program Files\Git\bin\bash.exe`, needed to run the repo's `.sh` scripts. |
| **Python venv** | Created by `renv::restore()` at `.venv\Scripts\python.exe` (Python 3.13). |

Restore the R + Python environments from the project root:

```powershell
R              # bootstraps renv
# then in the R console:
renv::restore()   # installs R packages (renv.lock) + Python packages (requirements.txt) into .venv
```

---

## Changes made for Windows

### 1. `jobs/001-demo-knitr/hpc/submit.py` — run shell scripts and pipe to ssh correctly

Three problems on Windows, three fixes:

**a) A `.sh` file is not directly executable on Windows** — it must be handed to bash:

```diff
- run([str(basedir / "scripts/check_git_status.sh")], check=True)
+ run(["C:/Program Files/Git/bin/bash.exe", str(basedir / "scripts/check_git_status.sh")], check=True)
```

**b) Remote (Linux) paths must use forward slashes.** `str(Path)` yields backslashes on
Windows, which corrupt the path once sent to the HPC:

```diff
- jobdir=str(jobdir.relative_to(basedir)),
+ jobdir=jobdir.relative_to(basedir).as_posix(),
```

**c) The bash script piped over ssh must have LF line endings.** Windows strings carry CRLF
(`\r\n`), which breaks `bash -s` on the remote. Normalize and send bytes instead of text:

```diff
  run(
      ["ssh", "-q", os.environ.get("HPC_SSH_ALIAS"), "bash", "-s"],
-     input=submit,
-     text=True,
+     input=submit.replace("\r\n", "\n").encode("utf-8"),
  )
```

> Note: the **same three fixes still need to be applied to the other jobs'** `submit.py`
> (`002-demo-jupyter`, `003-liesel_gam`) — only `001` has been patched so far.

### 2. `requirements.txt` — `uvloop` has no Windows wheels

`uvloop` is Unix-only and fails to install on Windows. It was constrained to non-Windows
platforms with a PEP 508 environment marker so `renv::restore()` / `pip` skips it on Windows
but still installs it on the (Linux) HPC:

```diff
- uvloop==0.22.1
+ uvloop==0.22.1; sys_platform != 'win32'
```

### 3. `.Renviron` — point `reticulate` at the project's `.venv`

The knitr engine runs Python through R's `reticulate`, which must be told which interpreter to
use. A new `.Renviron` was added in the project root with:

```
RENV_PYTHON=".venv/Scripts/python.exe"
RETICULATE_PYTHON=".venv/Scripts/python.exe"
```

⚠️ **This file currently does not work — see [Known issue](#known-issue-renviron-is-utf-16-encoded) below.**

### 4. `renv/activate.R` — regenerated (incidental)

Recreating the environment caused `renv` to rewrite its own bootstrap script
(`renv/activate.R`). This is an automatic artifact of the renv version in use, **not** a
hand-made Windows fix; no action needed.

---

## Known issue: `.Renviron` is UTF-16 encoded

The committed `.Renviron` was saved as **UTF-16 LE with a BOM** (`FF FE ...`). R only reads
`.Renviron` as ASCII/UTF-8, so on startup it prints:

```
File .Renviron contains invalid line(s)
   ��R (too long)
They were ignored
```

i.e. **`RENV_PYTHON` / `RETICULATE_PYTHON` are silently ignored.** Rendering the demo still
works today only because `RETICULATE_PYTHON` happens to be set in the ambient environment
(and recent `reticulate` auto-discovers a project-local `.venv`). On a clean shell this would
fail to bind the right Python.

**Fix:** rewrite `.Renviron` as plain **UTF-8 (no BOM), LF line endings**. In PowerShell:

```powershell
"RENV_PYTHON=`".venv/Scripts/python.exe`"`nRETICULATE_PYTHON=`".venv/Scripts/python.exe`"`n" |
  Set-Content -Path .Renviron -Encoding utf8 -NoNewline
```

Or set the variable explicitly per-session (what we used while testing):

```powershell
$env:RETICULATE_PYTHON = "C:\Users\ThinkPad\projects\hpc\.venv\Scripts\python.exe"
```

---

## Running the knitr demo as a test job on Windows

From the project root, with R on `PATH` and the Python binding set:

```powershell
$env:PATH = "C:\Program Files\R\R-4.5.1\bin\x64;" + $env:PATH
$env:RETICULATE_PYTHON = "$PWD\.venv\Scripts\python.exe"

quarto render jobs/001-demo-knitr/run.qmd --to gfm --execute `
  -P JOB_ROW:0 -P "JOB_DIR:jobs/001-demo-knitr" -P JOB_TESTING:True
```

Expected results (with `JOB_TESTING:True`):

- `jobs/001-demo-knitr/out-test/results/results-row0000.csv` is written (test mode → `out-test/`, not `out/`).
- `jobs/001-demo-knitr/log/run-0000.log` records the run.
- **No** `finished/0` marker is created (the finish-marker is skipped while testing).

Set `JOB_TESTING:False` for a real run (writes to `out/` and creates the `finished/` marker
so the row is not re-run).

---

## Open follow-ups

- [ ] Re-save `.Renviron` as UTF-8 (no BOM) so the Python binding actually loads.
- [ ] Apply the three `submit.py` fixes to `002-demo-jupyter` and `003-liesel_gam`.
- [ ] Consider a `.gitattributes` enforcing `eol=lf` on `*.sh` and `*.j2` so shell/template
      files never pick up CRLF when edited on Windows.
- [ ] `download.sh` / `clear.sh` and the `scripts/*_all.py` wrappers call bash directly and are
      not yet Windows-adapted (they will need the Git Bash treatment + a populated `.env`).
