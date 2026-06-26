#!/bin/bash

# ------------------------------------------------------------------------------
# Run this to clear the remote directory
# ATTENTION! DANGER OF DATA LOSS
# This will remove all log files.
# If the corresponding line is uncommented, it will
# even DELETE ALL OUTPUT DATA FROM THE REMOTE SERVER.
# USE WITH CAUTION!
# ------------------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------------------

DELETE_OUT=false

for arg in "$@"; do
  case "$arg" in
    --delete-out)
      DELETE_OUT=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# ==============================================================================
# USER CONFIG (update these when you move servers/repos/projects)
# ==============================================================================
# Change if your local scripts live elsewhere
GIT_CHECK_SCRIPT="scripts/check_git_status.sh"

# ==============================================================================
# STABLE SCRIPT LOGIC (usually leave untouched)
# ==============================================================================

SERVER="$HPC_SSH_ALIAS"                 # SSH host / alias
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="$(realpath "$SCRIPT_DIR_ABS/..")"
SCRIPT_DIR_REL="${SCRIPT_DIR_ABS#"$PWD"/}"
JOB_DIR_REL="${JOB_DIR#"$PWD"/}"

# ------------------------------------------------------------------------------
# Check git repo status (local)
# ------------------------------------------------------------------------------

if bash "$GIT_CHECK_SCRIPT"; then
  echo "Git repo status was checked. Continuing."
else
  echo "Git repo status check failed. Aborting." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Submit job (remote)
# ------------------------------------------------------------------------------

echo "Connecting to the server"

ssh -q "$SERVER" <<EOF
# set -eo pipefail

cd "$HPC_PROJECT_DIR"
rm -rv $JOB_DIR_REL/finished
rm -rv $JOB_DIR_REL/log
rm -rv $JOB_DIR_REL/slurm-err
rm -rv $JOB_DIR_REL/slurm-out
rm -rv $JOB_DIR_REL/out-test

if [ "$DELETE_OUT" = true ]; then
  echo "WARNING: Deleting $JOB_DIR_REL/out"
  rm -rv $JOB_DIR_REL/out
  unlink $JOB_DIR_REL/output.tar.gz
else
  echo "Keeping $JOB_DIR_REL/out (use --delete-out to remove)"
fi

EOF