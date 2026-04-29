#!/bin/bash

# ------------------------------------------------------------------------------
# Run this to download the output data from the server
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

# Variables
SERVER="$HPC_SSH_ALIAS"

# Get the directory containing this script (absolute path)
SCRIPTDIR_ABS=$(dirname "$0")

JOBDIR_ABS=$(realpath "$SCRIPTDIR_ABS/..")

# Get the current working directory (your "parent" directory)
parent_dir=$(realpath "$PWD")

# Strip the parent_dir prefix from script_dir
jobdir="${JOBDIR_ABS#$parent_dir/}"

REMOTE_TARGET_DIR="$HPC_PROJECT_DIR/$jobdir/out"
ARCHIVE_NAME="output.tar.gz"
LOCAL_PARENT_DIR="$jobdir"

echo "[INFO] Local job directory: $jobdir"
echo "[INFO] Remote target directory: $REMOTE_TARGET_DIR"
echo "[INFO] Archive name: $ARCHIVE_NAME"

# Create local parent dir, if it does not exist
echo "[INFO] Creating local directory: $LOCAL_PARENT_DIR"
mkdir -p "$LOCAL_PARENT_DIR"

# Compress the remote target directory on the server
echo "[INFO] Creating archive on server: $SERVER"
ssh "$SERVER" "tar -czf \"\$(dirname $REMOTE_TARGET_DIR)/$ARCHIVE_NAME\" -C \"\$(dirname $REMOTE_TARGET_DIR)\" \"\$(basename $REMOTE_TARGET_DIR)\""

# Download the compressed archive using scp with progress
echo "[INFO] Downloading archive from $SERVER..."
scp "$SERVER:$(dirname "$REMOTE_TARGET_DIR")/$ARCHIVE_NAME" "$LOCAL_PARENT_DIR/"

# Uncompress the archive locally
echo "[INFO] Extracting archive locally..."
tar -xzf "$LOCAL_PARENT_DIR/$ARCHIVE_NAME" -C "$LOCAL_PARENT_DIR"

# Remove the downloaded archive locally
echo "[INFO] Removing local archive..."
rm "$LOCAL_PARENT_DIR/$ARCHIVE_NAME"

# Uncomment below if you want to prompt for deletion
# echo "[INFO] Do you want to delete the remote directory $REMOTE_TARGET_DIR on $SERVER? [y/N]"
# read -r confirm
# if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
#     echo "[INFO] Deleting remote directory..."
#     ssh "$SERVER" "rm -rf \"$REMOTE_TARGET_DIR\""
#     echo "[INFO] Remote directory deleted."
# else
#     echo "[INFO] Skipped remote deletion."
# fi

echo "[INFO] All done!"