#!/usr/bin/env bash
#
# Mark the params.csv rows whose data_seed <= MAXSEED as already-finished, so render.py
# and the SLURM array SKIP them and run only the newer seeds. Use this when you bump the
# seed count (e.g. 50 -> 100) on a job that ALREADY ran the first MAXSEED seeds, to avoid
# re-estimating them. Outputs are keyed by dataset, so the older runs' out/ files stay put
# and the new seeds' outputs land alongside them.
#
# Run ON THE HPC, per already-run job, AFTER `git pull` brings the bumped params.csv and
# BEFORE submitting:
#
#     bash scripts/mark_seeds_done.sh jobs/005-k5-nuts-c2 50
#
# It CLEARS finished/ first (the old positional markers are stale once params.csv changes)
# then re-derives the markers from the new params.csv. Do NOT run it on a job that never
# ran the first MAXSEED seeds (e.g. a brand-new arm) - that would wrongly skip real work;
# for those, just clear finished/ and let the whole grid run.
set -euo pipefail

JOB="${1:?usage: mark_seeds_done.sh <jobdir> [max_done_seed=50]}"
MAXSEED="${2:-50}"

cd "$JOB"
[ -f params.csv ] || { echo "no params.csv in $JOB - did you git pull?" >&2; exit 1; }

mkdir -p finished
rm -f finished/*

# params.csv column 4 is data_seed; the 0-based row index is the line number after the
# header (awk NR-1 once the header is stripped). One marker file per already-done row.
tail -n +2 params.csv | awk -F, -v m="$MAXSEED" '$4 <= m { print NR - 1 }' \
  | while read -r i; do touch "finished/$i"; done

done_n=$(find finished -type f | wc -l)
total=$(( $(wc -l < params.csv) - 1 ))
echo "$JOB: marked $done_n/$total rows finished (data_seed<=$MAXSEED) -> $(( total - done_n )) rows left to run"
