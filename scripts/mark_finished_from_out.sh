#!/usr/bin/env bash
#
# Mark a job's params.csv rows as finished whenever their dataset ALREADY has an output file
# in out/runs, so render.py and the SLURM array run only the genuinely-new rows. Unlike
# mark_seeds_done.sh (which assumes "seeds 1..N are done"), this derives completion from the
# actual outputs, so it stays correct when the SEED SET itself changes - e.g. after screening
# out an unfittable seed and backfilling a replacement (kt1_s70 -> kt1_s101).
#
# Run ON THE HPC, per job, AFTER `git pull` brings the new params.csv and BEFORE submitting:
#
#     bash scripts/mark_finished_from_out.sh jobs/009-k5-bayesm-c2
#
# It clears finished/ first (old positional markers are stale once params.csv changes), then
# re-derives one marker per row that already has out/runs/<dataset>__*.csv.
set -euo pipefail

JOB="${1:?usage: mark_finished_from_out.sh <jobdir>}"
cd "$JOB"
[ -f params.csv ] || { echo "no params.csv in $JOB - did you git pull?" >&2; exit 1; }

mkdir -p finished
rm -f finished/*

# awk emits "<0-based row index>,<dataset_key>"; if that dataset already has a result file,
# touch its marker. The job's sampler/chains suffix on the output name is matched by the glob.
tail -n +2 params.csv | awk -F, '{ print (NR - 1) "," $1 }' | while IFS=, read -r i ds; do
  if ls "out/runs/${ds}__"*.csv >/dev/null 2>&1; then touch "finished/$i"; fi
done

done_n=$(find finished -type f | wc -l)
total=$(( $(wc -l < params.csv) - 1 ))
echo "$JOB: marked $done_n/$total rows finished (have output) -> $(( total - done_n )) rows left to run"
