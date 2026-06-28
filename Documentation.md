# Documentation

Notes on notable data/modelling issues in the `k5model_mixture` study and how they are handled.

---

## bayesm cannot fit dataset kt1_s70 (a never-chosen alternative)

**Discovered:** 2026-06-28, while scaling the grid from 50 to 100 replicate seeds per scenario.

### Symptom

When the 100-seed grid was run, `bayesm` (job `009-k5-bayesm-c2`) finished **399/400** runs
while `nuts` and `hmc` finished 400/400. The one missing bayesm run was **kt1_s70**. SLURM
reported the array task as `COMPLETED, ExitCode 0`, but it produced no output - the Quarto
render had errored internally and the wrapper's exit code did not propagate it.

The render log (`jobs/009-k5-bayesm-c2/log/captured-run-0069.log`) showed:

```
14/23 [bayesm-fit]
Error:
! y takes on  3  values -- must be = p
Backtrace:
 1. └─global run_bayesm_fit(...)
 2.   └─bayesm::rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)
 3.     └─bayesm:::pandterm(paste("y takes on ", length(levely), " values -- must be = p"))
Execution halted
```

### Root cause

`bayesm::rhierMnlRwMixture` enforces an input constraint: the pooled choice vector `y` must use
**every** alternative at least once (`length(unique(y)) == p`, where `p = n_alts`). For dataset
**kt1_s70**, only **3 of the 4** alternatives are ever chosen across all ~9000 choice occasions
(300 units x 30 observations), so bayesm aborts.

This is **deterministic**, not a transient/node failure: re-running fails identically. It is the
data, not the harness.

**Why this dataset, and why only bayesm:**

- It is a **kt1 (1comp) dataset.** With a homogeneous population, all units share one coefficient
  vector, so if that draw makes one alternative unattractive, *every* unit disfavours it and it can
  receive zero choices. Multi-component scenarios (kt2/kt3/kt5) almost never do this because some
  component favours the alternative. Empirically, of the first 105 seeds, kt1_s70 is the **only**
  unfittable dataset in any scenario.
- **NUTS and HMC have no such constraint.** The gradient-based samplers fit kt1_s70 without
  complaint. That is precisely why they reached 400 while bayesm sat at 399. (Job `008-k5-bayesm-c1`
  would hit the identical error on this same dataset.)

### How it is dealt with

The fix makes "fittable by bayesm" an explicit, screened property of the data, and keeps the study
**paired** (every sampler fits the same datasets) by backfilling the dropped seed:

1. **Screen at generation.** [data/generate_mixture_data.py](data/generate_mixture_data.py) computes,
   for every dataset, how many distinct alternatives are chosen and records it in the manifest as
   `n_alts_chosen` and `all_alts_chosen` (1 if all alternatives appear, else 0). kt1_s70 is flagged
   `all_alts_chosen = 0`. The generator also prints a `NOT bayesm-fittable` line when it screens one.

2. **Generate a small buffer.** `SEEDS = range(1, 106)` (100 target + 5 spare), so there are extra
   fittable seeds to backfill any screened-out one.

3. **Select the first 100 *fittable* seeds per scenario.** Every `params.R` (jobs 004-009) filters
   `all_alts_chosen == 1`, then keeps the first `MAX_DATA_SEED` (= 100) seeds per `k_true`. The same
   fittable set is used by **all** samplers, so the paired design is preserved. Result:

   | scenario | seeds used |
   |----------|------------|
   | kt1 (1comp)      | `{1..69, 71..101}` (70 dropped, 101 backfilled) |
   | kt2 (2comp_equal)| `{1..100}` |
   | kt3 (3comp_equal)| `{1..100}` |
   | kt5 (5comp_equal)| `{1..100}` |

   Each `params.R` logs `Screened out 1 unfittable dataset(s): kt1_s70` when it runs.

4. **Run only the backfill.** Because outputs are keyed by dataset, only the genuinely-new dataset
   (kt1_s101) needs to run. [scripts/mark_finished_from_out.sh](scripts/mark_finished_from_out.sh)
   marks every row whose dataset already has an `out/runs/` result as finished, so `render.py` and
   the SLURM array run only kt1_s101 (1 row per job). The kt1_s70 outputs that nuts/hmc had already
   produced are removed (`rm out/*/kt1_s70__*.csv`) so every sampler lands at exactly the same 100
   fittable, paired datasets.

### Outcome

- Each `(scenario, sampler)` cell has **100 fittable, paired datasets**.
- kt1_s70 is permanently recorded as unfittable (manifest `all_alts_chosen = 0`, plus the code and
  this note) and is never sent to any sampler again.
- No analysis code special-cases it; the recovery tables and figures simply use the 100 fittable
  seeds.

### How to verify / reproduce

Regenerate the manifest and inspect the screen (deterministic in `seed`):

```bash
python data/generate_mixture_data.py
# kt1_s70 prints "NOT bayesm-fittable: only 3/4 alternatives chosen"

# confirm from the manifest:
python -c "import csv; rows=list(csv.DictReader(open('data/in/k5model_mixture/manifest.csv'))); \
print([r['dataset_key'] for r in rows if r['all_alts_chosen']=='0'])"
# -> ['kt1_s70']
```

If a future grid expansion (more seeds, or a changed DGP) screens out additional datasets, the same
mechanism handles them automatically: they are flagged in the manifest, excluded by `params.R`, and
backfilled from the buffer - widen `SEEDS` in `generate_mixture_data.py` if a scenario ever has
fewer than `MAX_DATA_SEED` fittable seeds.
