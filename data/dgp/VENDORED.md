# Vendored DGP — provenance

These files are copied **verbatim** (do not edit) from the study repository so the
HPC harness can generate datasets reproducibly without depending on that repo.

| File | Source in `HierarchicalBayesianMNL` |
|------|-------------------------------------|
| `dgp.py` | `src/dgp.py` |
| `experiment_configs.py` | `hbmnl_mixture_experiments/experiment_configs.py` |

- **Source repo:** https://github.com/Paul-Jarschke/HierarchicalBayesianMNL
- **Vendored at commit:** `12ca13b` (2026-06-25)
- **Verified:** byte-identical to source (MD5 match) at vendoring time.

To refresh, re-copy from the source repo and update the commit hash above. If you ever
need to change the DGP, change it upstream and re-vendor — keeping these byte-identical
lets you `diff` against upstream and guarantees the data matches the study.

Used by [`../generate_mixture_data.py`](../generate_mixture_data.py).
