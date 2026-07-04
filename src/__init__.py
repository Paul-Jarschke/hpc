# Harness-side compatibility patch (NOT vendored): numpy>=2.0 removed np.trapz in
# favour of np.trapezoid, but the vendored marginal_comparison.py calls np.trapz.
# Restoring the alias here keeps every vendored module byte-identical to upstream
# (HierarchicalBayesianMultinomialLogit @ 893e63f) while working on the pinned
# numpy 2.4.1. Any import of src.* runs this first.
import numpy as _np

if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid
