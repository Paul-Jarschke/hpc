"""
Simulation scenarios for the HBMNL mixture comparison study.

Design:
  - One equal-weight scenario per K in {1, 2, 3, 5}
  - 300 decision-making units x 30 observations, to roughly match the structure of the Rossi (2006) margarine example

Reference: Rossi (2006) Chapter 5.5
"""

SCENARIOS: dict[str, dict] = {

    "1comp": {
        # K=1 degenerates to standard HMNL — sanity check that both
        # samplers agree on the baseline before adding mixture complexity
        "n_units":      300,
        "n_obs":        30,
        "n_alts":       4,
        "n_demos":      2,
        "n_components": 1,
        "custom_pvec":  [1.0],
        "seed":         1,
    },

    "2comp_equal": {
        "n_units":      300,
        "n_obs":        30,
        "n_alts":       4,
        "n_demos":      2,
        "n_components": 2,
        "custom_pvec":  [0.50, 0.50],
        "seed":         1,
    },

    "3comp_equal": {
        "n_units":      300,
        "n_obs":        30,
        "n_alts":       4,
        "n_demos":      2,
        "n_components": 3,
        "custom_pvec":  [1/3, 1/3, 1/3],
        "seed":         1,
    },

    "5comp_equal": {
        # ~60 units per component on average - recovery is genuinely hard and a warning is declared by Rossi
        "n_units":      300,
        "n_obs":        30,
        "n_alts":       4,
        "n_demos":      2,
        "n_components": 5,
        "custom_pvec":  [0.20, 0.20, 0.20, 0.20, 0.20],
        "seed":         1,
    },

}