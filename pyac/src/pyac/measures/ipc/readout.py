from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def fit_readout(
    states: np.ndarray,
    targets: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    states_arr = np.asarray(states, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)

    ridge = Ridge(alpha=alpha)
    ridge.fit(states_arr, targets_arr)

    pred = ridge.predict(states_arr)
    if float(np.std(pred)) == 0.0 or float(np.std(targets_arr)) == 0.0:
        capacity = 0.0
    else:
        corr = float(np.corrcoef(pred, targets_arr)[0, 1])
        if np.isnan(corr):
            return np.asarray(ridge.coef_, dtype=np.float64).ravel(), 0.0
        capacity = corr * corr

    coefficients = np.asarray(ridge.coef_, dtype=np.float64).ravel()
    return coefficients, float(capacity)
