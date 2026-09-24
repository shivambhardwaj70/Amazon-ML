"""Metrics. Pick the one the problem statement names -- and optimize FOR it (see README)."""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmsle(y, p):
    """RMSLE == RMSE on log1p scale. Negative preds are clipped to 0."""
    return rmse(np.log1p(np.asarray(y, float)), np.log1p(np.clip(np.asarray(p, float), 0, None)))


def smape(y, p):
    """Symmetric MAPE in percent (lower is better). Used in some Amazon ML Challenge editions."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2.0
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(np.abs(y - p) / denom) * 100.0)


def f1_macro(y, p):
    """p = hard labels. If you have probabilities (n, C), pass argmax."""
    return float(f1_score(y, p, average="macro"))


def accuracy(y, p):
    return float(accuracy_score(y, p))


def best_f1_threshold(y, prob, grid=None):
    """Binary F1: tune the probability threshold on OOF, then reuse it on test."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = [f1_score(y, (prob >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


# name -> (function, lower_is_better)
METRICS = {
    "rmse": (rmse, True),
    "rmsle": (rmsle, True),
    "mae": (mae, True),
    "smape": (smape, True),
    "f1_macro": (f1_macro, False),
    "accuracy": (accuracy, False),
}
