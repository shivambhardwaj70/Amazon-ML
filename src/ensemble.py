import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

def optimize_blend_weights(oof_list, y_true, metric_fn, minimize_metric=True):
    def loss(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-9)
        b = sum(wi * o for wi, o in zip(w, oof_list))
        s = metric_fn(y_true, b)
        return s if minimize_metric else -s

    r = minimize(
        loss, np.ones(len(oof_list)) / len(oof_list),
        method="Nelder-Mead", options={"maxiter": 1000},
    )
    w = np.clip(r.x, 0, None)
    w = w / w.sum()
    print("weights:", w.round(4), "score:", r.fun)
    return w

def rank_average(preds):
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
