"""Blend / rank-average OOF predictions. Always fit blend weights on OOF, apply to test preds."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


def blend(preds, w):
    return sum(wi * np.asarray(p) for wi, p in zip(w, preds))


def optimize_blend_weights(oof_list, y_true, metric_fn, minimize_metric=True):
    """Non-negative weights summing to 1, found by Nelder-Mead on the OOF metric."""
    n = len(oof_list)

    def norm(w):
        w = np.clip(w, 0, None)
        return w / (w.sum() + 1e-9)

    def loss(w):
        s = metric_fn(y_true, blend(oof_list, norm(w)))
        return s if minimize_metric else -s

    r = minimize(loss, np.ones(n) / n, method="Nelder-Mead", options={"maxiter": 1000})
    w = norm(r.x)
    print("weights:", w.round(4), "| blend score:", round(r.fun if minimize_metric else -r.fun, 5))
    return w


def rank_average(preds):
    """Scale-free average (good for AUC / ranking metrics; NOT for RMSE-type metrics)."""
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def stack_ridge(oof_list, y, test_list=None, folds=None, alpha=1.0):
    """Level-2 Ridge stacker on OOF preds (1-D preds only). Use the SAME folds as level 1."""
    from sklearn.linear_model import Ridge
    Z = np.column_stack(oof_list)
    Zt = np.column_stack(test_list) if test_list is not None else None
    if folds is None:
        m = Ridge(alpha=alpha).fit(Z, y)
        return m.predict(Z), (m.predict(Zt) if Zt is not None else None)
    oof2, tp = np.zeros(len(y)), []
    for tr, va in folds:
        m = Ridge(alpha=alpha).fit(Z[tr], np.asarray(y)[tr])
        oof2[va] = m.predict(Z[va])
        if Zt is not None:
            tp.append(m.predict(Zt))
    return oof2, (np.mean(tp, axis=0) if tp else None)
