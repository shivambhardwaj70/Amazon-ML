"""LightGBM workhorse. Plugs into cv_run.

RMSLE / skewed target -> log_target=True (fits on log1p(y), returns expm1(pred) in ORIGINAL scale).
Caveat: early stopping on the validation fold makes OOF slightly optimistic. Once you know the good
n_estimators from CV, you can pass params={"n_estimators": N} and early_stopping=0 for a cleaner run.
"""
import numpy as np


def make_lgb(task="regression", params=None, log_target=False, early_stopping=200):
    import lightgbm as lgb  # lazy: file imports even if lightgbm is missing
    d = dict(n_estimators=5000, learning_rate=0.02, num_leaves=127,
             subsample=0.8, subsample_freq=1,          # subsample has NO effect without subsample_freq>0
             colsample_bytree=0.6, reg_lambda=5.0, min_child_samples=50,
             n_jobs=-1, random_state=42, verbose=-1)
    if task == "regression":
        d.update(objective="regression", metric="rmse")
    if params:
        d.update(params)

    def fit(Xtr, ytr, Xval, yval, Xte=None):
        if task == "regression":
            m = lgb.LGBMRegressor(**d)
            ytr_, yval_ = (np.log1p(ytr), np.log1p(yval)) if log_target else (ytr, yval)
        else:
            m = lgb.LGBMClassifier(**d)
            ytr_, yval_ = ytr, yval
        cbs = [lgb.log_evaluation(0)]
        if early_stopping:
            cbs.append(lgb.early_stopping(early_stopping, verbose=False))
        m.fit(Xtr, ytr_, eval_set=[(Xval, yval_)], callbacks=cbs)

        def pred(X):
            if task == "regression":
                p = m.predict(X)
                return np.expm1(p) if log_target else p
            pr = m.predict_proba(X)
            return pr[:, 1] if pr.shape[1] == 2 else pr
        return pred(Xval), (pred(Xte) if Xte is not None else None)
    return fit
