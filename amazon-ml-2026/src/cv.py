"""CV harness -- the single most important file. Everything (LGBM, transformer, TF-IDF+Ridge)
plugs into `cv_run`, so every model gets the SAME folds and saves comparable OOF preds.

model_fn contract:
    model_fn(X_train, y_train, X_val, y_val, X_test=None) -> val_pred            (or)
                                                          -> (val_pred, test_pred)
X can be a DataFrame, Series (e.g. raw text for transformers), ndarray or scipy sparse matrix.
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold


def get_folds(df, target_col=None, n_splits=5, seed=42, stratified=False, group_col=None, y=None):
    """Build folds ONCE, save/reuse them for every model.

    - group_col given  -> no group appears in both train and val (prevents leakage when
      brands / sellers / products / IDs repeat). Stratified+group -> StratifiedGroupKFold.
    - stratified       -> for classification (target_col or y must be given).
    """
    n = len(df)
    labels = y if y is not None else (df[target_col].values if target_col is not None else None)
    if group_col is not None:
        groups = df[group_col].values
        if stratified and labels is not None:
            return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n), labels, groups))
        return list(GroupKFold(n_splits).split(np.zeros(n), labels, groups))
    if stratified and labels is not None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n), labels))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n)))


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def cv_run(X, y, model_fn, metric_fn, folds, X_test=None, verbose=True, name=""):
    """Returns (oof, test_pred, scores). test_pred = mean of per-fold test preds (None if no X_test)."""
    y = np.asarray(y)
    oof, test_preds, scores = None, [], []
    for i, (tr, val) in enumerate(folds):
        out = model_fn(_take(X, tr), y[tr], _take(X, val), y[val], X_test)
        vp, tp = out if isinstance(out, tuple) else (out, None)
        vp = np.asarray(vp)
        if oof is None:
            oof = np.zeros((len(y),) + vp.shape[1:], dtype=float)
        oof[val] = vp
        if tp is not None:
            test_preds.append(np.asarray(tp))
        s = metric_fn(y[val], vp)
        scores.append(s)
        if verbose:
            print(f"[{name}] fold {i}: {s:.5f}")
    if verbose:
        print(f"[{name}] CV: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    test_pred = np.mean(test_preds, axis=0) if test_preds else None
    return oof, test_pred, scores


def cv_score(df, target_col, feature_cols, model_fn, metric_fn, folds, test_df=None, **kw):
    """DataFrame convenience wrapper around cv_run."""
    X_test = test_df[feature_cols] if test_df is not None else None
    return cv_run(df[feature_cols], df[target_col].values, model_fn, metric_fn, folds, X_test, **kw)


def save_run(name, oof, test_pred, scores, out_dir="oof"):
    """Persist OOF + test preds so ensemble.py can blend/stack later (even from another machine)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{name}_oof.npy", oof)
    if test_pred is not None:
        np.save(f"{out_dir}/{name}_test.npy", test_pred)
    print(f"saved {out_dir}/{name}_*.npy  | CV {np.mean(scores):.5f} +/- {np.std(scores):.5f}")


def load_run(name, out_dir="oof"):
    import os
    oof = np.load(f"{out_dir}/{name}_oof.npy")
    tp = f"{out_dir}/{name}_test.npy"
    return oof, (np.load(tp) if os.path.exists(tp) else None)
