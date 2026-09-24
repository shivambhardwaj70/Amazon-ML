import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

def get_folds(df, target_col, n_splits=5, seed=42, stratified=False, group_col=None):
    if group_col is not None:
        return list(GroupKFold(n_splits).split(df, df[target_col], groups=df[group_col]))
    if stratified:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, df[target_col]))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(df))

def cv_score(df, target_col, feature_cols, model_fn, metric_fn, folds):
    oof, scores = np.zeros(len(df)), []
    for i, (tr, val) in enumerate(folds):
        Xtr, ytr = df.iloc[tr][feature_cols], df.iloc[tr][target_col]
        Xval, yval = df.iloc[val][feature_cols], df.iloc[val][target_col]
        preds = model_fn(Xtr, ytr, Xval, yval)
        oof[val] = preds
        s = metric_fn(yval, preds); scores.append(s)
        print(f"Fold {i}: {s:.5f}")
    print(f"CV: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    return oof, scores
