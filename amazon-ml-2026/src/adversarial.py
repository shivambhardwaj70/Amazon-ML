"""Adversarial validation: can a model tell train rows from test rows? AUC ~0.5 = same distribution.
AUC > ~0.75 = shift -> your CV may not reflect the test set; find the features that give it away."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score


def _prep(a, b, feats):
    both = pd.concat([a[feats], b[feats]], ignore_index=True)
    for c in both.columns:
        if both[c].dtype == object or str(both[c].dtype).startswith(("string", "category")):
            both[c] = pd.factorize(both[c])[0]
    return both


def adversarial_validation(train_df, test_df, feats, threshold=0.75, seed=42, show_importance=True):
    both = _prep(train_df, test_df, feats)
    y = np.r_[np.zeros(len(train_df)), np.ones(len(test_df))]
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=seed)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        show_importance = False
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, both, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    print(f"Adv AUC: {auc:.4f} ({'SHIFT' if auc > threshold else 'ok'})")
    if show_importance:
        model.fit(both, y)
        imp = pd.Series(model.feature_importances_, index=both.columns).sort_values(ascending=False)
        print("top shift features:\n", imp.head(10))
    return auc
