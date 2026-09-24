import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

def adversarial_validation(train_df, test_df, feats):
    a = train_df[feats].copy(); a["_t"] = 0
    b = test_df[feats].copy(); b["_t"] = 1
    both = pd.concat([a, b], ignore_index=True)
    y = both.pop("_t")
    p = cross_val_predict(
        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05),
        both, y, cv=5, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(y, p)
    print(f"Adv AUC: {auc:.4f} ({'SHIFT' if auc > 0.75 else 'ok'})")
    return auc
