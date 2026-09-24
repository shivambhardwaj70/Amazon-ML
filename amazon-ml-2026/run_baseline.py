"""Hour 4-8 baseline in one command. Fill CONFIG once you've seen the data; try it now with --demo.

    python run_baseline.py --demo                # end-to-end on synthetic data (writes to /tmp-like demo dir)
    python run_baseline.py                       # real data per CONFIG below

Produces: oof/<NAME>_oof.npy, oof/<NAME>_test.npy, subs/<timestamp>_<NAME>.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cv import get_folds, cv_run, save_run
from src.metrics import METRICS
from src.text_features import build_tfidf, text_stats_df, add_stats_to_sparse
from src.submit import set_seed, make_submission

CONFIG = dict(
    train_path="data/train.csv", test_path="data/test.csv", sample_sub_path="data/sample_submission.csv",
    target="price",                 # target column name
    text_cols=["text"],             # text column(s) -> joined
    group_col=None,                 # e.g. "brand_id" if entities repeat  (-> GroupKFold)
    task="regression",              # "regression" | "classification"
    metric="rmsle",                 # key of src.metrics.METRICS
    log_target=True,                # True for RMSLE / skewed positive targets (fit log1p, predict expm1)
    n_splits=5, seed=42, name="tfidf_ridge",
)


def ridge_fn(log_target, alpha=1.0):
    from sklearn.linear_model import Ridge

    def fit(Xtr, ytr, Xval, yval, Xte=None):
        m = Ridge(alpha=alpha).fit(Xtr, np.log1p(ytr) if log_target else ytr)
        inv = (lambda p: np.expm1(p)) if log_target else (lambda p: p)
        return inv(m.predict(Xval)), (inv(m.predict(Xte)) if Xte is not None else None)
    return fit


def logreg_fn(C=4.0):
    from sklearn.linear_model import LogisticRegression

    def fit(Xtr, ytr, Xval, yval, Xte=None):
        m = LogisticRegression(C=C, max_iter=300).fit(Xtr, ytr)
        return m.predict_proba(Xval), (m.predict_proba(Xte) if Xte is not None else None)
    return fit


def main(cfg, demo=False):
    set_seed(cfg["seed"])
    if demo:
        from src.synth import make_synthetic
        train, test, sample = make_synthetic()
        os.makedirs("data/demo", exist_ok=True)
        sample.to_csv("data/demo/sample_submission.csv", index=False)
        cfg = {**cfg, "sample_sub_path": "data/demo/sample_submission.csv", "group_col": "brand_id", "name": "demo_" + cfg["name"]}
    else:
        train, test = pd.read_csv(cfg["train_path"]), pd.read_csv(cfg["test_path"])
    join = lambda d: d[cfg["text_cols"]].astype(str).agg(" ".join, axis=1)
    ttr, tte = join(train), join(test)
    y = train[cfg["target"]].values
    metric_fn, lower = METRICS[cfg["metric"]]

    Xtr, Xte = build_tfidf(ttr, tte)
    Xtr = add_stats_to_sparse(Xtr, text_stats_df(ttr))
    Xte = add_stats_to_sparse(Xte, text_stats_df(tte))

    clf = cfg["task"] == "classification"
    folds = get_folds(train, cfg["target"], cfg["n_splits"], cfg["seed"], stratified=clf, group_col=cfg["group_col"])
    model_fn = logreg_fn() if clf else ridge_fn(cfg["log_target"])
    mfn = (lambda yt, p: metric_fn(yt, p.argmax(1))) if clf else metric_fn
    oof, tp, scores = cv_run(Xtr, y, model_fn, mfn, folds, Xte, name=cfg["name"])
    save_run(cfg["name"], oof, tp, scores)
    preds = tp.argmax(1) if clf else tp
    make_submission(preds, cfg["sample_sub_path"], tag=cfg["name"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    main(CONFIG, ap.parse_args().demo)
