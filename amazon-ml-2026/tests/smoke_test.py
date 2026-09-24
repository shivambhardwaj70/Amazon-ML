"""Run: python tests/smoke_test.py   (pytest-compatible too). Everything runs on synthetic data.
Steps needing lightgbm / torch are SKIPPED (not failed) if the library isn't installed -- but on
Kaggle/Colab/local you want ZERO skips."""
import os, sys, tempfile, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.synth import make_synthetic
from src.cv import get_folds, cv_run, save_run, load_run
from src.metrics import rmsle, rmse, smape, f1_macro, best_f1_threshold
from src.text_features import build_tfidf, text_stats_df, quantities_df, add_stats_to_sparse
from src.adversarial import adversarial_validation
from src.ensemble import optimize_blend_weights, rank_average, blend, stack_ridge
from src.submit import make_submission, validate_submission, set_seed
from run_baseline import ridge_fn

train, test, sample = make_synthetic(n_train=1500, n_test=400)
y = train["price"].values


def t_metrics():
    assert rmse([1, 2], [1, 2]) == 0 and smape([1, 2], [1, 2]) == 0
    assert abs(rmsle([0, 9], [0, 9])) < 1e-12
    assert f1_macro([0, 1, 1], [0, 1, 1]) == 1.0
    t, s = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9]))
    assert s == 1.0


def t_group_folds_no_leak():
    folds = get_folds(train, "price", group_col="brand_id")
    covered = np.concatenate([va for _, va in folds])
    assert len(covered) == len(train) and len(set(covered)) == len(train), "val folds must partition data"
    for tr, va in folds:
        assert not set(train.brand_id.iloc[tr]) & set(train.brand_id.iloc[va]), "group leaked across folds!"
    sf = get_folds(train.assign(c=(y > np.median(y)).astype(int)), "c", stratified=True)
    assert len(sf) == 5


def t_cv_tfidf_ridge_beats_mean():
    Xtr, Xte = build_tfidf(train["text"], test["text"], min_df=2)
    Xtr = add_stats_to_sparse(Xtr, text_stats_df(train["text"]))
    Xte = add_stats_to_sparse(Xte, text_stats_df(test["text"]))
    folds = get_folds(train, "price", group_col="brand_id")
    oof, tp, sc = cv_run(Xtr, y, ridge_fn(True), rmsle, folds, Xte, verbose=False)
    base = rmsle(y, np.full(len(y), y.mean()))
    assert oof.shape == (len(y),) and tp.shape == (len(test),)
    assert np.mean(sc) < 0.6 * base, f"CV {np.mean(sc):.3f} not clearly better than mean baseline {base:.3f}"
    with tempfile.TemporaryDirectory() as d:
        save_run("t", oof, tp, sc, out_dir=d)
        o2, t2 = load_run("t", out_dir=d)
        assert np.allclose(o2, oof) and np.allclose(t2, tp)
    return oof, tp


def t_features():
    s = pd.Series(["Brand1 premium tea 500 ml pack of 6", "plain"])
    st, q = text_stats_df(s), quantities_df(s)
    assert st.loc[0, "unit_hits"] >= 2 and q.loc[0, "pack_n"] == 6 and q.loc[0, "first_qty"] == 500


def t_adversarial():
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(600, 4)), columns=list("wxyz"))
    same = pd.DataFrame(rng.normal(size=(600, 4)), columns=list("wxyz"))
    shifted = pd.DataFrame(rng.normal(size=(600, 4)) + [2, 0, 0, 0], columns=list("wxyz"))
    assert adversarial_validation(a, same, list("wxyz")) < 0.6
    assert adversarial_validation(a, shifted, list("wxyz")) > 0.85


def t_lgb():
    try:
        import lightgbm  # noqa
    except ImportError:
        return "SKIP (lightgbm not installed)"
    from src.lgb_model import make_lgb
    X = train[["n_words", "qty", "pack", "brand_id"]]
    folds = get_folds(train, "price", group_col="brand_id")
    fn = make_lgb("regression", params=dict(n_estimators=200, learning_rate=0.1, num_leaves=15), log_target=True, early_stopping=20)
    oof, tp, sc = cv_run(X, y, fn, rmsle, folds, test[["n_words", "qty", "pack", "brand_id"]], verbose=False)
    assert tp.shape == (len(test),) and np.mean(sc) < rmsle(y, np.full(len(y), y.mean()))


def t_ensemble(oof):
    good = y * np.exp(np.random.default_rng(1).normal(0, .05, len(y)))
    bad = y * np.exp(np.random.default_rng(2).normal(0, .8, len(y)))
    w = optimize_blend_weights([good, bad], y, rmsle)
    assert w[0] > w[1] and abs(w.sum() - 1) < 1e-6
    assert rank_average([good, bad]).shape == y.shape
    s, _ = stack_ridge([np.log1p(good), np.log1p(bad)], np.log1p(y), folds=get_folds(train, "price"))
    assert s.shape == y.shape


def t_submission(tp):
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sample_submission.csv")
        sample.to_csv(sp, index=False)
        make_submission(tp, sp, out_dir=d, tag="smoke")
        out = [f for f in os.listdir(d) if f.endswith("smoke.csv")][0]
        validate_submission(os.path.join(d, out), sp)
        try:
            make_submission(tp[:-1], sp, out_dir=d)
            raise RuntimeError("wrong-length submission should have been rejected")
        except AssertionError:
            pass


def t_transformer_importable():
    import py_compile
    py_compile.compile(os.path.join(os.path.dirname(__file__), "..", "src", "transformer.py"), doraise=True)
    try:
        import torch, transformers  # noqa
    except ImportError:
        return "SKIP (torch/transformers not installed) -- run `python -m src.transformer` on the GPU box"


if __name__ == "__main__":
    set_seed(42)
    results, oof_tp = [], None
    steps = [("metrics", t_metrics), ("group folds don't leak", t_group_folds_no_leak),
             ("tfidf+ridge CV + save/load", t_cv_tfidf_ridge_beats_mean), ("text features", t_features),
             ("adversarial validation", t_adversarial), ("lightgbm CV", t_lgb),
             ("transformer module", t_transformer_importable)]
    for name, fn in steps:
        try:
            r = fn()
            if name.startswith("tfidf"):
                oof_tp = r
            print(f"{'SKIP' if isinstance(r, str) else 'PASS'}  {name}" + (f"  -> {r}" if isinstance(r, str) else ""))
            results.append("SKIP" if isinstance(r, str) else "PASS")
        except Exception:
            print(f"FAIL  {name}"); traceback.print_exc(); results.append("FAIL")
    for name, fn, arg in [("blend/stack", t_ensemble, oof_tp[0] if oof_tp else None), ("submission format", t_submission, oof_tp[1] if oof_tp else None)]:
        try:
            fn(arg); print(f"PASS  {name}"); results.append("PASS")
        except Exception:
            print(f"FAIL  {name}"); traceback.print_exc(); results.append("FAIL")
    print(f"\n{results.count('PASS')} passed, {results.count('SKIP')} skipped, {results.count('FAIL')} failed")
    sys.exit(1 if "FAIL" in results else 0)
