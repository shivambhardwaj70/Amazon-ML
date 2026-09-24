# amazon-ml-2026

Pre-built pipeline for the Amazon ML Challenge 2026 (72h hackathon). Day 1 = swap in real data, not write plumbing.

## 5-minute setup (every teammate, every machine)
```bash
git clone <repo-url> && cd amazon-ml-2026
pip install -r requirements.txt        # do NOT reinstall torch on Kaggle/Colab
python scripts/check_env.py            # versions + GPU + fp16/bf16
python tests/smoke_test.py             # target: 0 failed, 0 skipped
python run_baseline.py --demo          # full pipeline on synthetic data
python -m src.transformer              # GPU box only: DeBERTa dry run (download + OOM check)
```

## Layout
| Path | What |
|---|---|
| `src/cv.py` | Folds (Group/Stratified/KFold) + `cv_run` (OOF **and** averaged test preds) + `save_run` |
| `src/metrics.py` | rmse, rmsle, mae, smape, f1_macro, best_f1_threshold |
| `src/eda.py`, `src/adversarial.py` | Hour-1 profiling, target/ID/text reports, train-vs-test shift AUC |
| `src/text_features.py` | word+char TF-IDF, text stats, number/unit/pack extraction |
| `src/lgb_model.py` | LightGBM regressor/classifier as a `model_fn` (`log_target=True` for RMSLE) |
| `src/transformer.py` | DeBERTa fine-tuning as a `model_fn` (AMP, best-epoch, OOF + test preds) |
| `src/ensemble.py` | Weight-optimised blend, rank-average, Ridge stacker |
| `src/submit.py` | Seeds, format-validated submission, timestamped copies in `subs/` |
| `run_baseline.py` | TF-IDF + Ridge/LogReg baseline driven by a CONFIG block |
| `docs/` | Approach-doc template, results-tracker CSV (paste into a Google Sheet) |

## Contract: every model is `model_fn(X_tr, y_tr, X_val, y_val, X_test=None) -> (val_pred, test_pred)`
```python
from src.cv import get_folds, cv_run, save_run
folds = get_folds(train, "price", group_col="brand_id")      # build ONCE, reuse for every model
oof, test_pred, scores = cv_run(X, y, make_lgb("regression", log_target=True), rmsle, folds, X_test, name="lgb")
save_run("lgb", oof, test_pred, scores)                       # -> oof/lgb_oof.npy, oof/lgb_test.npy
```
Then `src.ensemble.optimize_blend_weights([oofs...], y, metric)` and apply the same weights to the saved test preds.

## Day-1 hour-1 checklist
1. Read the metric twice. RMSLE -> log1p target. F1 -> tune threshold on OOF. SMAPE -> log target + median-ish objective (L1/Huber) often helps.
2. `eda.profile`, `eda.target_report`, `eda.text_report`, `eda.repeated_ids_report` on the real data.
3. Repeating brand/seller/product IDs -> `group_col=` (leaky CV silently lies).
4. `adversarial_validation(train, test, feats)`: AUC > 0.75 = shift, investigate.
5. Read the submission rules: format, daily limit. Only the leader submits.
6. Edit CONFIG in `run_baseline.py`, run it, submit -> first leaderboard score by hour ~8.

## Known gotchas baked in
- `subsample` in LightGBM is ignored unless `subsample_freq>0` (set to 1 here).
- Early stopping on the val fold makes OOF slightly optimistic; keep it consistent across models.
- T4/P100 have no bf16 -> code auto-picks fp16 there and bf16 on A100/L4.
- DeBERTa-v3 fast-tokenizer can fail to convert -> `sentencepiece` + `protobuf`; code falls back to the slow tokenizer.
- Trust local CV over the public LB. Save OOF for every model. Set seeds. Never hand-edit a submission file.
