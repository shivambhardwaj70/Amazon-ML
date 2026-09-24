# Approach Document -- Team <name> -- Amazon ML Challenge 2026
(Keep to 1-2 pages: ONE architecture diagram + ONE results table. Clarity + correct reasoning beats jargon.)

## 1. Problem framing (2-3 lines)
Target: ... | Metric: ... | Key challenge: ...

## 2. EDA insights (3-4 non-obvious bullets)
- ...

## 3. Feature engineering (best features + WHY they work)
- ...

## 4. Modeling
Architecture(s): ... | CV strategy (folds, grouping, why): ... | Why these choices: ...
Note: CV was trusted over the public LB because ... (CV-to-LB gap: ...)

## 5. Ensemble
How blended (weights / rank-avg / stack) and CV gain: ...

## 6. Results
| Model | CV (mean +/- std) | Public LB |
|---|---|---|
| TF-IDF + Ridge | | |
| LightGBM | | |
| DeBERTa-v3 | | |
| **Blend** | | |

## 7. Reproducibility
Seeds fixed (src/submit.py:set_seed). Run order: ... Hardware: ... Runtime: ...

## 8. Future work
- ...

<!-- Finale (Top 10): 6-8 slides: Problem -> Key insight -> Approach -> Results -> Learnings. Lead with ONE differentiating insight.
Rehearse: "Why this metric behavior?" "How did you prevent leakage?" "Why did the ensemble help?" "What was your CV-to-LB gap?" -->
