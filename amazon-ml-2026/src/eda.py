"""Fast EDA. Hour-1 questions: exact metric? target skew? imbalance? repeating IDs? text cols? train/test shift?"""
import numpy as np


def profile(df, max_cols=60):
    print("Shape:", df.shape)
    print("\nMissing (top 20):")
    print(df.isnull().mean().sort_values(ascending=False).head(20))
    print("\nColumns:")
    for c in list(df.columns)[:max_cols]:
        print(f"  {c} [{df[c].dtype}]: {df[c].nunique()} unique | {list(df[c].dropna().unique()[:3])}")


def target_report(y, task="regression"):
    y = np.asarray(y)
    if task == "regression":
        qs = np.quantile(y, [0, .01, .25, .5, .75, .99, 1])
        skew = float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12))
        print("quantiles [0,1,25,50,75,99,100]%:", np.round(qs, 3))
        print(f"mean={y.mean():.3f} std={y.std():.3f} skew={skew:.2f}  -> skew>1: consider log1p target")
        print("zeros:", int((y == 0).sum()), "| negatives:", int((y < 0).sum()))
    else:
        vc = np.unique(y, return_counts=True)
        print("class counts:", dict(zip(*vc)))


def repeated_ids_report(train, test, col):
    """Do entities repeat inside train, and do they overlap with test? -> decides GroupKFold."""
    tr, te = set(train[col].dropna()), set(test[col].dropna())
    print(f"{col}: {train[col].nunique()} unique in {len(train)} train rows; "
          f"{len(tr & te)} shared with test ({len(tr & te) / max(len(te), 1):.1%} of test uniques)")


def text_report(s, n=3):
    s = s.astype(str)
    L = s.str.len()
    print(f"chars: mean={L.mean():.0f} p50={L.median():.0f} p95={L.quantile(.95):.0f} max={L.max()}")
    print(f"~tokens (chars/4): p95={L.quantile(.95) / 4:.0f}  -> pick transformer max_length from this")
    for t in s.sample(n, random_state=0):
        print("  >", t[:300].replace("\n", " "))
