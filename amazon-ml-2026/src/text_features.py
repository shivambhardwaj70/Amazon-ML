"""Text features (Amazon problems are text-heavy)."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

UNITS = r"(ml|l|litre|liter|kg|g|gm|gram|mg|cm|mm|m|inch|in|ft|oz|ounce|lb|pound|pack|count|ct|piece|pcs|pc|fl)"


def clean_text(s):
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def build_tfidf(tr, te=None, max_features=50000, min_df=3):
    """Word (1-2gram) + char_wb (3-5gram) TF-IDF, fit on train+test text (unsupervised -> OK).
    Returns (X_tr, X_te) as CSR; X_te is None if te is None."""
    tr = pd.Series(tr).astype(str)
    allt = pd.concat([tr, pd.Series(te).astype(str)]) if te is not None else tr
    w = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True, min_df=min_df)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=max_features, sublinear_tf=True, min_df=min_df)
    w.fit(allt); c.fit(allt)
    f = lambda x: hstack([w.transform(x), c.transform(x)]).tocsr()
    return f(tr), (f(pd.Series(te).astype(str)) if te is not None else None)


def text_stats(s):
    s = str(s)
    words = s.split()
    nums = re.findall(r"\d+\.?\d*", s)
    return {
        "char_len": len(s),
        "word_len": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "digit_count": sum(ch.isdigit() for ch in s),
        "num_count": len(nums),
        "max_num": max((float(n) for n in nums), default=0.0),
        "upper_ratio": sum(ch.isupper() for ch in s) / (len(s) + 1),
        "unit_hits": len(re.findall(rf"\b{UNITS}\b", s.lower())),
    }


def text_stats_df(series):
    return pd.DataFrame([text_stats(s) for s in series], index=series.index)


def extract_quantities(s):
    """(number, unit) pairs like '500 ml', '2 kg', 'pack of 6'. Returns dict of simple features.
    Adapt regexes once you've SEEN the real data (EDA hour)."""
    s = str(s).lower()
    pairs = [(float(n), u) for n, u in re.findall(rf"(\d+\.?\d*)\s*{UNITS}\b", s)]
    packs = [float(n) for n in re.findall(r"pack of\s*(\d+)", s)] + [float(n) for n in re.findall(r"(\d+)\s*(?:pack|pcs|count|ct)\b", s)]
    return {
        "n_qty": len(pairs),
        "first_qty": pairs[0][0] if pairs else np.nan,
        "max_qty": max((p[0] for p in pairs), default=np.nan),
        "pack_n": max(packs) if packs else np.nan,
    }


def quantities_df(series):
    return pd.DataFrame([extract_quantities(s) for s in series], index=series.index)


def add_stats_to_sparse(X, stats_df):
    """Append (scaled) dense stats to a sparse TF-IDF matrix for linear models."""
    d = stats_df.fillna(0).astype(float)
    d = (d - d.mean()) / (d.std() + 1e-9)
    return hstack([X, csr_matrix(d.values)]).tocsr()
