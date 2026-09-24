import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def clean_text(s):
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

def build_tfidf(tr, te, max_features=50000):
    w = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True, min_df=3)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=max_features, sublinear_tf=True)
    allt = pd.concat([tr, te])
    w.fit(allt)
    c.fit(allt)
    return (
        hstack([w.transform(tr), c.transform(tr)]).tocsr(),
        hstack([w.transform(te), c.transform(te)]).tocsr(),
    )

def text_stats(s):
    s = str(s)
    words = s.split()
    return {
        "char_len": len(s),
        "word_len": len(words),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "digit_count": sum(ch.isdigit() for ch in s),
        "num_count": len(re.findall(r"\d+", s)),
        "unit_hits": len(re.findall(r"\b(ml|kg|gm|cm|inch|pack|count|litre|oz)\b", s.lower())),
    }
