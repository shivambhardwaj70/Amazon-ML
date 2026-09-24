"""Synthetic Amazon-style product data so the whole pipeline can be tested BEFORE the real data drops."""
import numpy as np
import pandas as pd

CATS = {"shampoo": 250, "protein powder": 1800, "tea": 300, "phone case": 400, "olive oil": 900, "headphones": 2500}
ADJ = ["premium", "organic", "classic", "pro", "ultra", "family", "travel", "natural"]
UNIT = ["ml", "g", "kg", "oz", "count"]


def make_synthetic(n_train=3000, n_test=800, n_brands=60, seed=0):
    rng = np.random.default_rng(seed)
    brand_eff = rng.normal(0, 0.3, n_brands)

    def gen(n, offset=0):
        brand = rng.integers(0, n_brands, n)
        cat = rng.choice(list(CATS), n)
        adj = rng.choice(ADJ, n)
        qty = rng.choice([50, 100, 200, 250, 500, 750, 1000], n)
        pack = rng.choice([1, 1, 1, 2, 3, 6], n)
        unit = rng.choice(UNIT, n)
        text = [f"Brand{b} {a} {c} {q} {u} pack of {k}" for b, a, c, q, u, k in zip(brand, adj, cat, qty, unit, pack)]
        price = np.array([CATS[c] for c in cat]) * (1 + 0.15 * (pack - 1)) * (1 + qty / 2000) \
            * np.exp(brand_eff[brand] + rng.normal(0, 0.15, n))
        df = pd.DataFrame({"id": np.arange(offset, offset + n), "brand_id": brand, "text": text,
                           "n_words": [len(t.split()) for t in text], "qty": qty, "pack": pack})
        return df, price

    train, y = gen(n_train)
    train["price"] = y
    test, _ = gen(n_test, offset=n_train)
    sample = pd.DataFrame({"id": test["id"], "price": 0.0})
    return train, test, sample
