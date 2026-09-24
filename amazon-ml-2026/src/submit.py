"""Submission + reproducibility. A wrong column name/order/length scores 0 -- always validate."""
import os, random, datetime
import numpy as np
import pandas as pd


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_submission(preds, sample_sub_path, target_col=None, out_dir="subs", tag="sub", also_write=None):
    """Mirror sample_submission exactly (columns + order + row count), fill the target column.
    Saves subs/<timestamp>_<tag>.csv (every submission is kept). `also_write`: extra path (e.g. 'submission.csv')."""
    sub = pd.read_csv(sample_sub_path)
    if target_col is None:
        target_col = [c for c in sub.columns if c.lower() not in ("id", "index", "sample_id")][0]
    preds = np.asarray(preds)
    assert len(preds) == len(sub), f"pred length {len(preds)} != sample_submission rows {len(sub)}"
    assert not np.isnan(preds.astype(float)).any(), "NaN in predictions"
    sub[target_col] = preds
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{tag}.csv")
    sub.to_csv(path, index=False)
    if also_write:
        sub.to_csv(also_write, index=False)
    validate_submission(path, sample_sub_path)
    print("wrote", path)
    return sub


def validate_submission(path, sample_sub_path):
    a, b = pd.read_csv(path), pd.read_csv(sample_sub_path)
    assert list(a.columns) == list(b.columns), f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    assert len(a) == len(b), f"rows differ: {len(a)} vs {len(b)}"
    idc = b.columns[0]
    assert (a[idc].values == b[idc].values).all(), f"'{idc}' order/values differ from sample_submission"
    assert not a.isnull().any().any(), "NaNs in submission"
    return True
