import pandas as pd
import numpy as np
import random
import os

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

def make_submission(preds, sample_sub_path, out="submission.csv"):
    sub = pd.read_csv(sample_sub_path)  # mirror exact columns + order
    tcol = [c for c in sub.columns if c.lower() not in ("id", "index")][0]
    sub[tcol] = preds
    sub.to_csv(out, index=False)
    print(sub.head())
    return sub
