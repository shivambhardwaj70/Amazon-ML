"""Transformer fine-tuning (DeBERTa-v3 etc.) as a cv_run-compatible model_fn.

    fn = make_transformer_fn("microsoft/deberta-v3-base", task="regression", max_length=256, batch_size=16)
    oof, test_pred, scores = cv_run(train["text"], y_scaled_or_raw, fn, metric_fn, folds, X_test=test["text"])

Regression: target is standardized internally, preds are returned in ORIGINAL units (so if you want
RMSLE, pass y = log1p(price) yourself and expm1 afterwards). Classification: returns probabilities (n, C).
Best epoch is chosen on the val fold (loss); its weights are used for test preds.

Smoke test on the GPU box (checks download, tokenizer, OOM at max_length/batch, mixed precision):
    python -m src.transformer --model microsoft/deberta-v3-base --max_length 256 --batch_size 16
"""
import gc, math, argparse
import numpy as np


def _load_tokenizer(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # DeBERTa-v3 fast-tokenizer conversion sometimes fails -> slow tokenizer
        print(f"[tokenizer] fast load failed ({type(e).__name__}); falling back to use_fast=False")
        return AutoTokenizer.from_pretrained(name, use_fast=False)


def make_transformer_fn(model_name="microsoft/deberta-v3-base", task="regression", num_labels=None,
                        max_length=256, batch_size=16, eval_batch_size=64, lr=2e-5, epochs=3,
                        warmup_ratio=0.1, weight_decay=0.01, grad_accum=1, max_grad_norm=1.0,
                        seed=42, verbose=True):
    assert task in ("regression", "classification")
    if task == "classification":
        assert num_labels and num_labels >= 2, "pass num_labels for classification"
    n_out = 1 if task == "regression" else num_labels
    cache = {}

    def fit(Xtr, ytr, Xval, yval, Xte=None):
        import torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

        torch.manual_seed(seed); np.random.seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        use_cuda = dev == "cuda"
        bf16_ok = use_cuda and torch.cuda.get_device_capability()[0] >= 8   # T4/P100 -> fp16; A100/L4 -> bf16
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not bf16_ok)

        if "tok" not in cache:
            cache["tok"] = _load_tokenizer(model_name)
        tok = cache["tok"]

        def encode(texts):
            enc = tok([str(t) for t in texts], truncation=True, max_length=max_length)
            return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(texts))]

        def collate(rows):
            b = tok.pad([{k: r[k] for k in ("input_ids", "attention_mask")} for r in rows], return_tensors="pt")
            if "y" in rows[0]:
                b["labels"] = torch.tensor([r["y"] for r in rows])
            return b

        ytr = np.asarray(ytr)
        if task == "regression":
            mu, sd = float(ytr.mean()), float(ytr.std() + 1e-9)
            ytr_t, yval_t = ((ytr - mu) / sd).astype("float32"), ((np.asarray(yval) - mu) / sd).astype("float32")
        else:
            ytr_t, yval_t = ytr.astype("int64"), np.asarray(yval).astype("int64")

        tr_rows = encode(list(Xtr))
        for r, yy in zip(tr_rows, ytr_t):
            r["y"] = float(yy) if task == "regression" else int(yy)
        va_rows = encode(list(Xval))
        te_rows = encode(list(Xte)) if Xte is not None else None

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_out).to(dev)
        no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=lr)
        steps = math.ceil(len(tr_rows) / batch_size / grad_accum) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)

        def loss_fn(logits, labels):
            if task == "regression":
                return torch.nn.functional.mse_loss(logits.squeeze(-1).float(), labels.float())
            return torch.nn.functional.cross_entropy(logits.float(), labels)

        @torch.no_grad()
        def predict(rows, labels=None):
            model.eval()
            order = np.argsort([len(r["input_ids"]) for r in rows])          # length-sorted = faster
            outs = []
            for i in range(0, len(rows), eval_batch_size):
                b = collate([rows[j] for j in order[i:i + eval_batch_size]])
                b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    outs.append(model(**b).logits.float().cpu())
            lg = torch.cat(outs).numpy()
            res = np.empty_like(lg)
            res[order] = lg
            return res

        best_loss, best_val, best_state = float("inf"), None, None
        for ep in range(epochs):
            model.train()
            g = torch.Generator(); g.manual_seed(seed + ep)
            loader = torch.utils.data.DataLoader(tr_rows, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
            run = 0.0
            for i, b in enumerate(loader):
                b = {k: v.to(dev) for k, v in b.items()}
                labels = b.pop("labels")
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    logits = model(**b).logits
                loss = loss_fn(logits, labels) / grad_accum
                scaler.scale(loss).backward()
                run += loss.item() * grad_accum
                if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            lg = predict(va_rows)
            vl = float(loss_fn(torch.tensor(lg), torch.tensor(yval_t)).item())
            if verbose:
                print(f"  epoch {ep + 1}/{epochs} train_loss={run / len(loader):.4f} val_loss={vl:.4f}")
            if vl < best_loss:
                best_loss, best_val = vl, lg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if te_rows is not None else None

        def finish(lg):
            if task == "regression":
                return lg.squeeze(-1) * sd + mu
            e = np.exp(lg - lg.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        te_pred = None
        if te_rows is not None:
            model.load_state_dict(best_state)
            te_pred = finish(predict(te_rows))
        val_pred = finish(best_val)
        if use_cuda and verbose:
            print(f"  peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        del model, opt, sched, best_state
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        return val_pred, te_pred
    return fit


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dummy fine-tune smoke test (download + OOM + mixed precision check)")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    rng = np.random.default_rng(0)
    vocab = [f"word{i}" for i in range(500)]
    mk = lambda n: [" ".join(rng.choice(vocab, a.max_length * 2)) for _ in range(n)]   # long enough to hit max_length
    fn = make_transformer_fn(a.model, "regression", max_length=a.max_length, batch_size=a.batch_size, epochs=a.epochs)
    vp, tp = fn(mk(a.n), rng.normal(size=a.n), mk(64), rng.normal(size=64), mk(64))
    print("OK -> val", vp.shape, "test", tp.shape)
