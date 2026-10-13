"""Stage 4: LogBERT, trained on normal windows only.

    python -m logmit.logbert --run normal1 --epochs 10
    python -m logmit.logbert --run normal1 --no-vhm --name logbert_novhm     # ablations
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MASK, MODEL_DIR, PAD, WINDOW
from .windows import build, load_meta, split_by_time

DEVICE = os.environ.get("LOGMIT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 8  # scoring masks every position once, WINDOW / FOLDS positions per forward pass


class LogBERT(nn.Module):
    def __init__(self, vocab, n_svc, hidden=256, layers=4, heads=4, use_svc=True):
        super().__init__()
        self.key = nn.Embedding(vocab, hidden, padding_idx=PAD)
        self.pos = nn.Embedding(WINDOW, hidden)
        self.svc = nn.Embedding(n_svc, hidden) if use_svc else None
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(hidden, vocab)
        self.register_buffer("center", torch.zeros(hidden))

    def forward(self, x, svc):
        h = self.key(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        if self.svc is not None:
            h = h + self.svc(svc)[:, None, :]
        h = self.enc(h)
        return self.head(h), h.mean(1)


def batches(n, size, shuffle=False):
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


@torch.no_grad()
def mean_rep(model, X, S, bs=1024):
    model.eval()
    total = torch.zeros_like(model.center)
    for idx in batches(len(X), bs):
        total += model(X[idx].to(DEVICE), S[idx].to(DEVICE))[1].sum(0)
    return total / len(X)


def train(w, vocab, n_svc, epochs=10, vhm=0.1, bs=256, lr=1e-3, seed=0, **arch):
    torch.manual_seed(seed)
    tr, va = split_by_time(w)
    model = LogBERT(vocab, n_svc, **arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    X, S = torch.from_numpy(tr["X"]), torch.from_numpy(tr["svc"])
    Xv, Sv = torch.from_numpy(va["X"]), torch.from_numpy(va["svc"])
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, "
          f"{len(X)} train / {len(Xv)} validation windows, device {DEVICE}")

    def loss_fn(x, s):
        mask = torch.rand(x.shape, device=x.device) < 0.15
        mask[:, 0] |= ~mask.any(1)  # at least one masked position per window
        logits, rep = model(x.masked_fill(mask, MASK), s)
        mlkp = F.cross_entropy(logits[mask], x[mask])
        dist = ((rep - model.center) ** 2).sum(1).mean()
        return mlkp + vhm * dist, mlkp.item()

    model.history = []  # per epoch: mean training loss (total and MLKP part) and held-out MLKP loss
    for epoch in range(epochs):
        t = time.time()
        if vhm:
            model.center.copy_(mean_rep(model, X, S))
        model.train()
        total, mlkp = [], []
        for idx in batches(len(X), bs, shuffle=True):
            loss, part = loss_fn(X[idx].to(DEVICE), S[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total.append(loss.item()); mlkp.append(part)
        model.eval()
        with torch.no_grad():
            val = [loss_fn(Xv[idx].to(DEVICE), Sv[idx].to(DEVICE))[1] for idx in batches(len(Xv), 1024)]
        model.history.append({"epoch": epoch + 1, "train_loss": float(np.mean(total)),
                              "train_mlkp": float(np.mean(mlkp)), "val_mlkp": float(np.mean(val))})
        print(f"epoch {epoch + 1}/{epochs}  validation MLKP loss {np.mean(val):.4f}  ({time.time() - t:.0f}s)")
    return model


@torch.no_grad()
def ranks(model, X, S, bs=512):
    """Rank of the true key at each position when that position is masked (0 = top prediction)."""
    model.eval()
    X, S = torch.as_tensor(X), torch.as_tensor(S)
    out = torch.zeros(X.shape, dtype=torch.int32)
    pos = torch.arange(X.shape[1])
    for idx in batches(len(X), bs):
        x, s = X[idx].to(DEVICE), S[idx].to(DEVICE)
        for f in range(FOLDS):
            m = (pos % FOLDS == f).to(DEVICE)
            logits = model(x.masked_fill(m[None, :], MASK), s)[0][:, m]
            true = logits.gather(-1, x[:, m, None])
            out[idx[:, None], pos[m.cpu()][None, :]] = (logits > true).sum(-1).int().cpu()
    return out.numpy()


def score(rank, g):
    """Anomaly score: fraction of positions whose true key is outside the top-g predictions."""
    return (rank >= g).mean(1)


def save(model, args, name="logbert"):
    torch.save({"state": model.state_dict(), "args": args}, MODEL_DIR / f"{name}.pt")


def load(name="logbert"):
    ck = torch.load(MODEL_DIR / f"{name}.pt", map_location=DEVICE)
    model = LogBERT(**ck["args"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.args = ck["args"]
    return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="normal-traffic run to train on")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--no-vhm", action="store_true")
    ap.add_argument("--no-svc", action="store_true")
    ap.add_argument("--name", default="logbert")
    a = ap.parse_args()
    meta, w = load_meta(), build(a.run)
    assert (w["y"] == 0).all(), "train on a run with no injected faults"
    args = dict(vocab=meta["vocab"], n_svc=int(w["svc"].max()) + 1, hidden=a.hidden,
                layers=a.layers, heads=a.heads, use_svc=not a.no_svc)
    model = train(w, epochs=a.epochs, vhm=0.0 if a.no_vhm else 0.1, **args)
    save(model, args, a.name)
    (MODEL_DIR / f"{a.name}.history.json").write_text(json.dumps(
        {"run": a.run, "epochs": a.epochs, "vhm_weight": 0.0 if a.no_vhm else 0.1, "lr": 1e-3, "batch": 256,
         "mask_ratio": 0.15, "seed": 0, "device": DEVICE, **args, "history": model.history}, indent=1))
    print(f"saved {MODEL_DIR / (a.name + '.pt')}")
