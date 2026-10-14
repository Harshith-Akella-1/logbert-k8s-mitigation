"""A newer baseline on HDFS: LogRobust-style detection (Zhang et al., ESEC/FSE 2019).
Each template becomes a semantic vector from pretrained BERT; a session is the sequence of its
events' vectors; an attention-based Bi-LSTM classifies the session. Unlike the other detectors this
is **supervised**: it is trained on the labelled sessions (normal and anomalous) of the training
period. Same chronological 60/20/20 split as logmit.bench; threshold on the validation period;
test evaluated once. Writes results/hdfs_extra.json.

    $env:PYTHONPATH = "D:\\pylibs"; python -m logmit.bench_extra --epochs 5
"""
import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import logbert as lbm
from .bench import load_hdfs, split
from .config import RESULTS
from .evaluate import best_tau, prf

DEVICE = lbm.DEVICE
MAX_LEN = 128


@torch.no_grad()
def template_vectors(templates, model_name="bert-base-uncased"):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    ids = sorted(templates)
    text = [re.sub(r"\[\*\]|<\*>", " ", templates[i]).strip() for i in ids]
    enc = tok(text, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    h = bert(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    vec = ((h * m).sum(1) / m.sum(1)).cpu()          # mean of token states
    table = torch.zeros(max(ids) + 1, vec.shape[1])  # rows for PAD / MASK / UNK stay zero
    table[ids] = vec
    return table


class LogRobust(nn.Module):
    def __init__(self, table, hidden=128):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(table, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(table.shape[1], hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        a = self.att(torch.tanh(h)).squeeze(-1).masked_fill(x == 0, -1e9)
        return self.out((torch.softmax(a, 1).unsqueeze(-1) * h).sum(1))


def pad(seqs, ids):
    X = np.zeros((len(ids), MAX_LEN), dtype=np.int64)
    for j, i in enumerate(ids):
        s = seqs[i][:MAX_LEN]
        X[j, :len(s)] = s
    return torch.from_numpy(X)


@torch.no_grad()
def probs(model, X, bs=512):
    model.eval()
    return torch.cat([torch.softmax(model(X[i].to(DEVICE)), -1)[:, 1].cpu() for i in lbm.batches(len(X), bs)]).numpy()


def main(a):
    t0 = time.time()
    seqs, y, order, _, templates, meta = load_hdfs()
    tr, va, te = split(order)
    table = template_vectors(templates)
    Xtr, Xv, Xt = pad(seqs, tr), pad(seqs, va), pad(seqs, te)
    ytr = torch.from_numpy(y[tr]).long()
    torch.manual_seed(0)
    model = LogRobust(table).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    weight = torch.tensor([1.0, float((ytr == 0).sum()) / max(int(ytr.sum()), 1)], device=DEVICE)  # class balance
    t = time.time()
    for epoch in range(a.epochs):
        model.train(); losses = []
        for idx in lbm.batches(len(Xtr), 128, shuffle=True):
            loss = F.cross_entropy(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"  logrobust epoch {epoch + 1}/{a.epochs} loss {np.mean(losses):.4f}", flush=True)
    train_s = time.time() - t
    pv, pt = probs(model, Xv), probs(model, Xt)
    tau, vf1 = best_tau(pv, y[va])
    r = prf(pt > tau, y[te])
    r.update(tau=float(tau), val_f1=float(vf1), train_seconds=round(train_s, 1), epochs=a.epochs)
    print(f"LogRobust-style    P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}  (tau={tau:.4f}, val F1 {vf1:.3f})")
    out = {"dataset": meta["dataset"], "method": "LogRobust-style (attention Bi-LSTM over BERT template vectors)",
           "supervised": True,
           "settings": {"template_encoder": "bert-base-uncased, mean of last-layer token states, frozen",
                        "lstm": "2-layer bidirectional, hidden 128, attention pooling", "max_len": MAX_LEN,
                        "sessions_truncated": int(sum(len(seqs[i]) > MAX_LEN for i in range(len(seqs)))),
                        "optimizer": "Adam, lr 1e-3, batch 128, class-weighted cross-entropy", "seed": 0},
           "split": {"train": len(tr), "train_anomalous_used": int(y[tr].sum()), "val": len(va), "test": len(te)},
           "protocol": "train on all labelled sessions of the first 60 %; threshold on the anomaly probability "
                       "chosen on the middle 20 %; last 20 % evaluated once",
           "result": r, "total_seconds": round(time.time() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hdfs_extra.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS / 'hdfs_extra.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    main(ap.parse_args())
