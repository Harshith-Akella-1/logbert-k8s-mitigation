"""Public-benchmark evaluation (Loghub HDFS_v1 and BGL) with the same protocol as the testbed:
chronological 60/20/20 split, detectors trained on normal sequences of the first 60 % only, every
threshold chosen on the middle 20 %, the last 20 % evaluated once. Sequence-level precision/recall/F1.

    python -m logmit.bench --dataset hdfs --epochs 10
    python -m logmit.bench --dataset bgl --epochs 10

A sequence is an HDFS block session or a BGL window of consecutive lines. LogBERT sees a sequence as
windows of 32 keys (padded); a sequence's score is the highest score of its windows.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from . import logbert as lbm
from .baselines import DeepLog, HISTORY
from .config import DATA, MASK, PAD, RESULTS, UNK, WINDOW
from .evaluate import G_GRID, best_tau, prf
from .parse import ERROR_WORDS

DEVICE = lbm.DEVICE
BENCH = DATA / "benchmarks"


# ---------------------------------------------------------------- datasets
def load_hdfs():
    """Block sessions from Loghub's preprocessed event traces; chronological order from HDFS.log."""
    root = BENCH / "HDFS_v1"
    pre = next(root.rglob("Event_traces.csv")).parent
    templates = {}
    with open(pre / "HDFS.log_templates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            templates[int(row["EventId"][1:]) + 2] = row["EventTemplate"]
    first = {}  # block id -> position of its first line in HDFS.log (the log is in time order)
    blk = re.compile(rb"blk_-?\d+")
    with open(next(root.rglob("HDFS.log")), "rb") as f:
        for i, line in enumerate(f):
            for b in set(blk.findall(line)):
                first.setdefault(b.decode(), i)
    seqs, labels, order = [], [], []
    with open(pre / "Event_traces.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys = [int(e.strip()[1:]) + 2 for e in row["Features"].strip("[]").split(",") if e.strip()]
            seqs.append(np.array(keys, dtype=np.int64))
            labels.append(1 if row["Label"].strip().lower() in ("fail", "anomaly", "1") else 0)
            order.append(first[row["BlockId"]])
    bad = [k for k, t in templates.items() if ERROR_WORDS.search(t)]
    err = [float(np.isin(s, bad).mean()) for s in seqs]
    meta = {"dataset": "Loghub HDFS_v1", "sequence": "block session (block_id)",
            "order": "first line of the block in HDFS.log", "parser": "Loghub-provided templates (Drain), not re-parsed",
            "keyword_rule_input": "template text of each event (raw lines are not joined to sessions)"}
    return seqs, np.array(labels), np.array(order, dtype=np.float64), np.array(err), templates, meta


def load_bgl(lines_per_window=32, limit=None):
    """Consecutive-line windows over BGL.log (time order), parsed with Drain fitted on the training 60 %."""
    from .parse import make_miner
    root = BENCH / "BGL"
    raw, lab = [], []
    with open(next(root.rglob("BGL.log")), encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.rstrip("\n").split(maxsplit=9)
            if len(parts) < 10:
                parts += [""] * (10 - len(parts))
            lab.append(parts[0] != "-")
            raw.append(parts[9])
    n = len(raw) // lines_per_window * lines_per_window
    cut = int(n * 0.6) // lines_per_window * lines_per_window
    miner = make_miner(fit=True, state=BENCH / "BGL" / "drain.bin")
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        text = "bgl " + raw[i]
        if i < cut:
            keys[i] = miner.add_log_message(text)["cluster_id"] + 2
        else:
            # drain3's match() demands a perfect template match; this is Drain's normal tree search at the
            # training similarity threshold, without creating or updating templates
            tokens = miner.drain.get_content_as_tokens(miner.masker.mask(text))
            c = miner.drain.tree_search(miner.drain.root_node, tokens, miner.drain.sim_th, True)
            keys[i] = c.cluster_id + 2 if c else UNK
    templates = {c.cluster_id + 2: c.get_template() for c in miner.drain.clusters}
    errs = np.array([bool(ERROR_WORDS.search(raw[i])) for i in range(n)])
    lab = np.array(lab[:n])
    w = lines_per_window
    seqs = [keys[i:i + w] for i in range(0, n, w)]
    labels = lab.reshape(-1, w).any(1).astype(int)
    err = errs.reshape(-1, w).mean(1)
    meta = {"dataset": "Loghub BGL", "sequence": f"{w} consecutive lines, non-overlapping (stride {w})",
            "order": "line order of BGL.log", "lines_used": n, "lines_in_file_or_limit": len(raw),
            "parser": "drain3 (similarity 0.5, depth 4, same masks as the testbed) fitted on the first 60 % of "
                      "lines only; later lines are assigned by Drain's tree search at similarity 0.5 without "
                      "learning, and lines matching no template map to the unknown key",
            "label": "a window is anomalous if any of its lines has a non '-' label",
            "keyword_rule_input": "raw message text of each line",
            "unknown_key_share_after_cut": float((keys[cut:] == UNK).mean())}
    return seqs, labels, np.arange(len(seqs), dtype=np.float64), err, templates, meta


# ---------------------------------------------------------------- split
def split(order):
    idx = np.argsort(order, kind="stable")
    n = len(idx)
    return idx[: int(n * 0.6)], idx[int(n * 0.6): int(n * 0.8)], idx[int(n * 0.8):]


# ---------------------------------------------------------------- LogBERT with padding
def to_windows(seqs, ids):
    """Each sequence -> ceil(len/32) windows of 32 keys, PAD-filled. Returns windows and owner index."""
    X, owner = [], []
    for j, i in enumerate(ids):
        s = seqs[i]
        for k in range(0, max(len(s), 1), WINDOW):
            w = np.full(WINDOW, PAD, dtype=np.int64)
            part = s[k:k + WINDOW]
            w[:len(part)] = part
            X.append(w); owner.append(j)
    return np.stack(X), np.array(owner)


class PadLogBERT(lbm.LogBERT):
    def forward(self, x, svc=None):
        pad = x == PAD
        h = self.key(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        h = self.enc(h, src_key_padding_mask=pad)
        real = (~pad).unsqueeze(-1).float()
        return self.head(h), (h * real).sum(1) / real.sum(1).clamp(min=1)


def train_logbert(X, vocab, epochs, vhm=0.1, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = PadLogBERT(vocab, 1, use_svc=False).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    X = torch.from_numpy(X)
    history = []
    for epoch in range(epochs):
        t = time.time()
        if vhm:
            model.eval()
            with torch.no_grad():
                tot = torch.zeros_like(model.center)
                for idx in lbm.batches(len(X), 2048):
                    tot += model(X[idx].to(DEVICE))[1].sum(0)
            model.center.copy_(tot / len(X))
        model.train()
        losses = []
        for idx in lbm.batches(len(X), bs, shuffle=True):
            x = X[idx].to(DEVICE)
            real = x != PAD
            mask = (torch.rand(x.shape, device=DEVICE) < 0.15) & real
            first = real.float().argmax(1)
            none = ~mask.any(1)
            mask[none, first[none]] = True
            logits, rep = model(x.masked_fill(mask, MASK))
            loss = F.cross_entropy(logits[mask], x[mask]) + vhm * ((rep - model.center) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t, 1)})
        print(f"  logbert epoch {epoch + 1}/{epochs} loss {np.mean(losses):.4f} ({time.time() - t:.0f}s)", flush=True)
    return model.eval(), history


@torch.no_grad()
def logbert_ranks(model, X, bs=1024):
    X = torch.from_numpy(X)
    out = torch.zeros(X.shape, dtype=torch.int32)
    pos = torch.arange(WINDOW)
    for idx in lbm.batches(len(X), bs):
        x = X[idx].to(DEVICE)
        for f in range(lbm.FOLDS):
            m = (pos % lbm.FOLDS == f).to(DEVICE)
            logits = model(x.masked_fill(m[None, :] & (x != PAD), MASK))[0][:, m]
            true = logits.gather(-1, x[:, m, None])
            out[idx[:, None], pos[m.cpu()][None, :]] = (logits > true).sum(-1).int().cpu()
    return out.numpy()


def seq_scores_logbert(ranks, X, owner, n, g):
    real = X != PAD
    win = ((ranks >= g) & real).sum(1) / np.maximum(real.sum(1), 1)
    out = np.zeros(n)
    np.maximum.at(out, owner, win)
    return out


# ---------------------------------------------------------------- DeepLog with padding
def flat(seqs, ids):
    """Concatenate sequences, each preceded by HISTORY pads; return flat keys, target positions, owner."""
    parts, targets, owner, off = [], [], [], 0
    for j, i in enumerate(ids):
        s = seqs[i]
        parts.append(np.full(HISTORY, PAD, dtype=np.int64)); parts.append(s)
        targets.append(np.arange(off + HISTORY, off + HISTORY + len(s))); owner.append(np.full(len(s), j))
        off += HISTORY + len(s)
    return np.concatenate(parts), np.concatenate(targets), np.concatenate(owner)


def train_deeplog(seqs, ids, vocab, epochs=5, bs=4096, seed=0):
    torch.manual_seed(seed)
    keys, targets, _ = flat(seqs, ids)
    keys_t, hist = torch.from_numpy(keys), torch.arange(-HISTORY, 0)
    model = DeepLog(vocab).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = torch.from_numpy(targets)
    for epoch in range(epochs):
        t = time.time(); model.train(); losses = []
        for idx in lbm.batches(len(T), bs, shuffle=True):
            tg = T[idx]
            h = keys_t[tg[:, None] + hist].to(DEVICE)
            loss = F.cross_entropy(model(h), keys_t[tg].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"  deeplog epoch {epoch + 1}/{epochs} loss {np.mean(losses):.4f} ({time.time() - t:.0f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def deeplog_ranks(model, seqs, ids):
    keys, targets, owner = flat(seqs, ids)
    keys_t, hist, T = torch.from_numpy(keys), torch.arange(-HISTORY, 0), torch.from_numpy(targets)
    out = np.zeros(len(T), dtype=np.int32)
    for idx in lbm.batches(len(T), 16384):
        tg = T[idx]
        logits = model(keys_t[tg[:, None] + hist].to(DEVICE))
        true = logits.gather(-1, keys_t[tg].to(DEVICE)[:, None])
        out[idx.numpy()] = (logits > true).sum(-1).int().cpu().numpy()
    return out, owner


def seq_scores_deeplog(ranks, owner, n, g):
    miss = np.bincount(owner, weights=(ranks >= g).astype(float), minlength=n)
    return miss / np.maximum(np.bincount(owner, minlength=n), 1)


# ---------------------------------------------------------------- run
def counts(seqs, ids, vocab):
    return np.stack([np.bincount(seqs[i], minlength=vocab)[:vocab] for i in ids]).astype(np.float32)


def evaluate(name, val_fn, test_fn, yv, yt, grid=(None,)):
    best = None
    for g in grid:
        tau, f1 = best_tau(val_fn(g), yv)
        if best is None or f1 > best[2]:
            best = (g, tau, f1)
    g, tau, vf1 = best
    r = prf(test_fn(g) > tau, yt)
    r.update(g=g, tau=float(tau), val_f1=float(vf1))
    print(f"{name:<18} P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}  (g={g}, tau={tau:.4f}, val F1 {vf1:.3f})", flush=True)
    return r


def main(a):
    t0 = time.time()
    if a.dataset == "hdfs":
        seqs, y, order, err, templates, meta = load_hdfs()
    else:
        seqs, y, order, err, templates, meta = load_bgl(limit=a.limit)
    load_s = time.time() - t0
    vocab = max(max(templates), max(int(s.max()) for s in seqs if len(s))) + 1
    tr, va, te = split(order)
    tr_n = tr[y[tr] == 0]
    yv, yt = y[va], y[te]
    meta.update(sequences=len(seqs), anomalous=int(y.sum()), templates=len(templates), vocab=vocab,
                split={"train_all": len(tr), "train_normal_used": len(tr_n), "train_anomalous_dropped": int(y[tr].sum()),
                       "val": len(va), "val_anomalous": int(yv.sum()), "test": len(te), "test_anomalous": int(yt.sum())},
                load_seconds=round(load_s, 1))
    print(json.dumps(meta, indent=1), flush=True)
    res = {"meta": meta, "protocol": {
        "split": "chronological 60/20/20 by sequence start", "train": "normal sequences of the first 60 % only",
        "thresholds": f"chosen on the middle 20 % by maximising sequence-level F1 (g from {G_GRID} for LogBERT and DeepLog)",
        "test": "last 20 %, evaluated once"}, "methods": {}}

    def save():
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{a.dataset}.json").write_text(json.dumps(res, indent=1))

    # keyword rule
    res["methods"]["Keyword rule"] = evaluate("Keyword rule", lambda g: err[va], lambda g: err[te], yv, yt)
    res["methods"]["Keyword rule"]["train_seconds"] = 0.0
    save()
    # Isolation Forest
    t = time.time()
    forest = IsolationForest(n_estimators=100, random_state=0).fit(counts(seqs, tr_n, vocab))
    fs = time.time() - t
    sv, st = -forest.score_samples(counts(seqs, va, vocab)), -forest.score_samples(counts(seqs, te, vocab))
    res["methods"]["Isolation Forest"] = evaluate("Isolation Forest", lambda g: sv, lambda g: st, yv, yt)
    res["methods"]["Isolation Forest"]["train_seconds"] = round(fs, 1)
    save()
    # DeepLog
    t = time.time()
    dl = train_deeplog(seqs, tr_n, vocab, epochs=a.deeplog_epochs)
    ds = time.time() - t
    (rv, ov), (rt, ot) = deeplog_ranks(dl, seqs, va), deeplog_ranks(dl, seqs, te)
    res["methods"]["DeepLog"] = evaluate("DeepLog", lambda g: seq_scores_deeplog(rv, ov, len(va), g),
                                         lambda g: seq_scores_deeplog(rt, ot, len(te), g), yv, yt, G_GRID)
    res["methods"]["DeepLog"].update(train_seconds=round(ds, 1), epochs=a.deeplog_epochs)
    save()
    # LogBERT
    Xtr, _ = to_windows(seqs, tr_n)
    t = time.time()
    model, hist = train_logbert(Xtr, vocab, a.epochs)
    ls = time.time() - t
    Xv, owv = to_windows(seqs, va)
    Xt, owt = to_windows(seqs, te)
    rkv, rkt = logbert_ranks(model, Xv), logbert_ranks(model, Xt)
    res["methods"]["LogBERT"] = evaluate("LogBERT", lambda g: seq_scores_logbert(rkv, Xv, owv, len(va), g),
                                         lambda g: seq_scores_logbert(rkt, Xt, owt, len(te), g), yv, yt, G_GRID)
    res["methods"]["LogBERT"].update(train_seconds=round(ls, 1), epochs=a.epochs, train_windows=len(Xtr),
                                     params=sum(p.numel() for p in model.parameters()), history=hist)
    res["total_seconds"] = round(time.time() - t0, 1)
    save()
    print(f"wrote {RESULTS / (a.dataset + '.json')} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hdfs", "bgl"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--deeplog-epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, help="BGL: use only the first N lines")
    main(ap.parse_args())
