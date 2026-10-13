"""Stage 4 baselines: Isolation Forest on template counts and DeepLog (LSTM). The third baseline,
the error-rate rule, is just the windows' `err` column. Higher score = more anomalous."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from .logbert import DEVICE, batches

HISTORY = 10  # DeepLog input length


def counts(X, vocab):
    return np.stack([np.bincount(row, minlength=vocab) for row in X])


def isolation_forest(X_train, vocab, seed=0):
    model = IsolationForest(n_estimators=100, random_state=seed).fit(counts(X_train, vocab))
    return lambda X: -model.score_samples(counts(X, vocab))


class DeepLog(nn.Module):
    def __init__(self, vocab, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.out(self.lstm(self.emb(x))[0][:, -1])


def _pairs(X):
    seq = torch.as_tensor(X).unfold(1, HISTORY + 1, 1)  # [N, WINDOW - HISTORY, HISTORY + 1]
    return seq[..., :HISTORY], seq[..., HISTORY]


def deeplog(X_train, vocab, epochs=5, bs=1024, seed=0):
    torch.manual_seed(seed)
    model = DeepLog(vocab).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, nxt = _pairs(X_train)
    hist, nxt = hist.reshape(-1, HISTORY), nxt.reshape(-1)
    for epoch in range(epochs):
        model.train()
        for idx in batches(len(hist), bs, shuffle=True):
            loss = F.cross_entropy(model(hist[idx].to(DEVICE)), nxt[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"deeplog epoch {epoch + 1}/{epochs}  loss {loss.item():.4f}")

    @torch.no_grad()
    def ranks(X):
        model.eval()
        h, n = _pairs(X)
        shape = n.shape
        h, n = h.reshape(-1, HISTORY), n.reshape(-1)
        out = torch.zeros(len(n), dtype=torch.int32)
        for idx in batches(len(n), 4096):
            logits = model(h[idx].to(DEVICE))
            true = logits.gather(-1, n[idx, None].to(DEVICE))
            out[idx] = (logits > true).sum(-1).int().cpu()
        return out.reshape(shape).numpy()

    return ranks
