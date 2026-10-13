"""Stage 3b: per-service sliding windows over the key stream, labelled from the fault log."""
import json

import numpy as np
import pandas as pd

from .config import EXCLUDE, MODEL_DIR, STRIDE, WINDOW, read_jsonl, run_dir

SETTLE = 60.0  # seconds after a fault ends that are neither clean nor faulty; dropped from scoring


def load_meta():
    return json.loads((MODEL_DIR / "meta.json").read_text())


def build(run):
    """Returns arrays: X [N, WINDOW] keys, err [N] share of lines with an error keyword, svc [N] service index, t0/t1 [N] window time span,
    y [N] (1 faulty, 0 normal, -1 settling: ignore), fault [N] fault name or ''."""
    services = [s for s in load_meta()["services"] if s not in EXCLUDE]
    index = {s: i for i, s in enumerate(services)}
    df = pd.read_csv(run_dir(run) / "keys.csv")
    df = df[df.svc.isin(index)].sort_values("ts", kind="stable")
    episodes = read_jsonl(run_dir(run) / "faults.jsonl")

    X, E, S, T0, T1, Y, F = [], [], [], [], [], [], []
    for name, g in df.groupby("svc"):
        k, t, err = g.key.to_numpy(), g.ts.to_numpy(), g.err.to_numpy()
        for i in range(0, len(k) - WINDOW + 1, STRIDE):
            t0, t1 = t[i], t[i + WINDOW - 1]
            y, fault = 0, ""
            for e in episodes:
                if name not in e["affected"]:
                    continue
                if t0 <= e["end"] and t1 >= e["start"]:
                    y, fault = 1, e["fault"]
                    break
                if t0 <= e["end"] + SETTLE and t1 > e["end"]:
                    y = -1
            X.append(k[i:i + WINDOW]); E.append(err[i:i + WINDOW].mean()); S.append(index[name])
            T0.append(t0); T1.append(t1); Y.append(y); F.append(fault)
    return {"X": np.array(X, dtype=np.int64).reshape(-1, WINDOW), "err": np.array(E), "svc": np.array(S, dtype=np.int64),
            "t0": np.array(T0), "t1": np.array(T1), "y": np.array(Y), "fault": np.array(F, dtype=object)}


def select(w, mask):
    return {k: v[mask] for k, v in w.items()}


def split_by_time(w, frac=0.8):
    """Chronological split. A random split would leak, because neighbouring windows overlap."""
    cut = np.quantile(w["t0"], frac)
    return select(w, w["t1"] < cut), select(w, w["t0"] >= cut)
