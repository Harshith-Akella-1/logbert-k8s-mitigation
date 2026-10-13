"""Stage 4: detection quality. Thresholds are chosen on the validation fault run, then
frozen and reported on the test fault run. Writes results/detection.json.

    python -m logmit.evaluate --train normal1 --val fault_val --test fault_test
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import baselines, logbert
from .config import EXCLUDE, MODEL_DIR, RESULTS, read_jsonl, run_dir
from .windows import build, load_meta, select, split_by_time

G_GRID = [1, 2, 3, 4, 6, 9]


def prf(pred, y):
    tp = int((pred & (y == 1)).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int((y == 1).sum()), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def persist(pred, svc):
    """Two-consecutive-windows rule. Windows are stored grouped by service, in time order."""
    out = pred.copy()
    out[1:] &= pred[:-1] & (svc[1:] == svc[:-1])
    out[0] = False
    return out


def best_tau(scores, y):
    cands = np.unique(scores)
    cands = cands if len(cands) <= 200 else np.quantile(scores, np.linspace(0, 1, 201))
    f1 = [prf(scores > t, y)["f1"] for t in cands]
    return float(cands[int(np.argmax(f1))]), float(max(f1))


def tune(score_fn, val, g_grid=(None,)):
    """score_fn(windows, g) -> scores. Returns the (g, tau) with the best validation F1."""
    best = None
    for g in g_grid:
        tau, f1 = best_tau(score_fn(val, g), val["y"])
        if best is None or f1 > best[2]:
            best = (g, tau, f1)
    return best


def report(score_fn, g, tau, test, use_persist=False):
    pred = score_fn(test, g) > tau
    if use_persist:
        pred = persist(pred, test["svc"])
    out = prf(pred, test["y"])
    out["per_fault"] = {}
    for fault in sorted(set(test["fault"]) - {""}):
        keep = (test["y"] == 0) | (test["fault"] == fault)
        out["per_fault"][fault] = prf(pred[keep], test["y"][keep])["f1"]
    out.update(g=g, tau=tau)
    return out


def main(a):
    meta = load_meta()
    templates = json.loads((MODEL_DIR / "templates.json").read_text())
    train = build(a.train)
    fit, held_out = split_by_time(train)
    val, test = (select(w, w["y"] >= 0) for w in (build(a.val), build(a.test)))
    episodes = {r: read_jsonl(run_dir(r) / "faults.jsonl") for r in (a.val, a.test)}
    keys = pd.read_csv(run_dir(a.train) / "keys.csv")
    monitored = [s for s in meta["services"] if s not in EXCLUDE]
    for name, w in (("val", val), ("test", test)):
        print(f"{name}: {(w['y'] == 0).sum()} normal / {(w['y'] == 1).sum()} anomalous windows")

    cache = {}

    def cached(tag, fn):  # rank computation is the slow part; do it once per window set
        def score_fn(w, g):
            key = (tag, id(w["X"]))
            if key not in cache:
                cache[key] = fn(w)
            return logbert.score(cache[key], g)
        return score_fn

    methods = {"Error-rate rule": (lambda w, g: w["err"], (None,))}
    forest = baselines.isolation_forest(train["X"], meta["vocab"])
    methods["Isolation Forest"] = (lambda w, g: forest(w["X"]), (None,))
    dl = baselines.deeplog(train["X"], meta["vocab"], epochs=a.deeplog_epochs)
    methods["DeepLog"] = (cached("deeplog", lambda w: dl(w["X"])), G_GRID)
    models = {}
    for name in a.logbert:
        model = logbert.load(name)
        models[name] = {**model.args, "params": sum(p.numel() for p in model.parameters())}
        methods[name] = (cached(name, lambda w, m=model: logbert.ranks(m, w["X"], w["svc"])), G_GRID)

    results = {"runs": {"train": a.train, "val": a.val, "test": a.test},
               "dataset": {
                   "services": len(monitored), "templates": len(templates),
                   "normal_lines": int(keys.svc.isin(monitored).sum()),
                   "train_windows": len(fit["X"]), "held_out_windows": len(held_out["X"]),
                   "val_normal": int((val["y"] == 0).sum()), "val_anomalous": int((val["y"] == 1).sum()),
                   "test_normal": int((test["y"] == 0).sum()), "test_anomalous": int((test["y"] == 1).sum()),
                   "val_episodes": len(episodes[a.val]), "test_episodes": len(episodes[a.test]),
                   "episode_median_s": float(np.median([e["end"] - e["start"] for v in episodes.values() for e in v])),
               },
               "models": models, "methods": {}}
    print(f"\n{'method':<22}{'P':>7}{'R':>7}{'F1':>7}   g   tau")
    for name, (fn, grid) in methods.items():
        g, tau, _ = tune(fn, val, grid)
        r = report(fn, g, tau, test)
        if name in a.logbert:
            r["with_persistence"] = report(fn, g, tau, test, use_persist=True)
            r["tau_sweep"] = [{"tau": t, **prf(fn(test, g) > t, test["y"])} for t in np.arange(0.0, 0.55, 0.05).round(2).tolist()]
        results["methods"][name] = r
        print(f"{name:<22}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}   {g}   {tau:.3f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detection.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {RESULTS / 'detection.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--logbert", nargs="+", default=["logbert"], help="saved model names to evaluate")
    ap.add_argument("--deeplog-epochs", type=int, default=5)
    main(ap.parse_args())
