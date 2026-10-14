"""Seed repeats on the testbed data: retrain LogBERT and DeepLog with several seeds, re-tune (g, tau) on
the validation run for each seed with the same procedure as logmit.evaluate, report the test run once
per seed. Writes results/seeds.json.

    python -m logmit.seeds --train normal1 --val fault_val --test fault_test --seeds 0 1 2 3 4
"""
import argparse
import json
import time

import numpy as np

from . import baselines, logbert
from .config import RESULTS
from .evaluate import G_GRID, report, tune
from .windows import build, load_meta, select

FAULTS = ("cpu_throttle", "outage", "pod_kill")


def summary(rows):
    out = {}
    for k in ("precision", "recall", "f1"):
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max())}
    for f in FAULTS:
        v = np.array([r["per_fault"][f] for r in rows])
        out[f"f1_{f}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def main(a):
    meta = load_meta()
    train = build(a.train)
    val, test = (select(w, w["y"] >= 0) for w in (build(a.val), build(a.test)))
    n_svc = int(train["svc"].max()) + 1
    results = {"runs": {"train": a.train, "val": a.val, "test": a.test}, "seeds": a.seeds,
               "procedure": "per seed: train on the normal run, choose (g, tau) maximising window-level F1 on the "
                            "validation run over g in %s, report the test run once" % G_GRID,
               "logbert": [], "deeplog": []}
    for seed in a.seeds:
        t = time.time()
        args = dict(vocab=meta["vocab"], n_svc=n_svc, hidden=256, layers=4, heads=4, use_svc=True)
        model = logbert.train(train, epochs=a.epochs, vhm=0.1, seed=seed, **args)
        logbert.save(model, args, f"logbert_seed{seed}")
        train_s = time.time() - t
        cache = {}

        def lb(w, g, m=model):
            if id(w["X"]) not in cache:
                cache[id(w["X"])] = logbert.ranks(m, w["X"], w["svc"])
            return logbert.score(cache[id(w["X"])], g)

        g, tau, val_f1 = tune(lb, val, G_GRID)
        r = report(lb, g, tau, test)
        r.update(seed=seed, val_f1=val_f1, train_seconds=round(train_s, 1),
                 final_val_mlkp=model.history[-1]["val_mlkp"])
        results["logbert"].append(r)
        print(f"logbert seed {seed}: g={g} tau={tau:.4f} test P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} "
              f"({train_s:.0f}s)", flush=True)

        t = time.time()
        dl = baselines.deeplog(train["X"], meta["vocab"], epochs=5, seed=seed)
        train_s = time.time() - t
        dcache = {}

        def dlf(w, g, f=dl):
            if id(w["X"]) not in dcache:
                dcache[id(w["X"])] = f(w["X"])
            return logbert.score(dcache[id(w["X"])], g)

        g, tau, val_f1 = tune(dlf, val, G_GRID)
        r = report(dlf, g, tau, test)
        r.update(seed=seed, val_f1=val_f1, train_seconds=round(train_s, 1))
        results["deeplog"].append(r)
        print(f"deeplog seed {seed}: g={g} tau={tau:.4f} test P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}",
              flush=True)
        results["summary"] = {"logbert": summary(results["logbert"]), "deeplog": summary(results["deeplog"])}
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "seeds.json").write_text(json.dumps(results, indent=1))  # after every seed, in case of a crash
    print(json.dumps(results["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    main(ap.parse_args())
