"""Collects the raw facts the paper needs into results/paper_inputs/*.json.
Uses the thresholds already frozen in results/detection.json; nothing here is tuned.

    python -m logmit.report --train normal1 --val fault_val --test fault_test
"""
import argparse
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from . import logbert
from .config import EXCLUDE, MODEL_DIR, RESULTS, STRIDE, WINDOW, read_jsonl, run_dir
from .evaluate import persist, prf
from .windows import SETTLE, build, split_by_time

OUT = RESULTS / "paper_inputs"


def stamp(t):
    return datetime.fromtimestamp(float(t)).isoformat(timespec="seconds")


def run_facts(run):
    df = pd.read_csv(run_dir(run) / "keys.csv")
    mon = df[~df.svc.isin(EXCLUDE)]
    w = build(run)
    return {
        "run": run, "start": stamp(df.ts.min()), "end": stamp(df.ts.max()),
        "duration_min": round(float(df.ts.max() - df.ts.min()) / 60, 2),
        "lines_all_pods": len(df), "lines_monitored_services": len(mon),
        "lines_per_service": mon.groupby("svc").size().sort_values(ascending=False).to_dict(),
        "unknown_key_lines": int((mon.key == 2).sum()),
        "windows": {"total": len(w["y"]), "normal": int((w["y"] == 0).sum()),
                    "anomalous": int((w["y"] == 1).sum()), "excluded_settling": int((w["y"] == -1).sum()),
                    "anomalous_by_fault": {f: int((w["fault"] == f).sum()) for f in sorted(set(w["fault"]) - {""})}},
        "episodes": [{"n": i + 1, "fault": e["fault"], "target": e["target"], "affected": e["affected"],
                      "start": stamp(e["start"]), "end": stamp(e["end"]), "duration_s": round(e["end"] - e["start"], 1)}
                     for i, e in enumerate(read_jsonl(run_dir(run) / "faults.jsonl"))],
    }, w


def probes(run):
    t, ok = [], []
    with open(run_dir(run) / "logs.jsonl", encoding="utf-8") as f:
        for row in f:
            if '"prober"' in row:
                r = json.loads(row)
                p = r["line"].split()
                if r["svc"] == "prober" and len(p) == 4 and p[0] == "PROBE":
                    t.append(r["ts"]); ok.append(p[2][0] in "23")
    return np.array(t), np.array(ok)


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    det = json.loads((RESULTS / "detection.json").read_text())
    shutil.copy(RESULTS / "detection.json", OUT / "detection_test.json")
    for extra in ("cluster.json",):
        if (RESULTS / extra).exists():
            shutil.copy(RESULTS / extra, OUT / extra)
    shutil.copy(MODEL_DIR / "templates.json", OUT / "templates.json")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    services = [s for s in meta["services"] if s not in EXCLUDE]

    # ---- data
    facts = {}
    (facts["normal"], wn), (facts["validation"], wv), (facts["test"], wt) = (run_facts(r) for r in (a.train, a.val, a.test))
    fit, held = split_by_time(wn)
    cut = float(np.quantile(wn["t0"], 0.8))
    facts["normal"]["split"] = {
        "rule": "chronological: windows ending before the 80th percentile of window start times train the models; "
                "windows starting at or after it are the held-out normal set; windows straddling the cut are dropped",
        "train": {"from": stamp(fit["t0"].min()), "to": stamp(fit["t1"].max()), "windows": len(fit["y"])},
        "held_out": {"from": stamp(held["t0"].min()), "to": stamp(held["t1"].max()), "windows": len(held["y"])},
        "cut": stamp(cut), "dropped_straddling": len(wn["y"]) - len(fit["y"]) - len(held["y"]),
    }
    facts["window"] = {"size_keys": WINDOW, "stride_keys": STRIDE, "per": "service"}
    facts["monitored_services"] = services
    facts["templates"] = len(json.loads((MODEL_DIR / "templates.json").read_text()))
    facts["labelling"] = {
        "anomalous": "window [t0, t1] of service S is anomalous if t0 <= episode.end and t1 >= episode.start for an "
                     "episode whose affected set contains S",
        "affected_set": "the fault target plus every service that calls it, directly or transitively, in the static "
                        "Online Boutique call graph (logmit/config.py CALLS)",
        "excluded": f"not anomalous, but overlapping (episode.end, episode.end + {SETTLE:.0f} s] of an episode whose "
                    "affected set contains S: dropped from tuning and scoring",
        "normal": "everything else, including windows of unaffected services during a fault",
        "timestamps": "window times are kubelet log timestamps; episode times are the injector's host clock "
                      "(measured node-host difference: 0 s)",
    }
    (OUT / "data.json").write_text(json.dumps(facts, indent=1))

    # ---- training curves
    curves = {p.name.replace(".history.json", ""): json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*.history.json"))}
    (OUT / "training_curves.json").write_text(json.dumps(curves, indent=1))

    # ---- LogBERT: validation sweep and episode-level detection on the test run
    g, tau = det["methods"]["logbert"]["g"], det["methods"]["logbert"]["tau"]
    model = logbert.load("logbert")
    score = {name: logbert.score(logbert.ranks(model, w["X"], w["svc"]), g) for name, w in (("val", wv), ("test", wt))}

    keep = wv["y"] >= 0
    sweep = [{"tau": k / WINDOW, **prf(score["val"][keep] > k / WINDOW, wv["y"][keep])} for k in range(0, 17)]
    (OUT / "validation_threshold_sweep.json").write_text(json.dumps(
        {"method": "logbert", "g": g, "chosen_tau": tau, "note": "score = misses / 32, so tau steps of 1/32; "
         "prediction is score > tau; validation run only", "sweep": sweep}, indent=1))

    pred = score["test"] > tau
    verdict = persist(pred, wt["svc"])  # second of two consecutive anomalous windows of one service
    pt, pok = probes(a.test)
    rows = []
    for i, e in enumerate(read_jsonl(run_dir(a.test) / "faults.jsonl")):
        aff = np.isin(wt["svc"], [services.index(s) for s in e["affected"] if s in services])
        inside = aff & (wt["t1"] >= e["start"]) & (wt["t0"] <= e["end"])

        def first(mask):
            return (float(wt["t1"][mask].min()), services[int(wt["svc"][mask][np.argmin(wt["t1"][mask])])]) if mask.any() else (None, None)

        t_any, s_any = first(inside & pred)
        t_ver, s_ver = first(inside & verdict)
        fails = pt[(~pok) & (pt >= e["start"]) & (pt <= e["end"] + SETTLE)]
        m = (pt >= e["start"]) & (pt <= e["end"])
        first_fail = float(fails[0]) if len(fails) else None
        rows.append({
            "n": i + 1, "fault": e["fault"], "target": e["target"], "start": stamp(e["start"]), "end": stamp(e["end"]),
            "windows_in_episode": int(inside.sum()), "windows_flagged": int((inside & pred).sum()),
            "detected_any_window": t_any is not None, "first_window_lag_s": None if t_any is None else round(t_any - e["start"], 2),
            "first_window_service": s_any,
            "detected_verdict": t_ver is not None, "verdict_lag_s": None if t_ver is None else round(t_ver - e["start"], 2),
            "verdict_service": s_ver,
            "first_failed_probe_lag_s": None if first_fail is None else round(first_fail - e["start"], 2),
            "failed_probe_pct_during_fault": round(100 * float((~pok[m]).mean()), 2) if m.any() else None,
            "verdict_lead_over_first_failed_probe_s": None if (t_ver is None or first_fail is None) else round(first_fail - t_ver, 2),
        })
    normal = wt["y"] == 0
    hours = float(wt["t1"].max() - wt["t0"].min()) / 3600
    summary = {
        "method": "logbert", "g": g, "tau": tau, "run": a.test,
        "definitions": {
            "lag": "window end time minus fault start (offline; the live loop adds up to 2 s of collector polling)",
            "detected_any_window": "at least one window of an affected service overlapping the episode has score > tau",
            "detected_verdict": "same, but the window and the previous window of that service are both anomalous",
            "first_failed_probe": "first prober request with a non-2xx/3xx status or a 3 s timeout after fault start",
        },
        "episodes_detected_verdict": sum(r["detected_verdict"] for r in rows), "episodes": len(rows),
        "by_fault": {f: {"episodes": sum(r["fault"] == f for r in rows),
                         "detected_verdict": sum(r["detected_verdict"] for r in rows if r["fault"] == f),
                         "median_verdict_lag_s": float(np.median([r["verdict_lag_s"] for r in rows if r["fault"] == f and r["verdict_lag_s"] is not None] or [np.nan]))}
                     for f in sorted({r["fault"] for r in rows})},
        "false_verdict_windows_on_normal": int((verdict & normal).sum()), "normal_windows": int(normal.sum()),
        "run_hours": round(hours, 3), "per_episode": rows,
    }
    (OUT / "episode_detection_test.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_episode", "definitions")}, indent=1))
    for r in rows:
        print(f"{r['n']:>2} {r['fault']:<12} {r['target']:<22} verdict {str(r['verdict_lag_s']):>7}s ({r['verdict_service']})  "
              f"first failed probe {str(r['first_failed_probe_lag_s']):>7}s  failed {r['failed_probe_pct_during_fault']}%")
    print("\nvalidation sweep:", [(round(s["tau"], 3), round(s["precision"], 3), round(s["recall"], 3), round(s["f1"], 3)) for s in sweep[:8]])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    main(ap.parse_args())
