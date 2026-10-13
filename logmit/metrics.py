"""Stage 6: closed-loop metrics for one run, from the prober's lines, faults.jsonl and verdicts.jsonl.
Writes results/<run>.json.

    python -m logmit.metrics --run loop_reactive
    python -m logmit.metrics --run loop_proactive
"""
import argparse
import json

import numpy as np

from .config import RESULTS, read_jsonl, run_dir
from .windows import SETTLE

STABLE = 10.0  # seconds of uninterrupted successful probes that count as recovered
# Kept out of the headline comparison: the reactive baseline cannot undo a scale-to-zero until the
# fault is withdrawn, while the controller's scale-up reverses it directly.
SEPARATE = {"outage"}


def summarise(rows):
    mttr = [r["mttr"] for r in rows if r["mttr"] is not None]
    lead = [r["lead"] for r in rows if r["lead"] is not None]
    failed = [r["failed_pct"] for r in rows if r["failed_pct"] is not None]
    return {
        "episodes": len(rows),
        "episodes_with_user_impact": sum(r["mttr"] is not None or r.get("recovered") is False for r in rows),
        "episodes_not_recovered": sum(r.get("recovered") is False for r in rows),
        "mttr_mean_s": float(np.mean(mttr)) if mttr else None,
        "failed_requests_during_fault_pct": float(np.mean(failed)) if failed else None,
        "lead_time_median_s": float(np.median(lead)) if lead else None,
        "lead_time_iqr_s": [float(x) for x in np.percentile(lead, [25, 75])] if lead else None,
    }


def probes(run):
    """(timestamp, ok) per probe. Prober lines look like: PROBE /cart 200 0.043"""
    t, ok = [], []
    with open(run_dir(run) / "logs.jsonl", encoding="utf-8") as f:
        for row in f:
            if '"prober"' not in row:
                continue
            r = json.loads(row)
            parts = r["line"].split()
            if r["svc"] != "prober" or len(parts) != 4 or parts[0] != "PROBE":
                continue
            t.append(r["ts"])
            ok.append(parts[2][0] in "23")
    order = np.argsort(t)
    return np.array(t)[order], np.array(ok)[order]


def main(run):
    t, ok = probes(run)
    episodes = read_jsonl(run_dir(run) / "faults.jsonl")
    verdicts = read_jsonl(run_dir(run) / "verdicts.jsonl")
    # a dry-run loop logs what it would have done; those are verdicts, not actions
    acted = [v for v in verdicts if not v.get("dry_run") and not v["action"].startswith(("skip", "alert"))]
    rows = []
    for e in episodes:
        m = (t >= e["start"]) & (t <= e["end"] + SETTLE)
        te, oke = t[m], ok[m]
        row = {"fault": e["fault"], "target": e["target"], "failed_pct": 100 * float((~oke).mean()) if len(oke) else None,
               "mttr": None, "lead": None}
        fails = te[~oke]
        if len(fails):
            first = fails[0]
            # recovered at the first successful probe that starts a STABLE-second failure-free stretch
            good = [x for x in te[oke] if x > first and not ((fails > x) & (fails <= x + STABLE)).any()]
            row["mttr"] = float(good[0] - first) if good else None
            row["recovered"] = bool(good)
            seen = [v["log_ts"] for v in verdicts if v["svc"] in e["affected"] and e["start"] <= v["log_ts"] <= e["end"]]
            if seen:
                row["lead"] = float(first - min(seen))
        rows.append(row)

    in_fault = lambda x: any(e["start"] <= x <= e["end"] + SETTLE for e in episodes)
    hours = (t[-1] - t[0]) / 3600 if len(t) > 1 else 0
    # availability for the main comparison ignores probes sent during the separately reported faults
    aside = np.zeros(len(t), dtype=bool)
    for e in episodes:
        if e["fault"] in SEPARATE:
            aside |= (t >= e["start"]) & (t <= e["end"] + SETTLE)
    main = summarise([r for r in rows if r["fault"] not in SEPARATE])
    main["availability_pct"] = 100 * float(ok[~aside].mean()) if (~aside).any() else None
    out = {
        "run": run, "probes": len(t), "hours": hours,
        "availability_pct": 100 * float(ok.mean()) if len(ok) else None,
        "actions": len(acted),
        "unnecessary_actions_per_hour": sum(not in_fault(v["log_ts"]) for v in acted) / hours if hours else None,
        "main": main,
        "per_fault": {f: summarise([r for r in rows if r["fault"] == f]) for f in sorted({r["fault"] for r in rows})},
        "per_episode": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{run}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_episode"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(ap.parse_args().run)
