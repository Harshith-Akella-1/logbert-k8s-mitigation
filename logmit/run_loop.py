"""Stage 5/6: the online loop. Follows the collector's logs.jsonl, scores each new window with
LogBERT and passes verdicts to the controller. Writes <run>/verdicts.jsonl.

    python -m logmit.run_loop --run loop_proactive
    python -m logmit.run_loop --run loop_proactive --dry-run     # detect and log, never act
"""
import argparse
import json
import time
from collections import defaultdict, deque

import numpy as np

from . import logbert
from .config import EXCLUDE, RESULTS, STRIDE, WINDOW, run_dir
from .controller import Controller
from .parse import encode, make_miner
from .windows import load_meta

PERSISTENCE = 2  # consecutive anomalous windows before a verdict


def follow(path):
    while not path.exists():
        time.sleep(0.5)
    with open(path, "rb") as f:
        f.seek(0, 2)
        buf = b""
        while True:
            chunk = f.readline()
            if not chunk:
                time.sleep(0.2)
                continue
            buf += chunk
            if buf.endswith(b"\n"):  # otherwise the collector is mid-write; keep reading
                yield json.loads(buf)
                buf = b""


def main(a):
    services = [s for s in load_meta()["services"] if s not in EXCLUDE]
    index = {s: i for i, s in enumerate(services)}
    tuned = json.loads((RESULTS / "detection.json").read_text())["methods"][a.model]
    g, tau = a.g or tuned["g"], a.tau if a.tau is not None else tuned["tau"]
    model, miner = logbert.load(a.model), make_miner()
    d = run_dir(a.run)
    ctl = Controller(d / "verdicts.jsonl", dry_run=a.dry_run)
    keys = defaultdict(lambda: deque(maxlen=WINDOW))
    lines = defaultdict(lambda: deque(maxlen=WINDOW))
    fresh, streak = defaultdict(int), defaultdict(int)
    print(f"watching {d / 'logs.jsonl'}  (g={g}, tau={tau:.3f}, dry_run={a.dry_run})")
    for r in follow(d / "logs.jsonl"):
        svc = r["svc"]
        if svc not in index:
            continue
        keys[svc].append(encode(miner, svc, r["line"]))
        lines[svc].append(r["line"])
        fresh[svc] += 1
        if len(keys[svc]) < WINDOW or fresh[svc] < STRIDE:
            continue
        fresh[svc] = 0
        rank = logbert.ranks(model, np.array([keys[svc]]), np.array([index[svc]]))
        score = float(logbert.score(rank, g)[0])
        streak[svc] = streak[svc] + 1 if score > tau else 0
        if streak[svc] >= PERSISTENCE:
            action = ctl.handle(svc, list(lines[svc]), streak[svc], score, r["ts"])
            print(f"{time.strftime('%H:%M:%S')} {svc} score={score:.2f} streak={streak[svc]} -> {action}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="logbert")
    ap.add_argument("--g", type=int)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--dry-run", action="store_true")
    try:
        main(ap.parse_args())
    except KeyboardInterrupt:
        print("\nstopped")
