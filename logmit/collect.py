"""Stage 2a: poll every pod's logs through the Kubernetes API and append them to <run>/logs.jsonl.

    python -m logmit.collect --run normal1            # Ctrl+C to stop
    python -m logmit.collect --run normal1 --minutes 60
"""
import argparse
import json
import sys
import time
from datetime import datetime

from kubernetes import client, config

from .config import NAMESPACE, run_dir

TIMEOUT = 10    # seconds per API call
GIVE_UP = 60.0  # stop if the cluster stays unreachable this long


def to_epoch(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def collect(run, minutes=None, interval=2.0):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    out = run_dir(run) / "logs.jsonl"
    last = {}  # (pod, container) -> last kubelet timestamp string written
    present = set()  # containers already running when collection started
    started = time.time()
    first_pass, deadline = True, time.time() + minutes * 60 if minutes else None
    n, down_since = 0, None
    print(f"collecting into {out}")
    with open(out, "a", encoding="utf-8") as f:
        while deadline is None or time.time() < deadline:
            tick = time.time()
            try:
                pods = v1.list_namespaced_pod(NAMESPACE, _request_timeout=TIMEOUT).items
                down_since = None
            except Exception as e:  # API server unreachable: say so and stop rather than hang silently
                down_since = down_since or tick
                print(f"\ncluster unreachable for {tick - down_since:.0f}s: {type(e).__name__}", file=sys.stderr, flush=True)
                if tick - down_since > GIVE_UP:
                    raise SystemExit(f"aborting: no cluster for {GIVE_UP:.0f}s; {n} lines collected")
                time.sleep(interval)
                continue
            for pod in pods:
                name = pod.metadata.name
                svc = (pod.metadata.labels or {}).get("app", name)
                for c in pod.spec.containers:
                    key = (name, c.name)
                    kwargs = {}
                    if first_pass:
                        present.add(key)
                    if key in last:
                        kwargs["since_seconds"] = int(time.time() - to_epoch(last[key])) + 2
                    elif key in present:
                        # running before this run started: skip its earlier history
                        kwargs["since_seconds"] = int(time.time() - started) + 1
                    try:
                        text = v1.read_namespaced_pod_log(name, NAMESPACE, container=c.name, timestamps=True,
                                                          _request_timeout=TIMEOUT, **kwargs)
                    except Exception:
                        continue  # container not started yet, pod already gone, or a timeout
                    for row in text.splitlines():
                        ts, _, line = row.partition(" ")
                        # kubelet timestamps are fixed-width, so string order is time order
                        if not line or ts <= last.get(key, ""):
                            continue
                        try:
                            t = to_epoch(ts)
                        except ValueError:
                            continue
                        last[key] = ts
                        f.write(json.dumps({"ts": t, "svc": svc, "pod": name, "line": line}) + "\n")
                        n += 1
            first_pass = False
            f.flush()
            print(f"\r{n} lines", end="", flush=True)
            time.sleep(max(0.0, interval - (time.time() - tick)))
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()
    try:
        collect(a.run, a.minutes, a.interval)
    except KeyboardInterrupt:
        print("\nstopped")
