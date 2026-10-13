"""Shared settings. Set LOGMIT_DATA to keep bulky data out of the repo (e.g. D:\\logmit-data)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LOGMIT_DATA", ROOT / "data"))
MODEL_DIR = DATA / "model"
RESULTS = Path(os.environ.get("LOGMIT_RESULTS", ROOT / "results"))

NAMESPACE = "default"
# Collected, but never fed to the detector: they describe the users, not the application.
EXCLUDE = {"loadgenerator", "prober"}

WINDOW, STRIDE = 32, 8
PAD, MASK, UNK = 0, 1, 2  # template key = Drain cluster id + 2, so real keys start at 3

# Online Boutique call graph: caller -> callees.
CALLS = {
    "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice",
                 "productcatalogservice", "recommendationservice", "shippingservice"],
    "checkoutservice": ["cartservice", "currencyservice", "emailservice", "paymentservice",
                        "productcatalogservice", "shippingservice"],
    "recommendationservice": ["productcatalogservice"],
    "cartservice": ["redis-cart"],
}
SERVICES = sorted(set(CALLS) | {c for v in CALLS.values() for c in v})


def affected(target):
    """The target plus every service that calls it, directly or transitively."""
    out, frontier = {target}, [target]
    while frontier:
        s = frontier.pop()
        for caller, callees in CALLS.items():
            if s in callees and caller not in out:
                out.add(caller)
                frontier.append(caller)
    return sorted(out)


def run_dir(name):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
