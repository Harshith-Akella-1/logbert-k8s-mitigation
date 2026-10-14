"""Stage 2b: turn raw log lines into template keys with Drain.

    python -m logmit.parse --run normal1 --fit   # learn templates from normal traffic
    python -m logmit.parse --run fault1          # frozen templates; unseen lines -> UNK
"""
import argparse
import csv
import json
import re

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from .config import MODEL_DIR, UNK, run_dir

MASKS = [
    (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "UUID"),
    (r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "IP"),
    (r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}\b", "PID"),  # Online Boutique product ids
    (r"\b(?:0x)?[0-9a-fA-F]{12,}\b", "HEX"),
    (r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])", "NUM"),
]
STATUS = {2: "ok", 3: "redirect", 4: "clienterror", 5: "servererror"}
ERROR_WORDS =re.compile(r"error|fail|exception|fatal|panic|timeout|timed out|refused|unavailable", re.I)


def make_miner(fit=False, state=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    state = state or MODEL_DIR / "drain.bin"
    if fit and state.exists():
        state.unlink()
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5
    cfg.drain_depth = 4
    cfg.masking_instructions = [MaskingInstruction(p, m) for p, m in MASKS]
    return TemplateMiner(FilePersistence(str(state)), cfg)


def clean(svc, line):
    """Reduce structured (JSON) lines to their message. Service, severity and HTTP status class go
    into the first token: Drain never merges lines whose first token differs, so templates stay
    per service and a 5xx response can never share a template with a 2xx one."""
    line = line.strip()
    prefix = [svc]
    if line.startswith("{"):
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            prefix.append(str(obj.get("severity") or obj.get("level") or "").lower())
            status = obj.get("http.resp.status")
            if isinstance(status, int):
                # words, not "5xx": Drain treats any token containing a digit as a variable
                prefix.append(STATUS.get(status // 100, "other"))
            line = str(obj.get("message") or obj.get("msg") or "")
    return "/".join(p for p in prefix if p) + " " + line


def encode(miner, svc, line, fit=False):
    text = clean(svc, line)
    if fit:
        return miner.add_log_message(text)["cluster_id"] + 2
    cluster = miner.match(text)
    return cluster.cluster_id + 2 if cluster else UNK


def templates(miner):
    return {c.cluster_id + 2: c.get_template() for c in miner.drain.clusters}


def parse(run, fit):
    miner = make_miner(fit)
    d = run_dir(run)
    n = unk = 0
    services = set()
    with open(d / "logs.jsonl", encoding="utf-8") as src, \
            open(d / "keys.csv", "w", newline="", encoding="utf-8") as dst:
        w = csv.writer(dst)
        w.writerow(["ts", "svc", "key", "err"])
        for row in src:
            r = json.loads(row)
            key = encode(miner, r["svc"], r["line"], fit)
            # err feeds the error-rate baseline, which must see the raw text: new error messages become UNK
            w.writerow([r["ts"], r["svc"], key, int(bool(ERROR_WORDS.search(r["line"])))])
            services.add(r["svc"])
            n += 1
            unk += key == UNK
    tpl = templates(miner)
    if fit:
        miner.save_state("fit complete")
        meta = {"services": sorted(services), "vocab": max(tpl) + 1, "fit_run": run}
        (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
        (MODEL_DIR / "templates.json").write_text(json.dumps(tpl, indent=1))
    print(f"{run}: {n} lines, {len(tpl)} templates, {unk} unknown ({unk / max(n, 1):.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    parse(a.run, a.fit)
