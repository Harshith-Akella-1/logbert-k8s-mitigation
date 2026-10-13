"""Stage 5: mitigation controller. Table I policy with a per-service cooldown and an action budget."""
import re
import time
from collections import defaultdict, deque

from kubernetes import client, config

from .config import NAMESPACE, SERVICES, append_jsonl

COOLDOWN = 60.0          # seconds between actions on the same service
BUDGET, BUDGET_WINDOW = 3, 300.0  # automatic actions allowed per five minutes
MAX_REPLICAS = 5
CRITICAL_STREAK = 3      # anomalous windows in a row, after acting, before giving up and alerting

DEPENDENCY = re.compile(r"timeout|timed out|deadline|unavailable|refused|rpc error|failed to|could not", re.I)
CRASH = re.compile(r"panic|exception|fatal|traceback|segfault|out of memory|oom", re.I)


def classify(svc, lines):
    """Suspected fault class from the raw lines of the anomalous window, plus any dependency named in them.
    Keyword rules stand in for the paper's classifier head."""
    text = "\n".join(lines)
    deps = [s for s in SERVICES if s != svc
            and re.search(rf"\b{re.escape(s.replace('service', ''))}(s|service)?\b", text, re.I)]
    if DEPENDENCY.search(text):
        return "dependency", deps
    if CRASH.search(text):
        return "crash", []
    return "slow", []


class Controller:
    def __init__(self, log_path, dry_run=False):
        self.log_path, self.dry_run = log_path, dry_run
        self.last_action = defaultdict(float)
        self.recent = deque()
        if not dry_run:
            config.load_kube_config()
            self.v1, self.apps = client.CoreV1Api(), client.AppsV1Api()

    def restart_one(self, svc):
        """Delete a single replica; the Deployment replaces it."""
        if self.dry_run:
            return
        items = self.v1.list_namespaced_pod(NAMESPACE, label_selector=f"app={svc}").items
        if items:
            self.v1.delete_namespaced_pod(items[0].metadata.name, NAMESPACE)

    def scale_up(self, svc):
        if self.dry_run:
            return
        n = self.apps.read_namespaced_deployment_scale(svc, NAMESPACE).spec.replicas or 0
        if n < MAX_REPLICAS:
            self.apps.patch_namespaced_deployment_scale(svc, NAMESPACE, {"spec": {"replicas": n + 1}})

    def handle(self, svc, lines, streak, score, log_ts):
        """Called on every verdict (streak = consecutive anomalous windows for this service)."""
        now = time.time()
        fault, deps = classify(svc, lines)
        acted_before = self.last_action[svc] > 0 and now - self.last_action[svc] < BUDGET_WINDOW
        while self.recent and now - self.recent[0] > BUDGET_WINDOW:
            self.recent.popleft()

        if acted_before and streak > CRITICAL_STREAK + 1:
            action = "alert:persistent"
        elif now - self.last_action[svc] < COOLDOWN:
            action = "skip:cooldown"
        elif len(self.recent) >= BUDGET:
            action = "alert:budget"
        else:
            if fault == "crash":
                action = f"restart {svc}"
                self.restart_one(svc)
            elif fault == "dependency":
                action = f"restart {svc}" + "".join(f"; scale {d} +1" for d in deps)
                self.restart_one(svc)
                for d in deps:
                    self.scale_up(d)
            else:
                action = f"scale {svc} +1"
                self.scale_up(svc)
            self.last_action[svc] = now
            self.recent.append(now)
        append_jsonl(self.log_path, {"ts": now, "log_ts": log_ts, "svc": svc, "fault_class": fault,
                                     "streak": streak, "score": score, "action": action})
        return action
