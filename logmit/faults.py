"""Stage 3a: inject faults with plain Kubernetes API calls and log each interval to <run>/faults.jsonl.
Run the collector in another terminal at the same time.

    python -m logmit.faults --run fault_val --reps 3 --duration 120 --gap 180
"""
import argparse
import random
import time

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .config import NAMESPACE, SERVICES, affected, append_jsonl, run_dir

# fault -> services it is injected into
TARGETS = {
    "pod_kill": ["productcatalogservice", "currencyservice", "cartservice"],
    "outage": ["redis-cart", "recommendationservice", "shippingservice"],
    "cpu_throttle": ["frontend", "productcatalogservice", "currencyservice"],
}


def pods(v1, svc):
    return v1.list_namespaced_pod(NAMESPACE, label_selector=f"app={svc}").items


def pod_kill(v1, apps, svc, duration, every=20):
    """Crash loop: force-delete the pod repeatedly; the Deployment keeps recreating it."""
    end = time.time() + duration
    while time.time() < end:
        for p in pods(v1, svc)[:1]:
            v1.delete_namespaced_pod(p.metadata.name, NAMESPACE, grace_period_seconds=0)
        time.sleep(every)


def outage(v1, apps, svc, duration):
    """Dependency outage: scale the Deployment to zero, then restore the replica count."""
    before = apps.read_namespaced_deployment_scale(svc, NAMESPACE).spec.replicas or 1
    apps.patch_namespaced_deployment_scale(svc, NAMESPACE, {"spec": {"replicas": 0}})
    try:
        time.sleep(duration)
    finally:
        # the controller may have scaled it back up already; never scale down here
        now = apps.read_namespaced_deployment_scale(svc, NAMESPACE).spec.replicas or 0
        apps.patch_namespaced_deployment_scale(svc, NAMESPACE, {"spec": {"replicas": max(before, now)}})


def cpu_throttle(v1, apps, svc, duration, cpu="10m"):
    """CPU throttling: shrink the running pod's CPU limit in place (no restart), then restore it."""
    found = pods(v1, svc)
    if not found:
        return
    pod = found[0]
    c = pod.spec.containers[0]
    original = {"requests": {"cpu": (c.resources.requests or {}).get("cpu", "100m")},
                "limits": {"cpu": (c.resources.limits or {}).get("cpu", "200m")}}

    def resize(resources):
        body = {"spec": {"containers": [{"name": c.name, "resources": resources}]}}
        v1.patch_namespaced_pod_resize(pod.metadata.name, NAMESPACE, body)

    resize({"requests": {"cpu": cpu}, "limits": {"cpu": cpu}})
    try:
        time.sleep(duration)
    finally:
        try:
            resize(original)
        except ApiException as e:
            if e.status != 404:  # 404: the pod was replaced, so the throttle is already gone
                raise


FAULTS = {"pod_kill": pod_kill, "outage": outage, "cpu_throttle": cpu_throttle}


def main(a):
    config.load_kube_config()
    v1, apps = client.CoreV1Api(), client.AppsV1Api()
    v1.list_namespaced_pod(NAMESPACE, _request_timeout=10)  # fail now, not mid-run, if the cluster is down
    plan =[(f, t) for f in a.faults for t in TARGETS[f]] * a.reps
    random.Random(a.seed).shuffle(plan)
    log = run_dir(a.run) / "faults.jsonl"
    print(f"{len(plan)} episodes, about {len(plan) * (a.duration + a.gap) / 60:.0f} minutes")
    time.sleep(a.gap)
    for i, (fault, target) in enumerate(plan, 1):
        start = time.time()
        print(f"[{i}/{len(plan)}] {fault} -> {target}", flush=True)
        try:
            FAULTS[fault](v1, apps, target, a.duration)
        finally:
            append_jsonl(log, {"fault": fault, "target": target, "affected": affected(target),
                               "start": start, "end": time.time()})
        if a.reset_replicas:
            # closed-loop runs: undo any scale-ups once the settling period is over,
            # so every episode starts from one replica per service
            settle = min(60.0, a.gap)
            time.sleep(settle)
            for d in apps.list_namespaced_deployment(NAMESPACE).items:
                if d.metadata.name in SERVICES and d.spec.replicas != 1:
                    apps.patch_namespaced_deployment_scale(d.metadata.name, NAMESPACE, {"spec": {"replicas": 1}})
            time.sleep(a.gap - settle)
        else:
            time.sleep(a.gap)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--faults", nargs="+", default=list(FAULTS), choices=list(FAULTS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--duration", type=float, default=120, help="seconds per fault")
    ap.add_argument("--gap", type=float, default=180, help="normal seconds between faults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset-replicas", action="store_true", help="return every Deployment to 1 replica after each episode")
    main(ap.parse_args())
