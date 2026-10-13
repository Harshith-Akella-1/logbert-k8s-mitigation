"""Rebuilds the paper PDF. Every reported number is read from results/*.json, which the
logmit scripts write from real runs; anything not measured yet is printed as "pending".

    python build_paper.py                # writes docs/paper.pdf
    python build_paper.py other.pdf
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, KeepTogether, FrameBreak,
                                NextPageTemplate)

from logmit.faults import TARGETS

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8

ROOT = Path(__file__).resolve().parent
RES = Path(os.environ.get("LOGMIT_RESULTS", ROOT / "results"))
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "paper.pdf"

# ---------------------------------------------------------------- results
def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None

DET = load("detection.json")          # logmit.evaluate
REACT = load("loop_reactive.json")    # logmit.metrics --run loop_reactive
PRO = load("loop_proactive.json")     # logmit.metrics --run loop_proactive
CLUSTER = load("cluster.json")        # infra/setup.ps1
TBD = "pending"
missing = []  # grows each time a number that should be in the paper is not available

def num(v, f="{:.2f}"):
    if v is None:
        missing.append(f)
        return TBD
    return f.format(v)

def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

def loop(run, f, *path):
    """A closed-loop metric. Pending if the run is missing; n/a if the run exists but the metric is
    undefined (for example MTTR when no request failed)."""
    if run is None:
        return num(None)
    v = get(run, *path)
    return "n/a" if v is None else f.format(v)

LB = get(DET, "methods", "logbert")
DS = get(DET, "dataset") or {}
ARCH = get(DET, "models", "logbert") or {}
MAIN_FAULTS = ["pod_kill", "cpu_throttle"]
FAULT_NAMES = {"pod_kill": "Pod kill", "outage": "Outage", "cpu_throttle": "CPU throttle"}

# ---------------------------------------------------------------- figures
def fig_arch(path):
    fig, ax = plt.subplots(figsize=(3.4, 3.7), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
    boxes = [
        ("Microservices (Online Boutique, 11 services)\nMinikube cluster", "#dbe9f6"),
        ("Log collector (Kubernetes log API, 2 s poll)", "#e8f0dc"),
        ("Log parser (Drain) -> template keys", "#e8f0dc"),
        ("Sliding-window sequencer (w = 32)", "#e8f0dc"),
        ("LogBERT anomaly detector\n(MLM + VHM, top-g scoring)", "#f9e3c8"),
        ("Mitigation controller\n(policy map + cooldown guard)", "#f6d5d5"),
        ("Kubernetes API (restart / scale)", "#dbe9f6"),
    ]
    n = len(boxes); h = 1.2; gap = 0.45
    y = 11 - h
    ys = []
    for text, c in boxes:
        ax.add_patch(FancyBboxPatch((0.6, y), 8.0, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=c, ec="#333333", lw=0.7))
        ax.text(4.6, y + h / 2, text, ha="center", va="center", fontsize=6.6)
        ys.append(y)
        y -= h + gap
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((4.6, ys[i] - 0.02), (4.6, ys[i + 1] + h + 0.02),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#222222"))
    ax.add_patch(FancyArrowPatch((8.6, ys[6] + h / 2), (9.5, ys[6] + h / 2),
                                 arrowstyle="-", lw=0.8, color="#b03030"))
    ax.add_patch(FancyArrowPatch((9.5, ys[6] + h / 2), (9.5, ys[0] + h / 2),
                                 arrowstyle="-", lw=0.8, color="#b03030"))
    ax.add_patch(FancyArrowPatch((9.5, ys[0] + h / 2), (8.62, ys[0] + h / 2),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#b03030"))
    ax.text(9.75, 5.5, "feedback loop", rotation=90, va="center", ha="center", fontsize=6, color="#b03030")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def fig_f1(path):
    faults = sorted(LB["per_fault"])
    series = [("Isolation Forest", "Isolation Forest", "#b9c4d0"), ("DeepLog", "DeepLog", "#7f9bb8"),
              ("logbert", "LogBERT (ours)", "#c0562b")]
    x = np.arange(len(faults)); w = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    for i, (key, label, c) in enumerate(series):
        ax.bar(x + (i - 1) * w, [DET["methods"][key]["per_fault"][f] for f in faults], w, label=label, color=c)
    ax.set_xticks(x); ax.set_xticklabels([FAULT_NAMES.get(f, f) for f in faults], fontsize=6.5)
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1-score")
    ax.legend(fontsize=6, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def fig_mttr(path):
    x = np.arange(len(MAIN_FAULTS)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.4, 2.1), dpi=300)
    for run, off, label, c in ((REACT, -w / 2, "Reactive (liveness probe + HPA)", "#b9c4d0"),
                               (PRO, w / 2, "Proactive (ours)", "#c0562b")):
        vals = [get(run, "per_fault", f, "mttr_mean_s") for f in MAIN_FAULTS]
        ax.bar(x + off, [v or 0 for v in vals], w, label=label, color=c)
        for xi, v in zip(x + off, vals):
            if v is None:
                ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([FAULT_NAMES[f] for f in MAIN_FAULTS], fontsize=6.5)
    ax.set_ylabel("MTTR (s)")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def fig_thr(path):
    sweep = LB["tau_sweep"]
    thr = [s["tau"] for s in sweep]
    fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
    ax.plot(thr, [s["precision"] for s in sweep], "o-", ms=2.5, lw=1, label="Precision", color="#7f9bb8")
    ax.plot(thr, [s["recall"] for s in sweep], "s-", ms=2.5, lw=1, label="Recall", color="#6a8f4e")
    ax.plot(thr, [s["f1"] for s in sweep], "^-", ms=2.5, lw=1.2, label="F1", color="#c0562b")
    ax.axvline(LB["tau"], ls="--", lw=0.6, color="gray")
    ax.set_xlabel("Anomaly-score threshold"); ax.set_ylabel("Score")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(lw=0.3, alpha=0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

FIG = {k: str(BUILD / f"fig_{k}.png") for k in ("arch", "f1", "mttr", "thr")}
fig_arch(FIG["arch"])
if LB:
    fig_f1(FIG["f1"]); fig_thr(FIG["thr"])
if REACT and PRO:
    fig_mttr(FIG["mttr"])

# ---------------------------------------------------------------- styles
body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.8, leading=14, alignment=TA_JUSTIFY,
                      firstLineIndent=14, spaceAfter=5)
body0 = ParagraphStyle("body0", parent=body, firstLineIndent=0)
abst = ParagraphStyle("abst", parent=body, fontName="Times-Bold", firstLineIndent=0)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=11, leading=13, alignment=TA_CENTER,
                    spaceBefore=10, spaceAfter=5)
h2 = ParagraphStyle("h2", fontName="Times-BoldItalic", fontSize=10.8, leading=13, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle("title", fontName="Times-Bold", fontSize=17, leading=20, alignment=TA_CENTER, spaceAfter=6)
auth = ParagraphStyle("auth", fontName="Times-Roman", fontSize=9.5, leading=11.5, alignment=TA_CENTER)
authb = ParagraphStyle("authb", parent=auth, fontName="Times-Bold")
affil = ParagraphStyle("affil", parent=auth, fontName="Times-Italic", fontSize=9)
cap = ParagraphStyle("cap", fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=5)
tcap = ParagraphStyle("tcap", fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=2,
                      spaceBefore=3)
cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=10.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
ref = ParagraphStyle("ref", fontName="Times-Roman", fontSize=9.5, leading=11.5, alignment=TA_JUSTIFY,
                     leftIndent=16, firstLineIndent=-14, spaceAfter=1.5)

COLW = 4.6 * inch

def P(t, s=body): return Paragraph(t, s)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)

def fig(path, caption, w=COLW, h=None):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    h = w * ih / iw
    return KeepTogether([Image(path, width=w, height=h), Spacer(1, 2), Paragraph(caption, cap)])

def table(rows, widths, caption, bold_last=False, hl_row=None):
    data = []
    for i, r in enumerate(rows):
        st = cellb if i == 0 or (hl_row is not None and i == hl_row) else cell
        data.append([Paragraph(str(c), st) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    ts = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]
    if hl_row is not None:
        ts.append(("BACKGROUND", (0, hl_row), (-1, hl_row), colors.HexColor("#f4e6da")))
    t.setStyle(TableStyle(ts))
    return KeepTogether([Paragraph(caption, tcap), t, Spacer(1, 6)])

# ---------------------------------------------------------------- headline sentences
# Worded so they stay true whichever way the measured numbers fall.
if LB:
    detection_claim = (f"the detector reaches an F1-score of {LB['f1']:.2f} on held-out fault windows "
                       f"(DeepLog: {DET['methods']['DeepLog']['f1']:.2f})")
else:
    detection_claim = "detection results are pending the experimental runs"
if REACT and PRO:
    loop_claim = (f"In the closed-loop comparison on pod-kill and CPU-throttling faults, mean time to recovery is "
                  f"{loop(REACT, '{:.1f}', 'main', 'mttr_mean_s')} s with liveness probes and the Horizontal Pod "
                  f"Autoscaler and {loop(PRO, '{:.1f}', 'main', 'mttr_mean_s')} s with the proposed controller, and "
                  f"request availability is {loop(REACT, '{:.2f}', 'main', 'availability_pct')}% and "
                  f"{loop(PRO, '{:.2f}', 'main', 'availability_pct')}% respectively.")
else:
    loop_claim = "Closed-loop recovery results are pending the experimental runs."

# ---------------------------------------------------------------- content
S = []
S.append(NextPageTemplate("later"))
S.append(Paragraph("Proactive Microservice Failure Mitigation using LogBERT Anomaly Detection in Kubernetes", title))
S.append(Spacer(1, 2))
names = [("Akella Harshith Saipaavan", "160124737166"), ("Bavikadi Mohaseen Hussain", "160124737170"),
         ("M Jaidev", "160124737186"), ("Ponna Manish Reddy", "160124737192")]
cells = [[Paragraph(f"<b>{n}</b><br/>{r}", auth) for n, r in names]]
at = Table(cells, colWidths=[1.75 * inch] * 4)
at.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
S.append(at)
S.append(Spacer(1, 3))
S.append(Paragraph("Department of Information Technology (IT-3), Chaitanya Bharathi Institute of Technology (A), Hyderabad, India", affil))
S.append(Spacer(1, 8))
S.append(FrameBreak())

S.append(P("<b><i>Abstract</i></b>&#8212;Microservice applications on Kubernetes fail in ways that liveness probes and "
           "threshold alerts detect only after users are affected. We present a closed-loop framework that learns the "
           "normal behaviour of a service from its logs and acts before a degradation becomes an outage. Container logs "
           "are parsed into template sequences, scored by a LogBERT model trained with masked log-key prediction and a "
           "hypersphere objective, and the resulting anomaly score drives a mitigation controller that restarts or "
           "scales the affected workload through the Kubernetes API. On a single-node Minikube deployment of the "
           f"Online Boutique demo application with three injected fault types, {detection_claim}. {loop_claim}", abst))
S.append(P("<b><i>Index Terms</i></b>&#8212;Kubernetes, microservices, log anomaly detection, LogBERT, self-healing, "
           "AIOps, fault injection.", body0))

S.append(H1("I. INTRODUCTION"))
S.append(P("Microservice architectures split an application into many small, independently deployed services that "
           "communicate over the network [10]. The pay-off is agility and scalability; the price is a large failure "
           "surface. A single memory leak, a saturated connection pool or a slow downstream dependency can cascade "
           "through the call graph and turn a local fault into a user-visible incident. Kubernetes [6] automates "
           "deployment and offers basic recovery, but its built-in mechanisms are fundamentally reactive: a liveness "
           "probe restarts a container only after it has already stopped answering, and the Horizontal Pod "
           "Autoscaler (HPA) reacts to CPU or memory averages that lag behind the underlying fault."))
S.append(P("Logs are the richest and cheapest signal available to detect trouble early. Services emit characteristic "
           "message patterns while healthy, and the first symptoms of a fault, such as growing retry messages, "
           "garbage-collection warnings or timeouts to a dependency, appear in the log stream well before metrics "
           "cross an alerting threshold. Deep-learning detectors such as DeepLog [2] and LogBERT [1] model these "
           "sequences and have shown strong results on offline benchmark datasets. Far less work closes the loop by "
           "turning a detection into a safe, automatic remedy inside an orchestrator."))
S.append(P("This paper makes three contributions: (1) an end-to-end architecture that connects a log pipeline, a "
           "LogBERT detector and a mitigation controller around a Kubernetes cluster; (2) a graded policy that maps "
           "the suspected fault class and the persistence of an anomaly to a minimal-impact remedy, with a cool-down "
           "guard that prevents oscillation; and (3) an empirical study on a Minikube testbed with three injected "
           "fault types, comparing detection quality against classical baselines and recovery against the reactive "
           "mechanisms built into Kubernetes."))

S.append(H1("II. RELATED WORK"))
S.append(P("<b>Log anomaly detection.</b> DeepLog [2] models the log-key sequence with an LSTM and flags a window "
           "when the observed key is not among the top-<i>g</i> predictions. LogAnomaly [4] adds semantic template "
           "vectors to cope with unseen messages, and LogRobust [5] uses attention to tolerate log instability. "
           "LogBERT [1] applies a bidirectional Transformer trained with masked log-key prediction and a volume of "
           "hypersphere minimisation objective, capturing context on both sides of a position and outperforming "
           "unidirectional models on HDFS, BGL and Thunderbird. Our work reuses this detector but evaluates it on "
           "live microservice logs and couples it to remediation."))
S.append(P("<b>Log parsing.</b> Drain [3] groups messages with a fixed-depth parse tree, runs online and is robust "
           "enough for streaming use. We use it to convert raw lines into template identifiers (log keys). "
           "The Loghub collection [8] provides reference datasets that guided our window and vocabulary choices."))
S.append(P("<b>Self-healing and AIOps.</b> Commercial and open-source orchestrators ship restart and autoscaling "
           "rules, but these are threshold-driven. Chaos engineering tools such as Chaos Mesh [7] are used to test "
           "resilience rather than to train detectors. Our framework combines both ideas: injected faults produce "
           "labelled data for evaluation, and the learned detector triggers the remedy."))

S.append(H1("III. SYSTEM ARCHITECTURE"))
S.append(P("Fig. 1 shows the pipeline. Every pod writes to stdout; a collector polls the Kubernetes log API every "
           "2 s and forwards new lines, tagged with pod and service name, to the parser. The parser outputs a "
           "stream of (service, timestamp, log key) tuples. A sequencer keeps one sliding window per service. The "
           "detector scores each window and publishes a verdict to the mitigation controller, which acts through "
           "the Kubernetes API and whose actions are themselves observed through the next windows of logs."))
S.append(fig(FIG["arch"], "Fig. 1. Architecture of the proactive mitigation framework. Red arrow: the "
             "controller's actions change the cluster state, which is observed in subsequent log windows."))
S.append(H2("A. Log collection and parsing"))
S.append(P("Structured (JSON) lines are first reduced to their severity, message and HTTP status class. Lines are "
           "then masked (IP addresses, UUIDs, hexadecimal identifiers, numbers) and passed through Drain with a "
           "similarity threshold of 0.5 and depth 4. Templates are learned from normal traffic only, which yields "
           f"{num(DS.get('templates'), '{}')} templates; messages unseen at run time are mapped to a reserved unknown "
           "key rather than forcing a retrain. The service name is prefixed to every line, so keys are produced per "
           "service and the behaviour of one component is not diluted by the others."))
S.append(H2("B. LogBERT detector"))
S.append(P("For a window of <i>w</i> = 32 consecutive keys, the input embedding is the sum of key, positional and "
           f"service embeddings. The encoder has {ARCH.get('layers', 4)} layers, {ARCH.get('heads', 4)} attention "
           f"heads and a hidden size of {ARCH.get('hidden', 256)}, giving "
           f"{num(ARCH['params'] / 1e6 if ARCH else None)} M parameters. Training uses two self-supervised "
           "objectives on normal data only. The first is masked "
           "log-key prediction (MLKP), where 15% of positions are replaced by a mask token and predicted from "
           "bidirectional context. The second is volume of hypersphere minimisation (VHM), which pulls the "
           "window representation of normal sequences towards a common centre so that deviating sequences land "
           "far from it. The total loss is the MLKP loss plus a weighted VHM term with weight 0.1."))
S.append(P("At inference, every position of a window is masked exactly once (eight forward passes of four "
           "positions each) and we count the positions where the true key is outside the top-<i>g</i> predictions. "
           "The anomaly score is that fraction, and a window is anomalous if the score exceeds a threshold "
           "<i>tau</i>. Both are selected on a validation fault run by maximising F1, giving "
           f"<i>g</i> = {num(get(LB, 'g'), '{}')} and <i>tau</i> = {num(get(LB, 'tau'))} (Fig. 3). A verdict is "
           "raised only after two consecutive anomalous windows to suppress isolated noise."))
S.append(H2("C. Mitigation controller"))
S.append(P("The controller turns a verdict into an action using the suspected fault class and the persistence of "
           "the anomaly. The fault class is assigned by keyword rules over the raw lines of the anomalous window: "
           "timeouts, refused connections and RPC errors mark a dependency fault, and any other service named in "
           "those lines is taken as the failing dependency; panics, exceptions and out-of-memory messages mark a "
           "crash; an anomalous window with neither is treated as a slow handler. Table I lists the actions. Each "
           "service has a cool-down of 60 s between actions, and the controller as a whole is limited to three "
           "automatic actions per five minutes; beyond that it only raises an alert. A restart removes one replica "
           "at a time."))
S.append(table([
    ["Suspected fault", "Severity", "Action"],
    ["Crash / exception burst", "Medium", "Restart one pod of the service"],
    ["Dependency error", "Medium", "Restart one caller pod; scale each named dependency +1 (max 5)"],
    ["No explicit error (slow handler)", "High", "Scale replicas by +1 (max 5)"],
    ["Still anomalous for > 4 windows after an action", "Critical", "Alert only; stop automatic actions"],
], [2.2 * inch, 0.8 * inch, 3.0 * inch], "TABLE I. Mitigation policy (severity-ordered, least disruptive first)."))

S.append(H1("IV. EXPERIMENTAL SETUP"))
S.append(H2("A. Testbed"))
k8s = get(CLUSTER, "serverVersion", "gitVersion")
S.append(P("Experiments run on one laptop (Intel Core i5, 16 GB RAM, Windows 11) hosting a single-node Minikube "
           f"1.39 cluster{' (Kubernetes ' + k8s + ')' if k8s else ''} with the Docker driver, limited to 4 vCPU and "
           "6 GB of memory. The cluster runs release v0.10.3 of the Online Boutique demo [9], which consists of 11 "
           "services written in several languages and a Redis cache. Load comes from the bundled load generator at "
           "its default setting of 10 simulated users. A separate prober pod requests three frontend pages (home, "
           "a product page and the cart) in a loop with a 0.5 s pause and a 3 s timeout; its requests define what "
           "counts as a user-visible failure. The collector, detector and controller run as processes on the host "
           "and reach the cluster through the Kubernetes API; the detector is trained and run on the laptop's GPU "
           "(NVIDIA RTX 3050, 6 GB)."))
S.append(H2("B. Dataset"))
targets = lambda f: ", ".join(TARGETS[f])
S.append(P("Data are generated in two phases. In the normal phase the application runs under load with no faults; "
           "these logs train Drain and all detectors. Normal traffic is not error-free: the load generator itself "
           "occasionally triggers a failed checkout or a failed ad lookup, and these lines are kept in the training "
           "data. In the fault phase we inject three fault types through the "
           f"Kubernetes API: <i>pod kill</i>, where the pod is force-deleted every 20 s ({targets('pod_kill')}); "
           f"<i>outage</i>, where the Deployment is scaled to zero replicas ({targets('outage')}); and <i>CPU "
           f"throttling</i>, where the CPU limit of the running pod is cut in place to 10 millicores "
           f"({targets('cpu_throttle')}). Episodes last about {num(DS.get('episode_median_s'), '{:.0f}')} s and "
           "follow a shuffled schedule with normal traffic in between. Two fault runs are recorded: one to select "
           "thresholds (validation) and one to report results (test). A window is labelled anomalous if it overlaps "
           "an injected interval and belongs to the faulted service or to a service that calls it, directly or "
           "transitively. Windows in the 60 s after a fault ends are excluded from scoring. Table II summarises "
           "the data."))
pair = lambda a, b: TBD if DS.get(a) is None else f"{DS[a]:,} / {DS[b]:,}"
S.append(table([
    ["Item", "Value"],
    ["Monitored services / templates", pair("services", "templates")],
    ["Normal log lines", num(DS.get("normal_lines"), "{:,}")],
    ["Training / held-out windows (normal run)", pair("train_windows", "held_out_windows")],
    ["Validation windows (normal / anomalous)", pair("val_normal", "val_anomalous")],
    ["Test windows (normal / anomalous)", pair("test_normal", "test_anomalous")],
    ["Injected fault episodes (validation / test)", pair("val_episodes", "test_episodes")],
    ["Window size / stride", "32 keys / 8 keys"],
], [3.4 * inch, 1.4 * inch], "TABLE II. Dataset summary."))
S.append(H2("C. Baselines and metrics"))
S.append(P("For detection we compare against a static rule (the share of lines in a window that contain an error "
           "keyword), Isolation Forest [11] on template-count vectors, and DeepLog [2]. Every method's threshold "
           "is chosen on the validation run and frozen for the test run. For mitigation the baseline is the default "
           "Kubernetes behaviour: the probes shipped with Online Boutique plus an HPA targeting 70% CPU on the "
           "services that receive the CPU-throttling fault. We report precision, recall and F1 for detection; "
           "detection lead time (seconds between the first verdict on an affected service and the first failed "
           "probe); mean time to recovery (MTTR, from the first failed probe to the start of the first 10 s "
           "stretch without a failed probe); and request availability (the share of successful probes). Each "
           "condition is measured in a single run."))
S.append(P("The recovery comparison is reported on pod kill and CPU throttling. The outage fault is reported "
           "separately: a Deployment scaled to zero cannot be recovered by probes or the HPA until the fault is "
           "withdrawn, whereas the controller's scale-up action reverses it directly, so including it in the "
           "main comparison would favour the controller by construction."))

S.append(H1("V. RESULTS AND DISCUSSION"))
note_at = len(S)
S.append(H2("A. Detection quality"))
METHODS = [("Error-rate rule", "Error-rate rule"), ("Isolation Forest", "Isolation Forest"), ("DeepLog", "DeepLog"),
           ("logbert", "LogBERT (ours)")]
def prf_row(label, m):
    return [label] + [num(get(m, k)) for k in ("precision", "recall", "f1")]
S.append(P("Table III reports precision, recall and F1 of each method on the test windows, Fig. 2 breaks F1 down "
           "by fault type, and Fig. 3 shows how the LogBERT result depends on the threshold."))
S.append(table([["Method", "Precision", "Recall", "F1"]] + [prf_row(label, get(DET, "methods", key)) for key, label in METHODS],
               [2.0 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch], "TABLE III. Detection results on the test windows.", hl_row=4))
if LB:
    S.append(fig(FIG["f1"], "Fig. 2. F1-score per injected fault type, computed on the windows of that fault plus "
                 "all normal windows."))
    S.append(fig(FIG["thr"], "Fig. 3. Effect of the anomaly-score threshold on precision, recall and F1 on the test "
                 f"windows. The dashed line marks the value chosen on the validation run, tau = {LB['tau']:.2f}."))
else:
    S.append(P("Figs. 2 and 3 are pending.", body0))
S.append(H2("B. Early warning and recovery"))
S.append(P("Table IV compares the reactive baseline with the proposed controller on pod kill and CPU throttling, "
           "and Fig. 4 shows MTTR per fault type. Lead time is positive when the verdict came before the first "
           "failed probe."))
def loop_row(label, f, *path):
    return [label, loop(REACT, f, *path), loop(PRO, f, *path)]
S.append(table([
    ["Metric", "Reactive", "Proactive"],
    loop_row("Mean time to recovery (s)", "{:.1f}", "main", "mttr_mean_s"),
    loop_row("Request availability (%)", "{:.2f}", "main", "availability_pct"),
    loop_row("Failed requests during fault (%)", "{:.1f}", "main", "failed_requests_during_fault_pct"),
    loop_row("Episodes with failed requests", "{}", "main", "episodes_with_user_impact"),
    loop_row("Episodes not recovered before the fault ended", "{}", "main", "episodes_not_recovered"),
    loop_row("Unnecessary actions per hour", "{:.1f}", "unnecessary_actions_per_hour"),
    ["Median detection lead time (s)", "n/a", loop(PRO, "{:.1f}", "main", "lead_time_median_s")],
], [2.9 * inch, 0.9 * inch, 0.9 * inch], "TABLE IV. Mitigation outcomes on pod kill and CPU throttling (one run each)."))
if REACT and PRO:
    S.append(fig(FIG["mttr"], "Fig. 4. Mean time to recovery per fault type, reactive baseline versus the "
                 "proposed proactive framework."))
else:
    S.append(P("Fig. 4 is pending.", body0))
S.append(P("For the outage fault, reported separately for the reason given in Section IV-C, MTTR is "
           f"{loop(REACT, '{:.1f}', 'per_fault', 'outage', 'mttr_mean_s')} s for the reactive baseline and "
           f"{loop(PRO, '{:.1f}', 'per_fault', 'outage', 'mttr_mean_s')} s for the controller, with "
           f"{loop(REACT, '{:.1f}', 'per_fault', 'outage', 'failed_requests_during_fault_pct')}% and "
           f"{loop(PRO, '{:.1f}', 'per_fault', 'outage', 'failed_requests_during_fault_pct')}% of requests "
           "failing during the fault."))
S.append(H2("C. Ablation"))
S.append(P("Table V repeats the detection evaluation with one component changed at a time. The last row applies "
           "the two-window persistence rule used by the controller to the full model's window verdicts."))
ABL = [("logbert", "Full model"), ("logbert_novhm", "Without VHM objective"),
       ("logbert_nosvc", "Without service embedding"), ("logbert_2layer", "Encoder with 2 layers")]
S.append(table([["Variant", "Precision", "Recall", "F1"]] + [prf_row(label, get(DET, "methods", key)) for key, label in ABL]
               + [prf_row("Full model + 2-window persistence rule", get(LB, "with_persistence"))],
               [2.6 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch], "TABLE V. Ablation study on the test windows.", hl_row=1))
S.append(H2("D. Threats to validity and limitations"))
S.append(P("The study uses a single demo application, synthetic faults and a one-node cluster on a laptop; real "
           "production logs are noisier, drift with every release and are produced at far higher volume. Only "
           "three fault types are covered, injected through the Kubernetes API rather than a chaos tool, so memory "
           "leaks and network faults are not evaluated. Each condition is a single run, so the results carry no "
           "confidence intervals. Faults are labelled by interval and by the static call graph, which blurs the "
           "exact onset and marks some unaffected windows as anomalous. The fault class comes from keyword rules "
           "rather than a learned classifier, and the policy has no reschedule or resource-request action. The "
           "CPU-throttling fault lives in the running pod, so a restart removes it, which may not hold for a "
           "throttle caused by the workload itself. Logs are polled every 2 s, which bounds how early a verdict "
           "can arrive and can lose the last lines of a container that crashes between polls. Automatic "
           "remediation can also do harm if the detector is wrong, which is why the framework limits its action "
           "budget and escalates persistent anomalies to a human. Future work includes more fault types, repeated "
           "runs, a learned fault classifier, drift-aware incremental training, evaluation on a public benchmark "
           "such as those in Loghub [8], and trials on a multi-node cluster."))

S.append(H1("VI. CONCLUSION"))
S.append(P("We presented a closed-loop framework that uses LogBERT anomaly detection to trigger proactive "
           "mitigation in Kubernetes. On a single-node Minikube deployment of the Online Boutique application, "
           f"{detection_claim}. {loop_claim} The approach needs only logs and standard Kubernetes APIs, and the "
           "explicit action budget keeps its behaviour predictable."))

S.append(H1("REFERENCES"))
refs = [
    "H. Guo, S. Yuan and X. Wu, &#8220;LogBERT: Log anomaly detection via BERT,&#8221; in <i>Proc. Int. Joint Conf. Neural Networks (IJCNN)</i>, 2021.",
    "M. Du, F. Li, G. Zheng and V. Srikumar, &#8220;DeepLog: Anomaly detection and diagnosis from system logs through deep learning,&#8221; in <i>Proc. ACM SIGSAC Conf. Computer and Communications Security (CCS)</i>, 2017, pp. 1285&#8211;1298.",
    "P. He, J. Zhu, Z. Zheng and M. R. Lyu, &#8220;Drain: An online log parsing approach with fixed depth tree,&#8221; in <i>Proc. IEEE Int. Conf. Web Services (ICWS)</i>, 2017, pp. 33&#8211;40.",
    "W. Meng <i>et al.</i>, &#8220;LogAnomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs,&#8221; in <i>Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)</i>, 2019, pp. 4739&#8211;4745.",
    "X. Zhang <i>et al.</i>, &#8220;Robust log-based anomaly detection on unstable log data,&#8221; in <i>Proc. ACM ESEC/FSE</i>, 2019, pp. 807&#8211;817.",
    "B. Burns, B. Grant, D. Oppenheimer, E. Brewer and J. Wilkes, &#8220;Borg, Omega, and Kubernetes,&#8221; <i>ACM Queue</i>, vol. 14, no. 1, 2016.",
    "Chaos Mesh Authors, &#8220;Chaos Mesh: A chaos engineering platform for Kubernetes,&#8221; Cloud Native Computing Foundation. [Online]. Available: https://chaos-mesh.org",
    "J. Zhu <i>et al.</i>, &#8220;Loghub: A large collection of system log datasets for AI-driven log analytics,&#8221; in <i>Proc. IEEE Int. Symp. Software Reliability Engineering (ISSRE)</i>, 2023.",
    "Google Cloud Platform, &#8220;Online Boutique (microservices-demo),&#8221; GitHub repository. [Online]. Available: https://github.com/GoogleCloudPlatform/microservices-demo",
    "N. Dragoni <i>et al.</i>, &#8220;Microservices: Yesterday, today, and tomorrow,&#8221; in <i>Present and Ulterior Software Engineering</i>, Springer, 2017, pp. 195&#8211;216.",
    "F. T. Liu, K. M. Ting and Z.-H. Zhou, &#8220;Isolation forest,&#8221; in <i>Proc. IEEE Int. Conf. Data Mining (ICDM)</i>, 2008, pp. 413&#8211;422.",
    "J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, &#8220;BERT: Pre-training of deep bidirectional transformers for language understanding,&#8221; in <i>Proc. NAACL-HLT</i>, 2019, pp. 4171&#8211;4186.",
]
for i, r in enumerate(refs, 1):
    S.append(Paragraph(f"[{i}] {r}", ref))

# ---------------------------------------------------------------- build
COMPLETE = not missing and None not in (DET, REACT, PRO)
if not COMPLETE:
    S.insert(note_at, P("<i>Note: entries marked &#8220;pending&#8221; have not been measured yet. This draft is "
                        "provisional until every one of them is replaced with a measured result.</i>", body0))

def footer(c, d):
    c.saveState()
    c.setFont("Times-Roman", 8)
    c.drawCentredString(letter[0] / 2, 0.45 * inch, str(d.page))
    if not COMPLETE:
        c.setFont("Times-Italic", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawCentredString(letter[0] / 2, 0.3 * inch,
                            "Draft — reported values are provisional and pending experimental validation")
    c.restoreState()

doc = BaseDocTemplate(str(OUT),
                      pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                      title="Proactive Microservice Failure Mitigation using LogBERT Anomaly Detection in Kubernetes",
                      author="A. Harshith Saipaavan, B. Mohaseen Hussain, M. Jaidev, P. Manish Reddy")
W, Hh = letter; lm = 0.75 * inch; gut = 0.25 * inch; top = Hh - 0.75 * inch; bot = 0.8 * inch
title_h = 1.7 * inch
f_title = Frame(lm, top - title_h, 7 * inch, title_h, id="t", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
f1 = Frame(lm, bot, 7 * inch, top - title_h - bot, id="1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
f2 = Frame(lm, bot, 7 * inch, top - bot, id="2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="first", frames=[f_title, f1], onPage=footer),
                      PageTemplate(id="later", frames=[f2], onPage=footer)])
doc.build(S)
print(f"built {OUT}" + ("" if COMPLETE else "  (provisional: some results are still pending)"))
