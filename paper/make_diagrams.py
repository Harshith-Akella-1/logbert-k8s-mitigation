import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

INK = "#1f2328"
GREY = "#6b7280"
C_OBS = "#e6eef7"; E_OBS = "#3b6ea8"
C_DET = "#fbeadb"; E_DET = "#b8641f"
C_MIT = "#f6e1e1"; E_MIT = "#a83b3b"
C_SVC = "#ffffff"; E_SVC = "#555b63"
C_K8S = "#e9efe3"; E_K8S = "#4f7a3a"
C_AUX = "#f3f1f8"; E_AUX = "#6a5a9a"
KILL = "#c0392b"; OUTG = "#7d3c98"; CPU = "#d68910"


def box(ax, x, y, w, h, title, sub=None, fc="white", ec=INK, fs=7, sfs=5.6, lw=0.7,
        ls="-", r=0.6, bold=True, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    tx = x + w / 2 if align == "center" else x + 0.8
    if sub:
        ax.text(tx, y + h * 0.66, title, ha=align, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)
        ax.text(tx, y + h * 0.28, sub, ha=align, va="center", fontsize=sfs, color="#30343a", zorder=3)
    else:
        ax.text(tx, y + h / 2, title, ha=align, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)


def region(ax, x, y, w, h, label, fc, ec, ls="-", fs=7.2, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=0))
    ax.text(x + 0.9, y + h - 0.9, label, ha="left", va="top", fontsize=fs, fontweight="bold",
            color=ec, zorder=1)


def arrow(ax, pts, color=INK, lw=0.8, ls="-", head=True, z=4, ms=6):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms,
                                    shrinkA=0, shrinkB=0), zorder=z)


def label(ax, x, y, t, fs=5.4, color=GREY, ha="center", rot=0, style="italic", bg=True):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, style=style, rotation=rot,
            zorder=6, bbox=dict(fc="white", ec="none", pad=0.6) if bg else None)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# =====================================================================
# Fig. 1  System architecture
# =====================================================================
def fig_arch():
    fig, ax = plt.subplots(figsize=(7.16, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 81); ax.axis("off")

    # ---------------- cluster side ----------------
    region(ax, 0.5, 0.5, 48.5, 77, "Minikube cluster (1 node, 4 vCPU / 6 GB)",
           "#f7faf5", E_K8S)
    region(ax, 2.3, 27.5, 45, 44.5, "Namespace: Online Boutique v0.10.3", "#ffffff", E_SVC, ls=(0, (4, 2)), fs=6.4)

    S = {}
    def svc(key, cx, cy, name, w=8.6, h=3.6, tags=()):
        box(ax, cx - w / 2, cy - h / 2, w, h, name, fc=C_SVC, ec=E_SVC, fs=5.8, bold=False, r=0.5)
        S[key] = (cx, cy, w, h)
        for i, c in enumerate(tags):
            ax.add_patch(Circle((cx + w / 2 - 0.9 - i * 1.5, cy + h / 2), 0.62, fc=c, ec="white",
                                lw=0.4, zorder=5))

    svc("lg", 9.0, 66.5, "loadgenerator", w=10.5)
    svc("pr", 39.0, 66.5, "prober", w=8.0)
    svc("fe", 24.0, 58.5, "frontend", w=10.5, tags=(CPU,))
    row = 48.5
    svc("ad", 7.5, row, "ad", w=6.4)
    svc("rec", 19.5, row, "recommendation", w=11.6, tags=(OUTG,))
    svc("chk", 36.5, row, "checkout", w=9.0)
    row2 = 37.5
    xs = [(3.0, 7.4, "pc", "product-\ncatalog", (KILL, CPU)), (11.1, 6.0, "cart", "cart", (KILL,)),
          (17.8, 7.0, "cur", "currency", (KILL, CPU)), (25.5, 7.0, "ship", "shipping", (OUTG,)),
          (33.2, 7.0, "pay", "payment", ()), (40.9, 6.0, "em", "email", ())]
    for x, w, k, n, t in xs:
        svc(k, x + w / 2, row2, n, w=w, h=4.6, tags=t)
    svc("redis", 14.1, 30.6, "redis-cart", w=8.6, h=3.0, tags=(OUTG,))

    def top(k): cx, cy, w, h = S[k]; return (cx, cy + h / 2)
    def bot(k): cx, cy, w, h = S[k]; return (cx, cy - h / 2)

    g = dict(color="#9aa3ad", lw=0.45, ms=4, z=1.5)
    # call graph (gRPC), drawn behind the service boxes
    arrow(ax, [(S["lg"][0] + 5.25, 66.5), (S["fe"][0] - 2, 60.3)], **g)
    arrow(ax, [(S["pr"][0] - 4.0, 66.5), (S["fe"][0] + 2, 60.3)], **g)
    for k in ["ad", "rec", "chk", "pc", "cart", "cur", "ship"]:
        arrow(ax, [bot("fe"), top(k)], **g)
    for k in ["pc", "cart", "cur", "ship", "pay", "em"]:
        arrow(ax, [bot("chk"), top(k)], **g)
    arrow(ax, [bot("rec"), top("pc")], **g)
    arrow(ax, [bot("cart"), top("redis")], **g)
    label(ax, 44.0, 58.5, "gRPC / HTTP\ncall graph", fs=5.0)

    # fault legend
    ly = 30.6
    ax.text(23.6, ly, "faults:", fontsize=5.1, va="center", color=GREY, style="italic")
    for x0, c, t in [(28.2, KILL, "pod kill"), (34.4, OUTG, "outage"), (40.0, CPU, "CPU cut")]:
        ax.add_patch(Circle((x0, ly), 0.62, fc=c, ec="white", lw=0.4, zorder=5))
        ax.text(x0 + 1.0, ly, t, fontsize=5.1, va="center", color=INK)

    # control plane + node
    box(ax, 2.3, 15.0, 45, 10.0, "", fc=C_K8S, ec=E_K8S, r=0.6)
    ax.text(24.8, 23.4, "kube-apiserver", ha="center", va="center", fontsize=7, fontweight="bold")
    for i, t in enumerate(["pods/log\n(GET, since)", "deployments/scale\n(PATCH)", "pods\n(DELETE, resize)"]):
        box(ax, 3.6 + i * 14.6, 16.0, 13.6, 5.2, t, fc="white", ec=E_K8S, fs=5.4, bold=False, r=0.4)
    box(ax, 2.3, 2.0, 21.5, 10.5, "kubelet + Docker runtime", "runs containers · stdout to node logs\nliveness / readiness probes",
        fc="white", ec=E_K8S, fs=6.4, sfs=5.2)
    box(ax, 25.8, 2.0, 21.5, 10.5, "Horizontal Pod Autoscaler", "reactive baseline only\n70 % CPU target",
        fc="white", ec=E_K8S, fs=6.4, sfs=5.2, ls=(0, (3, 2)))
    arrow(ax, [(14.0, 27.5), (14.0, 21.2)], color=E_K8S, lw=0.7)
    label(ax, 14.0, 24.6, "stdout", fs=5.0, color=E_K8S)
    arrow(ax, [(13.0, 15.0), (13.0, 12.5)], color=E_K8S, lw=0.6)
    arrow(ax, [(36.5, 15.0), (36.5, 12.5)], color=E_K8S, lw=0.6, ls=(0, (3, 2)))

    # ---------------- host side ----------------
    region(ax, 50.5, 0.5, 49, 77, "Host (i5, 16 GB, RTX 3050)", "#fbfbfc", "#3d4450")

    X0, W = 52.5, 27.0
    stages = [
        ("obs", "Observation", C_OBS, E_OBS, [
            ("Log collector", "polls pods/log every 2 s · tags pod, service"),
            ("Pre-processor", "JSON → text · mask IP / UUID / hex / numbers"),
            ("Drain parser", "sim. 0.5, depth 4 · 48 templates · unseen → UNK"),
            ("Sequencer", "per-service windows, w = 32, stride 8"),
        ]),
        ("det", "Detection (GPU)", C_DET, E_DET, [
            ("LogBERT encoder", "4 layers × 4 heads, d = 256 · 3.20 M params"),
            ("Masked scorer", "mask each position · miss if key not in top-g"),
            ("Verdict", "score > τ  ·  2 consecutive windows"),
        ]),
        ("mit", "Mitigation", C_MIT, E_MIT, [
            ("Fault classifier", "keyword rules: crash / dependency / slow"),
            ("Policy (Table I)", "restart pod · restart caller + scale dep · scale +1"),
            ("Safety guard", "60 s cool-down per service · ≤ 3 actions / 5 min"),
            ("Actuator", "Kubernetes Python client"),
        ]),
    ]
    h, gap, ghead, ggap = 4.1, 1.05, 3.0, 1.4
    y = 73.5
    centers = {}
    for key, gname, fc, ec, items in stages:
        gh = ghead + len(items) * (h + gap) + 0.2
        region(ax, X0 - 0.9, y - gh, W + 1.8, gh, gname, fc, ec, fs=6.4, lw=0.7)
        yy = y - ghead - h
        for t, s in items:
            box(ax, X0, yy, W, h, t, s, fc="white", ec=ec, fs=6.2, sfs=5.0, r=0.5)
            centers[t] = (X0 + W / 2, yy, yy + h)
            yy -= h + gap
        y -= gh + ggap

    names = [t for _, _, _, _, it in stages for t, _ in it]
    for a, b in zip(names[:-1], names[1:]):
        xa, ya0, ya1 = centers[a]; xb, yb0, yb1 = centers[b]
        arrow(ax, [(xa, ya0), (xb, yb1)], lw=0.7, ms=5)

    # side column
    SX, SW = 82.0, 16.6
    tr = centers["LogBERT encoder"]; sc = centers["Masked scorer"]
    box(ax, SX, tr[1] + 2.6, SW, 7.0, "Offline training",
        "normal run only\nMLKP + 0.1 · VHM loss", fc=C_AUX, ec=E_AUX, fs=6.0, sfs=5.0)
    arrow(ax, [(SX, tr[1] + 5.0), (X0 + W, tr[1] + 2.0)], color=E_AUX, lw=0.7, ls=(0, (3, 2)))
    label(ax, SX - 1.3, tr[1] + 4.6, "weights", fs=4.8, color=E_AUX)
    box(ax, SX, sc[1] - 3.6, SW, 6.4, "Threshold selection",
        "validation fault run\nmax-F1 → g = 3, τ = 1/32", fc=C_AUX, ec=E_AUX, fs=6.0, sfs=5.0)
    arrow(ax, [(SX, sc[1] - 0.4), (X0 + W, sc[1] + 1.0)], color=E_AUX, lw=0.7, ls=(0, (3, 2)))

    gd = centers["Safety guard"]
    box(ax, SX, gd[1] - 1.0, SW, 6.6, "Escalation", "budget hit or > 4 windows\n→ alert only",
        fc="#fff6e8", ec="#b07a10", fs=6.0, sfs=4.8)
    arrow(ax, [(X0 + W, gd[1] + 2.0), (SX, gd[1] + 2.0)], color="#b07a10", lw=0.7, ms=5)

    box(ax, SX, 1.8, SW, 6.6, "Fault injector", "experiment harness only\nshuffled 90 s episodes",
        fc="#f2f2f2", ec="#555555", fs=6.0, sfs=4.8, ls=(0, (3, 2)))

    # ---------------- cross-boundary flows ----------------
    col = centers["Log collector"]
    arrow(ax, [(10.4, 21.2), (10.4, 26.3), (1.4, 26.3), (1.4, 79.6), (76.0, 79.6), (76.0, col[2])],
          color=E_OBS, lw=1.1, ms=7)
    label(ax, 25.0, 79.6, "container logs via pods/log (HTTPS, every 2 s)", fs=5.4, color=E_OBS)

    act = centers["Actuator"]
    am = (act[1] + act[2]) / 2
    arrow(ax, [(X0, am), (50.6, am), (50.6, 23.0), (47.3, 23.0)], color=KILL, lw=1.1, ms=7)
    label(ax, 50.6, 15.5, "restart / scale", fs=4.8, color=KILL, rot=90)
    arrow(ax, [(SX + SW / 2, 1.8), (SX + SW / 2, 1.1), (49.6, 1.1), (49.6, 18.6), (47.3, 18.6)],
          color="#555555", lw=0.7, ls=(0, (3, 2)), ms=6)
    label(ax, 66.0, 1.1, "inject faults (experiments)", fs=5.0, color="#555555")

    save(fig, "fig1_architecture")


# =====================================================================
# Fig. 2  LogBERT model, training and scoring
# =====================================================================
def fig_logbert():
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    # input tokens
    toks = ["k$_1$", "k$_2$", "[M]", "k$_4$", "…", "[M]", "k$_{31}$", "k$_{32}$"]
    tx0, tw, ty = 2.0, 3.6, 3.0
    ax.text(2.0, 10.8, "Window of log keys (w = 32, one service)", fontsize=6.4, fontweight="bold", color=INK)
    for i, t in enumerate(toks):
        m = t == "[M]"
        box(ax, tx0 + i * (tw + 0.5), ty, tw, 4.2, t, fc="#ffe9c7" if m else "white",
            ec=E_DET if m else E_SVC, fs=6.0, bold=False, r=0.4)
    label(ax, 18.4, 1.3, "[M] = mask token (15 % in training; one position per pass at inference)", fs=5.0, bg=False)

    # embeddings
    ex = 37.0
    region(ax, ex, 1.0, 15.5, 12.5, "Input embedding", "#ffffff", E_OBS, fs=5.8, lw=0.6)
    for i, t in enumerate(["key", "position", "service"]):
        box(ax, ex + 1.0, 8.0 - i * 3.1, 9.0, 2.5, t, fc=C_OBS, ec=E_OBS, fs=5.6, bold=False, r=0.3)
    ax.add_patch(Circle((ex + 12.8, 5.0), 1.3, fc="white", ec=E_OBS, lw=0.7, zorder=3))
    ax.text(ex + 12.8, 5.0, "+", ha="center", va="center", fontsize=8, zorder=4)
    for i in range(3):
        arrow(ax, [(ex + 10.0, 9.25 - i * 3.1), (ex + 11.6, 5.6 - (i - 1) * 0.6)], lw=0.5, ms=4, color=E_OBS)
    arrow(ax, [(tx0 + 8 * (tw + 0.5), 5.1), (ex, 5.1)], lw=0.8)

    # encoder stack
    ncx, nw = 58.0, 18.0
    for k in range(3, -1, -1):
        off = k * 0.9
        ax.add_patch(FancyBboxPatch((ncx + off, 6.0 + off), nw, 15.0, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=C_DET if k == 0 else "#f6dcc4", ec=E_DET, lw=0.7, zorder=2 + (3 - k) * 0.1))
    box(ax, ncx + 2, 16.0, nw - 4, 3.2, "Multi-head self-attention (4 heads)", fc="white", ec=E_DET, fs=5.2, bold=False, r=0.3)
    box(ax, ncx + 2, 12.2, nw - 4, 2.6, "Add & LayerNorm", fc="white", ec=E_DET, fs=5.2, bold=False, r=0.3)
    box(ax, ncx + 2, 8.6, nw - 4, 2.6, "Feed-forward (d = 256)", fc="white", ec=E_DET, fs=5.2, bold=False, r=0.3)
    ax.text(ncx + nw / 2, 21.9 + 2.7, "Transformer encoder × 4", ha="center", fontsize=6.4, fontweight="bold", color=E_DET)
    ax.text(ncx + nw / 2, 6.9, "bidirectional context", ha="center", fontsize=5.0, style="italic", color=GREY, zorder=4)
    arrow(ax, [(ex + 14.1, 5.0), (ex + 15.5, 5.0), (ex + 15.5, 13.5), (ncx, 13.5)], lw=0.8)

    # heads
    hx = 82.0
    box(ax, hx, 25.0, 17.0, 8.0, "MLKP head", "softmax over 48 + UNK keys\npredict masked key",
        fc="white", ec=E_DET, fs=6.0, sfs=5.0)
    box(ax, hx, 13.5, 17.0, 8.5, "VHM objective", "pull window vector toward\ncentre c (training only)",
        fc=C_AUX, ec=E_AUX, fs=6.0, sfs=5.0, ls=(0, (3, 2)))
    arrow(ax, [(ncx + nw + 2.7, 19.0), (79.5, 19.0), (79.5, 29.0), (hx, 29.0)], lw=0.8)
    arrow(ax, [(79.5, 17.7), (hx, 17.7)], lw=0.7, color=E_AUX, ls=(0, (3, 2)))

    # scoring
    box(ax, hx, 1.0, 17.0, 10.0, "", fc="#fff6e8", ec="#b07a10", fs=6.0)
    ax.text(hx + 8.5, 9.4, "Anomaly score", ha="center", va="center", fontsize=6.0, fontweight="bold", zorder=4)
    ax.text(hx + 8.5, 6.2, "s = (# masked positions whose\ntrue key is not in top-g) / w", ha="center", va="center", fontsize=5.0, zorder=4)
    ax.text(hx + 8.5, 2.6, "anomalous if s > τ", ha="center", va="center", fontsize=5.2, fontweight="bold", zorder=4)
    arrow(ax, [(hx + 3.0, 25.0), (hx + 3.0, 23.6), (80.5, 23.6), (80.5, 6.0), (hx, 6.0)], lw=0.7, color="#b07a10")

    # loss annotation
    ax.text(2.0, 29.5, r"Training on normal windows only:   $\mathcal{L} = \mathcal{L}_{\mathrm{MLKP}} + 0.1\,\mathcal{L}_{\mathrm{VHM}}$",
            ha="left", fontsize=7.2, color=INK)
    ax.text(2.0, 25.0, "Inference: 8 forward passes × 4 masked positions\ncover all 32 positions exactly once",
            ha="left", fontsize=6.0, color=GREY, style="italic")

    save(fig, "fig2_logbert")


# =====================================================================
# Fig. 3  Mitigation controller decision flow
# =====================================================================
def fig_controller():
    fig, ax = plt.subplots(figsize=(3.45, 4.7))
    ax.set_xlim(-1, 51); ax.set_ylim(0, 70); ax.axis("off")

    def dia(cx, cy, w, h, t, fs=5.4):
        ax.add_patch(plt.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                                 fc="#fff6e8", ec="#b07a10", lw=0.7, zorder=2))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=fs, zorder=3)

    box(ax, 12, 64, 26, 4.6, "Window verdict", "s > τ for one service", fc=C_DET, ec=E_DET, fs=6.0, sfs=5.0)
    dia(25, 56.0, 26, 7.0, "2 consecutive\nanomalous windows?")
    arrow(ax, [(25, 64), (25, 59.5)], lw=0.7, ms=5)
    arrow(ax, [(38, 56), (44, 56), (44, 52)], lw=0.6, ms=4, color=GREY)
    ax.text(44.5, 57.0, "no", fontsize=5.0, color=GREY)
    box(ax, 39.5, 47.5, 9.5, 4.4, "wait", fc="white", ec=GREY, fs=5.2, bold=False)
    ax.text(26.0, 51.2, "yes", fontsize=5.0, color=GREY)

    dia(25, 44.0, 28, 7.6, "keyword class of the\nwindow's raw lines")
    arrow(ax, [(25, 52.5), (25, 47.8)], lw=0.7, ms=5)

    cols = [(7.5, "crash", "panic / exception\n/ OOM", "Restart one pod\nof the service"),
            (25.0, "dependency", "timeout / refused\n/ RPC error", "Restart one caller pod;\nscale named dep. +1"),
            (42.5, "slow", "no explicit\nerror", "Scale service\nreplicas +1")]
    for cx, t, s, a in cols:
        arrow(ax, [(25, 40.2), (cx, 37.0)] if cx != 25 else [(25, 40.2), (25, 37.0)], lw=0.6, ms=4)
        box(ax, cx - 7.6, 30.6, 15.2, 6.2, t, s, fc="white", ec=E_MIT, fs=5.6, sfs=4.6)
        box(ax, cx - 7.6, 21.8, 15.2, 6.2, a, fc=C_MIT, ec=E_MIT, fs=4.9, bold=False)
        arrow(ax, [(cx, 30.6), (cx, 28.0)], lw=0.6, ms=4)
        arrow(ax, [(cx, 21.8), (cx, 19.4), (25, 19.4)] if cx != 25 else [(25, 21.8), (25, 19.4)], lw=0.6,
              ms=4, head=(cx == 25))
    ax.text(25, 18.6, "max 5 replicas", ha="center", fontsize=4.6, color=GREY, style="italic", va="top")

    dia(25, 12.0, 30, 7.4, "cool-down 60 s free and\n< 3 actions in last 5 min?")
    arrow(ax, [(25, 17.3), (25, 15.7)], lw=0.7, ms=5)
    box(ax, 1.0, 1.0, 17.0, 5.2, "Actuate via API", fc=C_K8S, ec=E_K8S, fs=5.6)
    box(ax, 32.0, 1.0, 17.0, 5.2, "Alert only", "stop auto actions", fc="#fff6e8", ec="#b07a10", fs=5.6, sfs=4.6)
    arrow(ax, [(10, 12.0), (9.5, 12.0), (9.5, 6.2)], lw=0.6, ms=4)
    ax.text(10.5, 13.0, "yes", fontsize=5.0, color=GREY)
    arrow(ax, [(40, 12.0), (40.5, 12.0), (40.5, 6.2)], lw=0.6, ms=4)
    ax.text(38.0, 13.0, "no", fontsize=5.0, color=GREY)
    arrow(ax, [(18.0, 3.6), (32.0, 3.6)], lw=0.5, ms=4, ls=(0, (2, 1.5)), color="#b07a10")
    ax.text(25.0, 5.0, "still anomalous\n> 4 windows", ha="center", fontsize=4.3, color="#b07a10")

    save(fig, "fig3_controller")


# =====================================================================
# Fig. 4  Experimental protocol
# =====================================================================
def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")

    y, h = 22.0, 5.0
    phases = [
        (1.0, 30.0, "Normal run", "no faults · trains Drain and all detectors", C_OBS, E_OBS),
        (33.0, 32.0, "Validation fault run", "18 episodes · selects g, τ and baseline thresholds", C_AUX, E_AUX),
        (67.0, 32.0, "Test fault run", "18 episodes, new schedule · evaluation only", C_MIT, E_MIT),
    ]
    for x, w, t, s, fc, ec in phases:
        box(ax, x, y, w, h + 1.6, t, s, fc=fc, ec=ec, fs=6.4, sfs=5.0)
    arrow(ax, [(31.0, y + 3.3), (33.0, y + 3.3)], lw=0.7, ms=5)
    arrow(ax, [(65.0, y + 3.3), (67.0, y + 3.3)], lw=0.7, ms=5)
    ax.add_patch(Rectangle((1.0, 18.0), 24.0, 2.2, fc=E_OBS, ec="none", alpha=0.85))
    ax.add_patch(Rectangle((25.0, 18.0), 6.0, 2.2, fc=E_OBS, ec="none", alpha=0.4))
    ax.text(13.0, 19.1, "train 19,635 windows", ha="center", va="center", fontsize=5.0, color="white")
    ax.text(28.0, 16.6, "held-out 4,918", ha="center", va="center", fontsize=4.8, color=E_OBS)

    # episode zoom
    ax.text(50.0, 14.6, "one fault episode (validation and test runs)", ha="center", fontsize=5.4, color=GREY, style="italic")

    t0, scale = 6.0, 88.0 / 300.0
    def X(sec): return t0 + sec * scale
    yb = 7.0
    ax.add_patch(Rectangle((X(0), yb), X(60) - X(0), 3.0, fc="#eef2f6", ec=E_SVC, lw=0.5))
    ax.add_patch(Rectangle((X(60), yb), X(150) - X(60), 3.0, fc="#f3c6c6", ec=E_MIT, lw=0.6))
    ax.add_patch(Rectangle((X(150), yb), X(210) - X(150), 3.0, fc="#e5e5e5", ec=GREY, lw=0.5, hatch="////"))
    ax.add_patch(Rectangle((X(210), yb), X(300) - X(210), 3.0, fc="#eef2f6", ec=E_SVC, lw=0.5))
    ax.text((X(0) + X(60)) / 2, yb + 1.5, "normal traffic", ha="center", va="center", fontsize=5.2)
    ax.text((X(60) + X(150)) / 2, yb + 1.5, "fault active (≈ 90 s)", ha="center", va="center", fontsize=5.4, fontweight="bold")
    ax.text((X(150) + X(210)) / 2, yb + 1.5, "excluded 60 s", ha="center", va="center", fontsize=5.0,
            bbox=dict(fc="white", ec="none", pad=0.4))
    ax.text((X(210) + X(300)) / 2, yb + 1.5, "normal traffic (gap ≈ 2 min)", ha="center", va="center", fontsize=5.2)

    # window ticks
    for i in range(26):
        xx = X(10 + i * 11)
        ax.plot([xx, xx], [yb - 0.4, yb - 1.4], color=GREY, lw=0.4)
    ax.text(X(0), yb - 2.6, "sliding windows (stride 8 keys)", fontsize=4.9, color=GREY, va="center")
    lx = X(60)
    for c, t, x in [(E_MIT, "labelled anomalous if it overlaps the fault and belongs to the faulted service or a (transitive) caller", X(60))]:
        ax.text(X(300), yb - 2.6, t, fontsize=4.9, color=c, va="center", ha="right")
    ax.text(X(300), yb - 4.6, "user-visible failure = failed prober request (3 pages, 0.5 s pause, 3 s timeout)",
            fontsize=4.9, color=GREY, va="center", ha="right")

    save(fig, "fig4_protocol")


fig_arch(); fig_logbert(); fig_controller(); fig_protocol()
print(sorted(os.listdir(OUT)))
