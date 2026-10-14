"""Paper Fig. 6: LogBERT precision, recall and F1 on the test run against the anomaly-score threshold.
Reads results/detection.json (tau_sweep and the validation-chosen tau); writes a vector PDF and a 300-dpi PNG.

    python results/figures/make_fig6.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
det = json.loads((HERE.parent / "detection.json").read_text())["methods"]["logbert"]
sweep, tau = det["tau_sweep"], det["tau"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
                     "pdf.fonttype": 42, "font.size": 8})
x = [s["tau"] for s in sweep]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.plot(x, [s["precision"] for s in sweep], "o-", ms=3, lw=1.1, color="#7f9bb8", label="Precision", zorder=3)
ax.plot(x, [s["recall"] for s in sweep], "s-", ms=3, lw=1.1, color="#6a8f4e", label="Recall", zorder=3)
ax.plot(x, [s["f1"] for s in sweep], "^-", ms=3.5, lw=1.4, color="#c0562b", label="F1", zorder=3)
ax.axvline(tau, ls="--", lw=0.7, color="#777777", zorder=2)
ax.set_xlabel("Anomaly-score threshold")
ax.set_ylabel("Score (test run)")
ax.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax.xaxis.grid(True, lw=0.4, color="#dddddd", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#888888")
ax.tick_params(colors="#30343a", length=2)
ax.legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.6)
fig.savefig(HERE / "fig6_threshold.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(HERE / "fig6_threshold.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(f"wrote {HERE / 'fig6_threshold.pdf'} and .png (tau = {tau:.5f}, {len(sweep)} sweep points)")
