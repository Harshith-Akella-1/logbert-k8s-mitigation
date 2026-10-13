import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"Liberation Serif","pdf.fonttype":42,"font.size":8})
faults=["Pod kill","Outage","CPU throttling"]
data={"Isolation Forest":[0.269,0.224,0.161],"DeepLog":[0.905,0.885,0.577],"LogBERT":[0.930,0.934,0.612]}
cols={"Isolation Forest":"#6f9a3a","DeepLog":"#3f6fb0","LogBERT":"#c0562b"}
x=np.arange(3); w=0.26
fig,ax=plt.subplots(figsize=(3.45,2.25))
for i,(k,v) in enumerate(data.items()):
    b=ax.bar(x+(i-1)*w, v, w-0.03, color=cols[k], label=k, zorder=3)
    for xi,vi in zip(x+(i-1)*w, v):
        ax.text(xi, vi+0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.8, color="#30343a")
ax.set_xticks(x); ax.set_xticklabels(faults)
ax.set_ylim(0,1.08); ax.set_ylabel("F1-score (test run)")
ax.yaxis.grid(True, lw=0.4, color="#dddddd", zorder=0); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#888888"); ax.spines["bottom"].set_color("#888888")
ax.tick_params(colors="#30343a", length=2)
ax.legend(ncol=3, frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5,1.14), handlelength=1.2, columnspacing=1.0)
fig.savefig("ieee/fig5_f1_per_fault.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/fig5_f1_per_fault.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
