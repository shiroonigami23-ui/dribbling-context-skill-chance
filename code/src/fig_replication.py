"""Figure: out-of-sample gain of the shrunk, context-adjusted rating across every dataset."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = Path("results"), Path("figures"); F.mkdir(exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"
d1 = json.load(open(R / "decision.json"))["validity"]
rp = json.load(open(R / "replication.json"))
s1 = d1["scores"]
rows = [("Men, 4 leagues 2015/16\n(half-season)", s1["Raw success rate"]["log_loss"] - s1["Context + shrunk player rating"]["log_loss"],
         d1["gain_ci"]["vs_raw"], s1["Context model only"]["log_loss"] - s1["Context + shrunk player rating"]["log_loss"],
         d1["gain_ci"]["vs_context"], d1["players"])]


def add(label, t):
    ll = t["log_loss"]
    rows.append((label, ll["raw"] - ll["context_shrunk"], t["gain_vs_raw_ci"], ll["context"] - ll["context_shrunk"],
                 t["gain_vs_context_ci"], t["players"]))


add("Women, 7 league seasons\n(half-season)", rp["women"]["halves"])
for k, t in rp["women"]["season_to_season"].items():
    a, b = [x.split()[-1] for x in k.split(" -> ")]
    add(f"WSL {a[2:4]}/{a[-2:]} → {b[2:4]}/{b[-2:]}\n(next season)", t)
add("Men, ISL 2021/22\n(half-season)", rp["men_new"]["halves_isl"])

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
ys = range(len(rows))[::-1]
for ax, j, col, title in [(axs[0], 1, ORANGE, "vs the player's raw completion rate"),
                          (axs[1], 3, BLUE, "vs the context model alone")]:
    for y, r in zip(ys, rows):
        lo, hi = r[j + 1]
        ax.plot([1000 * lo, 1000 * hi], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(1000 * r[j], y, "o", color=col, ms=6, mec="white", mew=1.4)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel("Log-loss gain of the shrunk rating (×1000)", fontsize=8.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axs[0].set_yticks(list(ys)); axs[0].set_yticklabels([f"{r[0]}  (n={r[5]})" for r in rows], fontsize=8)
fig.suptitle("Out-of-sample prediction of take-on success, 95% intervals (right of zero: shrunk rating better)",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(F / "fig_replication.png", dpi=200); plt.close(fig)
for r in rows:
    print(r)
