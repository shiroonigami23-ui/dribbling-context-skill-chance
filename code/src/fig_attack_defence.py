"""Figure: between-player spread of take-on success for attackers, individual defenders and defending teams."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

h = json.load(open("results/hierarchical.json"))
fig, ax = plt.subplots(figsize=(7.2, 3.3))
labels = ["Attacker", "Individual defender", "Defending team"]
cols = ["#2a78d6", "#eb6834", "#9ca3af"]
w = 0.36
for j, (tag, name) in enumerate([("sb", "StatsBomb 2015/16"), ("wy", "Wyscout 2017/18")]):
    k = h[tag]["pbar"] * (1 - h[tag]["pbar"])
    v = [h[tag]["attacker_sd_pp"], h[tag]["defender_sd_pp"], 100 * h[tag]["sd_m2"]["defteam"] * k]
    x = np.arange(3) + (j - 0.5) * w
    bars = ax.bar(x, v, w * 0.92, color=cols, alpha=1.0 if j == 0 else 0.55, edgecolor="white")
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.15, f"{vi:.1f}", ha="center", fontsize=8.5)
    ax.text(x[0], -1.1, "SB" if j == 0 else "WY", ha="center", fontsize=7.5, color="#4b5563")
    ax.text(x[1], -1.1, "SB" if j == 0 else "WY", ha="center", fontsize=7.5, color="#4b5563")
    ax.text(x[2], -1.1, "SB" if j == 0 else "WY", ha="center", fontsize=7.5, color="#4b5563")
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.tick_params(axis="x", pad=14)
ax.set_ylabel("SD of take-on success (points)")
ax.set_ylim(0, 11)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.set_title("Who decides a take-on? Crossed random effects (SB: StatsBomb, solid; WY: Wyscout, light)",
             loc="left", fontsize=9.5)
fig.tight_layout(); Path("figures").mkdir(exist_ok=True); fig.savefig("figures/fig_attack_defence.png", dpi=200)
