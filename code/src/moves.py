"""Hand-sketched schematics of three signature moves (same visual language as Figure 1)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = Path("figures"); F.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#4a87d6", "#f07f35", "#c9c9c9"
plt.xkcd(scale=0.6, length=120, randomness=2)
plt.rcParams.update({"font.family": "DejaVu Sans", "path.effects": []})


def arrow(a, pts, num=None, off=(0, 0.35)):
    pts = np.array(pts, float)
    a.plot(pts[:, 0], pts[:, 1], ls=(0, (3, 2)), color=BLUE, lw=2.2)
    a.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.2, mutation_scale=16))
    if num:
        m = pts[len(pts) // 2] if len(off) == 2 else np.array(off[1:])
        if len(off) == 3: off = (0, 0)
        a.text(m[0] + off[0], m[1] + off[1], num, fontsize=15, ha="center", va="center", color="#222")


def player(a, xy, c, r=0.32):
    a.add_patch(plt.Circle(xy, r, color=c, ec="#222", lw=1.4, zorder=5))


def base(a, title):
    a.set_xlim(0, 8); a.set_ylim(0, 6); a.set_aspect("equal"); a.axis("off")
    a.plot([0.2, 7.8, 7.8, 0.2, 0.2], [0.2, 0.2, 5.8, 5.8, 0.2], color=GREY, lw=1.2)
    a.text(4, 6.35, title, ha="center", fontsize=13, color=BLUE, weight="bold")
    a.annotate("", xy=(7.5, 0.55), xytext=(6.2, 0.55), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    a.text(6.85, 0.8, "attack", fontsize=8, color="#888", ha="center")


f, ax = plt.subplots(1, 3, figsize=(13, 3.9), dpi=220)
# Maradona turn: approach, drag back, spin with back to defender, drag back again, exit on defender's side
a = ax[0]; base(a, "Maradona turn"); player(a, (4.6, 3.0), ORANGE); player(a, (2.2, 3.0), BLUE)
arrow(a, [(2.6, 3.0), (3.5, 3.0), (3.9, 3.05)], "1", (0, 0.4))
arrow(a, [(3.9, 3.05), (3.6, 2.6), (3.25, 2.45)], "2", (-0.35, -0.3))
t = np.linspace(np.pi * 0.95, np.pi * 2.6, 30)
arrow(a, np.c_[3.6 + 0.45 * np.cos(t), 2.15 + 0.45 * np.sin(t)].tolist(), "3", ("abs", 2.85, 1.75))
arrow(a, [(4.0, 2.0), (4.8, 1.7), (5.9, 1.9), (6.9, 2.4)], "4", (0, -0.4))
# Elastico: small push outside, instant hook inside, burst past
a = ax[1]; base(a, "Elástico"); player(a, (5.0, 3.0), ORANGE); player(a, (1.2, 3.0), BLUE)
arrow(a, [(1.6, 3.0), (2.6, 3.0), (3.4, 3.0)], "1", ("abs", 2.5, 3.45))
arrow(a, [(3.4, 3.0), (3.75, 2.55), (3.95, 2.1)], "2", ("abs", 4.3, 2.35))
arrow(a, [(3.95, 2.1), (3.6, 2.7), (3.45, 3.4), (3.55, 4.0)], "3", ("abs", 3.05, 3.75))
arrow(a, [(3.55, 4.0), (4.6, 4.5), (5.8, 4.5), (6.9, 4.1)], "4", ("abs", 5.3, 4.95))
a.text(2.45, 1.55, "push out (2), hook back\ninside (3), burst past (4)", fontsize=8, color="#666", ha="center")
# Inside cut at pace: run down the wing, sharp cut inside, accelerate towards goal
a = ax[2]; base(a, "Inside cut at pace"); player(a, (5.1, 1.75), ORANGE); player(a, (0.9, 0.9), BLUE)
arrow(a, [(1.3, 0.9), (2.8, 0.95), (4.3, 1.0)], "1", (0, 0.4))
arrow(a, [(4.3, 1.0), (4.55, 1.35), (4.6, 1.8), (4.5, 2.3)], "2", (-0.45, 0))
arrow(a, [(4.5, 2.3), (5.2, 3.0), (6.2, 3.4), (7.2, 3.5)], "3", (0, 0.4))
f.text(0.5, 0.01, "Blue: attacker with the ball (dashed line = ball path, numbered in order). Orange: defender.",
       ha="center", fontsize=9, color="#555")
f.tight_layout(rect=(0, 0.04, 1, 1)); f.savefig(F / "fig_moves.png"); plt.close(f)
print("ok")
