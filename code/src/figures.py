"""Figures for the dribbling paper. Palette validated with the dataviz checker (blue, orange, teal + greys)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Rectangle

R, F = Path("results"), Path("figures"); F.mkdir(exist_ok=True)
BLUE, ORANGE, TEAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#2b2a27", "#8a877f", "#e6e4df"
plt.rcParams.update({"figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7, "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                     "axes.titlesize": 9.5, "axes.titlelocation": "left"})
S = json.load(open(R / "summary.json"))

# ---------------- Figure 1: Messi's career
c = pd.read_csv(R / "messi_career.csv")
c = c[c["attempts"] >= 50].copy()
# match-clustered bootstrap intervals (robustness.py) replace attempt-level ones
cc_ = pd.read_csv(R / "messi_career_clustered.csv")[["competition", "season", "lo", "hi"]].rename(columns={"lo": "oe_lo", "hi": "oe_hi"})
c = c.drop(columns=["oe_lo", "oe_hi"]).merge(cc_, on=["competition", "season"])
c["lab"] = c["season"].str[2:4] + "/" + c["season"].str[7:9] + np.where(c["competition"] == "Ligue 1", "*", "")
x = np.arange(len(c))
f, ax = plt.subplots(1, 3, figsize=(11, 3.2), dpi=220)
ax[0].plot(x, c["att90"], "o-", color=BLUE, lw=2, ms=4, label="attempted")
ax[0].plot(x, c["won90"], "s-", color=ORANGE, lw=2, ms=4, label="completed")
ax[0].set_ylabel("Take-ons per 90"); ax[0].set_title("Volume"); ax[0].legend(frameon=False, fontsize=8)
ax[1].fill_between(x, 100 * c["lo"], 100 * c["hi"], color=BLUE, alpha=0.15, lw=0)
ax[1].plot(x, 100 * c["success"], "o-", color=BLUE, lw=2, ms=4)
ax[1].axhline(100 * S["base_success"], color=MUTED, ls="--", lw=1)
ax[1].text(x[-1], 100 * S["base_success"] + 0.8, "all players", ha="right", color=MUTED, fontsize=7.5)
ax[1].set_ylabel("Success rate (%)"); ax[1].set_title("Completion, 95% interval")
ax[2].fill_between(x, 100 * c["oe_lo"], 100 * c["oe_hi"], color=TEAL, alpha=0.18, lw=0)
ax[2].plot(x, 100 * c["oe"], "o-", color=TEAL, lw=2, ms=4)
ax[2].axhline(0, color=MUTED, lw=1)
ax[2].set_ylabel("Points above expected"); ax[2].set_title("Above context expectation (match bootstrap)")
for a in ax:
    a.set_xticks(x, c["lab"], rotation=60, fontsize=7)
f.text(0.995, 0.01, "* Ligue 1 (PSG). Seasons with ≥50 attempts.", ha="right", fontsize=7, color=MUTED)
f.tight_layout(); f.savefig(F / "fig_career.png"); plt.close(f)

# ---------------- Figure 2: 2015/16 benchmark
b = pd.read_csv(R / "benchmark_2015_16.csv")
q = b.dropna(subset=["oe_shrunk"])
f, ax = plt.subplots(figsize=(6.6, 4.2), dpi=220)
ax.scatter(q["takeons90"], 100 * q["oe_shrunk"], s=10, color="#c9c6be", lw=0, zorder=1)
OFFS = {"Perotti": (4, -10), "Wszołek": (4, 4), "Sabaly": (4, -11), "Zaha": (4, 5), "Carrasco": (-4, -10), "Neymar": (-4, 4)}
lab = q.sort_values("oe_shrunk", ascending=False).head(6)["player_id"].tolist() + \
    q.sort_values("takeons90", ascending=False).head(6)["player_id"].tolist()
short = {"Lionel Andrés Messi Cuccittini": "Messi", "Neymar da Silva Santos Junior": "Neymar",
         "Mousa Sidi Yaya Dembélé": "M. Dembélé", "Francisco Román Alarcón Suárez": "Isco",
         "Adama Traoré Diarra": "Adama Traoré", "Yannick Ferreira Carrasco": "Carrasco",
         "Juan Guillermo Cuadrado Bello": "Cuadrado", "Armindo Tué Na Bangna": "Bruma"}
for _, r in q[q["player_id"].isin(set(lab))].iterrows():
    is_m = r["player_id"] == 5503
    ax.scatter(r["takeons90"], 100 * r["oe_shrunk"], s=46 if is_m else 22, color=ORANGE if is_m else BLUE,
               edgecolor="white", lw=1.2, zorder=3)
    name = short.get(r["player"], r["player"].split()[-1])
    off = OFFS.get(name, (4, 3))
    ax.annotate(name, (r["takeons90"], 100 * r["oe_shrunk"]), xytext=off, textcoords="offset points",
                fontsize=7.5, color=INK, weight="bold" if is_m else "normal", ha="right" if off[0] < 0 else "left")
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("Take-ons attempted per 90"); ax.set_ylabel("Success above expectation (points, shrunk)")
ax.set_title(f"2015/16, four complete leagues: {len(q)} players with ≥30 take-ons")
f.tight_layout(); f.savefig(F / "fig_benchmark.png"); plt.close(f)

# ---------------- Figure 3: measured duel-seeking vs carrying spectrum
sp = pd.read_csv(R / "spectrum_2015_16.csv")
co = pd.read_csv(R / "cohort.csv")
f, ax = plt.subplots(figsize=(6.6, 4.4), dpi=220)
ax.scatter(sp["prog90"], sp["takeons90"], s=8, color="#d5d2ca", lw=0, label="all players 2015/16 (≥900 min)")
TX = []
for _, r in co.iterrows():
    is_m = r["label"] == "Lionel Messi"
    ax.scatter(r["prog90"], r["att90"], s=50 if is_m else 30, color=ORANGE if is_m else BLUE, edgecolor="white", lw=1.2, zorder=3)
    tag = f"{r['label'].split()[-1] if r['label'] != 'Ronaldinho' else 'Ronaldinho'} ({'LL' if r['competition'] == 'La Liga' else 'PL' if r['competition'] == 'Premier League' else 'L1'})"
    TX.append(ax.text(r["prog90"], r["att90"], tag, fontsize=7.2, color=INK, weight="bold" if is_m else "normal"))
from adjustText import adjust_text
adjust_text(TX, x=co["prog90"].values, y=co["att90"].values, ax=ax, expand=(1.3, 1.6),
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.axvline(sp["prog90"].quantile(0.9), color=MUTED, ls=":", lw=1); ax.axhline(sp["takeons90"].quantile(0.9), color=MUTED, ls=":", lw=1)
ax.set_xlabel("Progressive carries per 90 (structured carrying)")
ax.set_ylabel("Take-ons attempted per 90 (duel-seeking)")
ax.set_title("A measured DFD/SED map (dotted: 90th percentiles)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
f.tight_layout(); f.savefig(F / "fig_spectrum.png"); plt.close(f)

# ---------------- Figure 4: where take-ons happen and what follows
dr = pd.read_parquet(R / "takeons.parquet", columns=["player_id", "x", "y", "success", "poss_xg", "competition", "season"])
bench = dr[dr["competition"].isin(["La Liga", "Premier League", "Serie A", "Ligue 1"]) & (dr["season"] == "2015/2016")]
messi = dr[dr["player_id"] == 5503]


def pitch(a):
    a.set_xlim(0, 120); a.set_ylim(0, 80); a.set_aspect("equal"); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks([]); a.set_yticks([])
    kw = dict(fill=False, ec=MUTED, lw=0.8)
    a.add_patch(Rectangle((0, 0), 120, 80, **kw)); a.plot([60, 60], [0, 80], color=MUTED, lw=0.8)
    a.add_patch(Rectangle((102, 18), 18, 44, **kw)); a.add_patch(Rectangle((0, 18), 18, 44, **kw))
    a.add_patch(Arc((60, 40), 20, 20, ec=MUTED, lw=0.8))


f, ax = plt.subplots(1, 3, figsize=(11.5, 3.3), dpi=220, gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
bins = [np.linspace(0, 120, 13), np.linspace(0, 80, 9)]
hb, _, _ = np.histogram2d(bench["x"], bench["y"], bins=bins); hm, _, _ = np.histogram2d(messi["x"], messi["y"], bins=bins)
for a, h, t in [(ax[0], hm / hm.sum(), f"Messi, all seasons ({len(messi):,} take-ons)"),
                (ax[1], hb / hb.sum(), f"All players, 2015/16 ({len(bench):,})")]:
    pitch(a)
    im = a.imshow(h.T, origin="lower", extent=[0, 120, 0, 80], cmap="Blues", vmin=0, vmax=0.06, alpha=0.9)
    a.set_title(t); a.annotate("", xy=(112, -4), xytext=(80, -4), arrowprops=dict(arrowstyle="->", color=MUTED), annotation_clip=False)
f.colorbar(im, ax=ax[:2], shrink=0.8, label="Share of take-ons")
zones = pd.cut(dr["x"], [0, 60, 80, 102, 121], labels=["own half", "60–80", "80–102", "box"])
v = dr.assign(z=zones).groupby(["z", "success"], observed=True)["poss_xg"].mean().unstack()
yy = np.arange(len(v))
ax[2].barh(yy + 0.2, v[1], 0.38, color=TEAL, label="completed"); ax[2].barh(yy - 0.2, v[0], 0.38, color=MUTED, label="failed")
ax[2].set_yticks(yy, v.index); ax[2].set_xlabel("xG later in the same possession")
ax[2].set_title("What a take-on leads to, by start zone"); ax[2].legend(frameon=False, fontsize=7.5)
ax[2].grid(axis="y", visible=False)
f.savefig(F / "fig_pitch.png", bbox_inches="tight"); plt.close(f)

# ---------------- Figure 5: published completion trend vs measured
pub = {"2009/2010": 61, "2010/2011": 59, "2011/2012": 56, "2012/2013": 62, "2013/2014": 59, "2014/2015": 56,
       "2015/2016": 62, "2016/2017": 61, "2017/2018": 72, "2018/2019": 60, "2019/2020": 72}
cc = pd.read_csv(R / "messi_career.csv"); cc = cc[cc["season"].isin(pub) & (cc["competition"] == "La Liga")]
xx = np.arange(len(cc))
f, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
ax.fill_between(xx, 100 * cc["lo"], 100 * cc["hi"], color=BLUE, alpha=0.15, lw=0)
ax.plot(xx, 100 * cc["success"], "o-", color=BLUE, lw=2, ms=4, label="StatsBomb events (this study)")
ax.plot(xx, [pub[s] for s in cc["season"]], "s--", color=ORANGE, lw=1.6, ms=4, label="Previously cited trend (web graphic)")
ax.set_xticks(xx, [s[2:4] + "/" + s[7:9] for s in cc["season"]], fontsize=7.5)
ax.set_ylabel("Take-on success (%)"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.set_title("Same player, different sources: definitions matter")
f.tight_layout(); f.savefig(F / "fig_sources.png"); plt.close(f)
print("ok")

# ---------------- Figure 6: data coverage timeline
mt = pd.read_csv("data/matches.csv")
g = mt.groupby(["competition", "season"]).size().reset_index(name="n")
g["y0"] = g["season"].str[:4].astype(int)
rows = ["La Liga", "Premier League", "Serie A", "Ligue 1", "1. Bundesliga"]
f, ax = plt.subplots(figsize=(7.2, 2.6), dpi=220)
for k, comp in enumerate(rows):
    for _, r in g[g["competition"] == comp].iterrows():
        full = r["n"] >= 300
        ax.barh(k, 0.86, left=r["y0"] + 0.07, height=0.62, color=BLUE if full else ("#9cc3ee" if r["n"] >= 3 else ORANGE),
                edgecolor="white", lw=0.8)
ax.set_yticks(range(len(rows)), rows); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(1972, 2024); ax.set_xticks([1973, 1986, 2004, 2010, 2015, 2021])
ax.set_xticklabels(["73/74", "86/87", "04/05", "10/11", "15/16", "21/22"])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="complete league"), Patch(color="#9cc3ee", label="one club's matches"),
                   Patch(color=ORANGE, label="single match")], frameon=False, fontsize=7, loc="lower left", ncol=3,
          bbox_to_anchor=(0, -0.42))
ax.set_title("What the open data cover (by season start year)")
f.tight_layout(); f.savefig(F / "fig_coverage.png"); plt.close(f)
print("coverage ok")


# ---------------- Figure 7: model validation and ranking uncertainty
cal = pd.read_csv(R / "calibration.csv")
rb = json.load(open(R / "robustness.json"))
q = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"]).reset_index(drop=True)
rng = np.random.default_rng(7)
mi = q.index[q["player_id"] == 5503][0]
dr_ = rng.normal(q["oe_shrunk"].to_numpy()[None, :], q["oe_sd"].to_numpy()[None, :], (20000, len(q)))
r1 = (dr_ > dr_[:, [mi]]).sum(1) + 1
d90 = dr_ * q["takeons90"].to_numpy()[None, :]
r2 = (d90 > d90[:, [mi]]).sum(1) + 1
f, ax = plt.subplots(1, 2, figsize=(10, 3.4), dpi=220)
ax[0].plot([0.3, 0.8], [0.3, 0.8], color=MUTED, lw=1, ls="--")
ax[0].plot(cal["pred"], cal["obs"], "o-", color=BLUE, lw=2, ms=5)
ax[0].set_xlabel("Predicted completion (out-of-fold)"); ax[0].set_ylabel("Observed completion")
ax[0].set_title(f"Context model on unseen matches: AUC {rb['model_fit']['auc_oof']:.3f}")
ax[0].text(0.79, 0.33, "deciles of prediction;\ndashed = perfect calibration", ha="right", fontsize=7, color=MUTED)
bins = np.arange(1, 31)
h1 = np.array([(r1 == k).mean() for k in bins]); h2 = np.array([(r2 == k).mean() for k in bins])
ax[1].bar(bins - 0.2, 100 * h1, 0.4, color=BLUE, label="success above expectation (rate)")
ax[1].bar(bins + 0.2, 100 * h2, 0.4, color=ORANGE, label="extra completions per 90 (rate × volume)")
ax[1].set_xlabel("Messi's rank among 604 players (2015/16)"); ax[1].set_ylabel("Posterior probability (%)")
ax[1].set_title("How certain is 'first'?"); ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_xlim(0.3, 20.7)
ax[1].set_xticks([1, 5, 10, 15, 20])
f.tight_layout(); f.savefig(F / "fig_uncertainty.png"); plt.close(f)
print("uncertainty ok")
