"""Every figure in the paper, drawn from the files in results/ (no PNGs needed).

The manuscript (index.qmd) calls these functions in its code chunks, as JSDSS requires:
the code that draws each graphic lives with the manuscript and reads the released results.
Each function takes the results directory and returns a matplotlib Figure.

The same figures used to be written as PNGs by the analysis scripts (practice.py, decision.py,
study360.py, decision_bayes.py, figures.py, fig_replication.py, fig_attack_defence.py); the drawing
code is the same, only the data now come from results/ instead of in-memory variables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, TEAL, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#1baf7a", "#6b7280"
INK, MUTED, GRID = "#2b2a27", "#8a877f", "#e6e4df"
MESSI = 5503
STYLE = {"axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "font.size": 9, "axes.edgecolor": MUTED,
         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
         "axes.titlesize": 9.5}


def _clean(ax, keep_grid=False):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if not keep_grid:
        ax.grid(False)


def _load(R, name):
    return json.load(open(Path(R) / name))


# ---------------------------------------------------------------- Study 1

def fig_context(R="results"):
    """Context-adjusted take-on success by pitch zone (from zone_success.csv)."""
    zt = pd.read_csv(Path(R) / "zone_success.csv")
    xs = {"def": (0, 40), "mid1": (40, 60), "mid2": (60, 80), "final": (80, 102), "box": (102, 120)}
    ys = {"w1": (0, 18), "h1": (18, 30), "c": (30, 50), "h2": (50, 62), "w2": (62, 80)}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("Blues")
    lo, hi = zt["adjusted"].min(), zt["adjusted"].max()
    for _, r in zt.iterrows():
        (x0, x1), (y0, y1) = xs[r["xb"]], ys[r["yb"]]
        c = cmap(0.15 + 0.8 * (r["adjusted"] - lo) / (hi - lo))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="white", lw=1.5))
        dark = (r["adjusted"] - lo) / (hi - lo) > 0.55
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.2, f"{100 * r['adjusted']:.0f}%", ha="center", va="center",
                fontsize=10, color="white" if dark else "#1c1d21", weight="bold")
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 2.6, f"n={int(r['n']):,}", ha="center", va="center", fontsize=7,
                color="white" if dark else "#5f6470")
    ax.set_xlim(0, 120), ax.set_ylim(0, 80), ax.set_aspect("equal")
    ax.set_xticks([0, 40, 60, 80, 102, 120]), ax.set_yticks([])
    ax.set_xlabel("Distance from own goal line (StatsBomb yards); attack runs left to right")
    ax.set_title("Context-adjusted take-on success by pitch zone", loc="left", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


def fig_reliability(R="results"):
    """Reliability curve and the distribution of season attempts (summary.json, benchmark_2015_16.csv)."""
    s = _load(R, "summary.json")
    tau2, p = s["tau2_oe"], s["base_success"]
    noise = p * (1 - p)
    need = {r: int(np.ceil(r / (1 - r) * noise / tau2)) for r in (0.5, 0.7)}
    b = pd.read_csv(Path(R) / "benchmark_2015_16.csv")
    b = b[b["takeons"] > 0]
    n = np.arange(5, 601)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(n, tau2 / (tau2 + noise / n), color=BLUE, lw=2)
    for r, col in [(0.5, GREY), (0.7, ORANGE)]:
        k = need[r]
        ax.axhline(r, color=col, lw=0.8, ls="--")
        ax.plot([k], [r], "o", color=col, ms=6, mec="white", mew=1.5)
        ax.text(k + 8, r - 0.045, f"{k} attempts for reliability {r}", fontsize=8.5, color="#1c1d21")
    ax2 = ax.twinx()
    ax2.hist(b["takeons"].clip(upper=600), bins=40, color=GREEN, alpha=0.25)
    ax2.set_yticks([])
    _clean(ax), _clean(ax2)
    ax.set_xlim(0, 600), ax.set_ylim(0, 1)
    ax.set_xlabel("Take-on attempts in the sample")
    ax.set_ylabel("Reliability of the context-adjusted rating")
    ax.text(0.99, 0.04, f"Green: season attempts of {len(b)} players, 2015/16 (median {b['takeons'].median():.0f})",
            transform=ax.transAxes, ha="right", fontsize=8, color="#128a60")
    ax.set_title("A season of take-ons rarely gives a reliable rating", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def fig_validity(R="results"):
    """Out-of-sample log loss of five first-half predictions (decision.json)."""
    sc = _load(R, "decision.json")["validity"]["scores"]
    names = list(sc)
    vals = [sc[k]["log_loss"] for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(range(len(names)), vals, color=[GREY, GREY, ORANGE, BLUE, GREEN], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names))), ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 0.004, max(vals) + 0.004)
    ax.set_xlabel("Log loss on second-half take-ons (lower is better)")
    ax.set_title("Which first-half number best predicts second-half take-ons?", loc="left", fontsize=11)
    _clean(ax)
    fig.tight_layout()
    return fig


def fig_breakeven(R="results"):
    """Break-even map on third x lane (breakeven_third_lane.csv)."""
    at = pd.read_csv(Path(R) / "breakeven_third_lane.csv")
    xs = {"defensive": (0, 40), "middle": (40, 80), "final": (80, 102), "box": (102, 120)}
    ys = {"wide": [(0, 18), (62, 80)], "half-space": [(18, 30), (50, 62)], "central": [(30, 50)]}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("RdBu")
    mx = np.nanmax(np.abs(at["margin"]))
    for _, r in at.iterrows():
        (x0, x1) = xs[r["third"]]
        v = r["margin"]
        c = cmap(0.5 + 0.42 * v / mx)
        tc = "white" if abs(v) / mx > 0.6 else "#1c1d21"
        for (y0, y1) in ys[r["lane"]]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="white", lw=1.5))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.3, f"need {100 * r['breakeven']:.0f}%", ha="center",
                    va="center", fontsize=8.5, color=tc, weight="bold")
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 2.4, f"avg {100 * r['success']:.0f}%", ha="center",
                    va="center", fontsize=7.5, color=tc)
    ax.set_xlim(0, 120), ax.set_ylim(0, 80), ax.set_aspect("equal")
    ax.set_xticks([0, 40, 80, 102, 120]), ax.set_yticks([])
    ax.set_xlabel("Distance from own goal line (StatsBomb yards); attack runs left to right")
    ax.set_title("Success rate a take-on needs to match an open-play pass (blue: average beats it)",
                 loc="left", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


def fig_uncertainty(R="results", draws=20000, seed=7):
    """Left: out-of-fold calibration. Right: posterior rank of the top-rated player (empirical Bayes draws)."""
    with plt.rc_context(STYLE):
        return _fig_uncertainty(R, draws, seed)


def _fig_uncertainty(R, draws, seed):
    R = Path(R)
    cal = pd.read_csv(R / "calibration.csv")
    rb = _load(R, "robustness.json")
    q = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    mi = q.index[q["player_id"] == MESSI][0]
    dr_ = rng.normal(q["oe_shrunk"].to_numpy()[None, :], q["oe_sd"].to_numpy()[None, :], (draws, len(q)))
    r1 = (dr_ > dr_[:, [mi]]).sum(1) + 1
    d90 = dr_ * q["takeons90"].to_numpy()[None, :]
    r2 = (d90 > d90[:, [mi]]).sum(1) + 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot([0.3, 0.8], [0.3, 0.8], color=MUTED, lw=1, ls="--")
    ax[0].plot(cal["pred"], cal["obs"], "o-", color=BLUE, lw=2, ms=5)
    ax[0].set_xlabel("Predicted completion (out-of-fold)"), ax[0].set_ylabel("Observed completion")
    ax[0].set_title(f"Context model on unseen matches: AUC {rb['model_fit']['auc_oof']:.3f}", loc="left", fontsize=9.5)
    ax[0].text(0.79, 0.33, "deciles of prediction;\ndashed = perfect calibration", ha="right", fontsize=7, color=MUTED)
    bins = np.arange(1, 31)
    h1 = np.array([(r1 == k).mean() for k in bins])
    h2 = np.array([(r2 == k).mean() for k in bins])
    ax[1].bar(bins - 0.2, 100 * h1, 0.4, color=BLUE, label="success above expectation (rate)")
    ax[1].bar(bins + 0.2, 100 * h2, 0.4, color=ORANGE, label="extra completions per 90 (rate × volume)")
    ax[1].set_xlabel(f"Messi's rank among {len(q)} players (2015/16)"), ax[1].set_ylabel("Posterior probability (%)")
    ax[1].set_title("How certain is 'first'?", loc="left", fontsize=9.5)
    ax[1].legend(frameon=False, fontsize=7.5), ax[1].set_xlim(0.3, 20.7), ax[1].set_xticks([1, 5, 10, 15, 20])
    for a in ax:
        _clean(a, keep_grid=True)
    fig.tight_layout()
    return fig


def fig_career(R="results"):
    """Messi's career: volume, completion and success above expectation (messi_career*.csv)."""
    with plt.rc_context(STYLE):
        return _fig_career(R)


def _fig_career(R):
    R = Path(R)
    S = _load(R, "summary.json")
    c = pd.read_csv(R / "messi_career.csv")
    c = c[c["attempts"] >= 50].copy()
    cc_ = pd.read_csv(R / "messi_career_clustered.csv")[["competition", "season", "lo", "hi"]] \
        .rename(columns={"lo": "oe_lo", "hi": "oe_hi"})
    c = c.drop(columns=["oe_lo", "oe_hi"]).merge(cc_, on=["competition", "season"])
    c["lab"] = c["season"].str[2:4] + "/" + c["season"].str[7:9] + np.where(c["competition"] == "Ligue 1", "*", "")
    x = np.arange(len(c))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(x, c["att90"], "o-", color=BLUE, lw=2, ms=4, label="attempted")
    ax[0].plot(x, c["won90"], "s-", color=ORANGE, lw=2, ms=4, label="completed")
    ax[0].set_ylabel("Take-ons per 90"), ax[0].set_title("Volume", loc="left"), ax[0].legend(frameon=False, fontsize=8)
    ax[1].fill_between(x, 100 * c["lo"], 100 * c["hi"], color=BLUE, alpha=0.15, lw=0)
    ax[1].plot(x, 100 * c["success"], "o-", color=BLUE, lw=2, ms=4)
    ax[1].axhline(100 * S["base_success"], color=MUTED, ls="--", lw=1)
    ax[1].text(x[-1], 100 * S["base_success"] + 0.8, "all players", ha="right", color=MUTED, fontsize=7.5)
    ax[1].set_ylabel("Success rate (%)"), ax[1].set_title("Completion, 95% interval", loc="left")
    ax[2].fill_between(x, 100 * c["oe_lo"], 100 * c["oe_hi"], color=TEAL, alpha=0.18, lw=0)
    ax[2].plot(x, 100 * c["oe"], "o-", color=TEAL, lw=2, ms=4)
    ax[2].axhline(0, color=MUTED, lw=1)
    ax[2].set_ylabel("Points above expected"), ax[2].set_title("Above context expectation (match bootstrap)", loc="left")
    for a in ax:
        a.set_xticks(x, c["lab"], rotation=60, fontsize=7)
        _clean(a, keep_grid=True)
    fig.text(0.995, 0.01, "* Ligue 1 (PSG). Seasons with ≥50 attempts.", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Studies 2-6

def fig_replication(R="results"):
    """Out-of-sample gain of the shrunk rating in every dataset (decision.json, replication.json)."""
    d1 = _load(R, "decision.json")["validity"]
    rp = _load(R, "replication.json")
    s1 = d1["scores"]
    rows = [("Men, 4 leagues 2015/16\n(half-season)",
             s1["Raw success rate"]["log_loss"] - s1["Context + shrunk player rating"]["log_loss"], d1["gain_ci"]["vs_raw"],
             s1["Context model only"]["log_loss"] - s1["Context + shrunk player rating"]["log_loss"],
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
        _clean(ax)
    axs[0].set_yticks(list(ys)), axs[0].set_yticklabels([f"{r[0]}  (n={r[5]})" for r in rows], fontsize=8)
    fig.suptitle("Out-of-sample prediction of take-on success, 95% intervals (right of zero: shrunk rating better)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_360_context(R="results"):
    """Adjusted success by nearest-defender distance and goal-side defenders (study360.json)."""
    out = _load(R, "study360.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, key, lv, lab in [(axs[0], "adjusted_near", ["<1", "1-2", "2-3", "3+"], "Nearest opponent (yards)"),
                             (axs[1], "adjusted_cone", ["0", "1", "2", "3", "4", "5"], "Opponents goal-side (30° cone)")]:
        v = [100 * out[key][x] for x in lv]
        ax.plot(range(len(lv)), v, "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.5)
        for i, y in enumerate(v):
            ax.text(i, y + 1.2, f"{y:.0f}%", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(lv))), ax.set_xticklabels(lv if key == "adjusted_near" else lv[:-1] + ["5+"])
        ax.set_xlabel(lab, fontsize=9)
        _clean(ax)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axs[0].set_ylabel("Adjusted take-on success (%)")
    fig.suptitle("Defender context from 360 freeze frames", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def fig_360_breakeven(R="results"):
    """Break-even success by whether a forward teammate is open (breakeven_360.csv)."""
    bt = pd.read_csv(Path(R) / "breakeven_360.csv")
    order = ["own half to middle", "final third", "box"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.36
    for j, (o, col) in enumerate([("1+ open forward", GREY), ("none open forward", ORANGE)]):
        sub = bt[bt["open"] == o].set_index("zone").reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * w
        ax.bar(x, 100 * sub["breakeven"], w * 0.92, color=col,
               label="Needed: a forward teammate open" if o.startswith("1+") else "Needed: no forward teammate open")
        ax.errorbar(x, 100 * sub["breakeven"], yerr=[100 * (sub["breakeven"] - sub["be_lo90"]),
                                                     100 * (sub["be_hi90"] - sub["breakeven"])],
                    fmt="none", ecolor="#374151", lw=1, capsize=3)
        ax.plot(x, 100 * sub["success"], "D", color="#111827", ms=5, label="Average take-on success" if j == 0 else None)
    ax.set_xticks(range(len(order))), ax.set_xticklabels(["Own half / middle third", "Final third", "Penalty box"])
    ax.set_ylabel("Success rate (%)"), ax.set_ylim(0, 105)
    _clean(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.set_title("Break-even success vs a pass, by whether a forward teammate is open", loc="left", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_attack_defence(R="results"):
    """SD of take-on success for attackers, individual defenders and defending teams (hierarchical.json)."""
    h = _load(R, "hierarchical.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    labels = ["Attacker", "Individual defender", "Defending team"]
    cols = ["#2a78d6", "#eb6834", "#9ca3af"]
    w = 0.36
    for j, tag in enumerate(["sb", "wy"]):
        k = h[tag]["pbar"] * (1 - h[tag]["pbar"])
        v = [h[tag]["attacker_sd_pp"], h[tag]["defender_sd_pp"], 100 * h[tag]["sd_m2"]["defteam"] * k]
        x = np.arange(3) + (j - 0.5) * w
        ax.bar(x, v, w * 0.92, color=cols, alpha=1.0 if j == 0 else 0.55, edgecolor="white")
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.15, f"{vi:.1f}", ha="center", fontsize=8.5)
            ax.text(xi, -1.1, "SB" if j == 0 else "WY", ha="center", fontsize=7.5, color="#4b5563")
    ax.set_xticks(range(3)), ax.set_xticklabels(labels), ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("SD of take-on success (points)"), ax.set_ylim(0, 11)
    _clean(ax)
    ax.set_title("Who decides a take-on? Crossed random effects (SB: StatsBomb, solid; WY: Wyscout, light)",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_decision_quality(R="results"):
    """Take-on skill vs decision score (decision_quality_2015_16.csv)."""
    q = pd.read_csv(Path(R) / "decision_quality_2015_16.csv")
    big = q[q["attempts"] >= 60]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(100 * big["oe_shrunk"], big["exp_gain_per100"], s=big["attempts"] / 4, color=BLUE, alpha=0.45,
               edgecolor="white", lw=0.5)
    ax.axhline(0, color="#9ca3af", lw=0.8), ax.axvline(0, color="#9ca3af", lw=0.8)
    offs = [(5, 3), (5, 4), (-45, 4), (5, 3)]
    for (_, r), o in zip(big.nlargest(4, "exp_gain_per100").iterrows(), offs):
        lab = "Messi" if r["player_id"] == MESSI else r["player"].split()[-1]
        ax.annotate(lab, (100 * r["oe_shrunk"], r["exp_gain_per100"]), fontsize=8, xytext=o, textcoords="offset points")
    ax.set_xlabel("Take-on skill: shrunk success above expectation (points)")
    ax.set_ylabel("Expected xG vs passing, per 100 take-ons")
    ax.set_title("Skill and choice: does a player's take-on habit beat passing? (2015/16, ≥60 attempts)",
                 loc="left", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    return fig


ALL = {name: f for name, f in globals().items() if name.startswith("fig_")}


def main():
    """Write every figure to figures/ (handy for the README; the manuscript draws them itself)."""
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for name, f in ALL.items():
        fig = f("results")
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
        print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()
