"""Practice-facing analyses for the revision.

1. Where take-ons succeed: model-adjusted success by pitch zone, play pattern and
   score state (marginal predictions from the context GLM).
2. How many attempts a take-on rating needs before it says more about the player
   than about chance (reliability from the empirical Bayes variance components).
3. What a completed take-on is worth: shot value later in the same possession,
   completed vs failed, by pitch third.
Outputs results/practice.json, results/zone_success.csv, figures fig_context.png
and fig_reliability.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from analysis import BENCH, R, add_score_state, load  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"
F = Path("figures"); F.mkdir(exist_ok=True)


def main():
    ev, m, mins = load()
    dr = ev[ev["type"] == "Dribble"].copy()
    dr["success"] = (dr["outcome"] == "Complete").astype(int)
    dr = add_score_state(dr, ev)
    sh = ev[ev["type"] == "Shot"][["match_id", "possession", "index", "team", "shot_xg"]].astype({"team": str})
    d = dr[["match_id", "possession", "index", "team"]].astype({"team": str}).assign(rid=np.arange(len(dr)))
    j = d.merge(sh, on=["match_id", "possession", "team"], suffixes=("", "_s"))
    j = j[j["index_s"] > j["index"]]
    dr["poss_xg"] = j.groupby("rid")["shot_xg"].sum().reindex(np.arange(len(dr))).fillna(0).to_numpy()
    xb = pd.cut(dr["x"], [0, 40, 60, 80, 102, 121], labels=["def", "mid1", "mid2", "final", "box"]).astype(str)
    yb = pd.cut(dr["y"], [-1, 18, 30, 50, 62, 81], labels=["w1", "h1", "c", "h2", "w2"]).astype(str)
    dr["xb"], dr["yb"] = xb, yb
    dr["zone"] = xb + "_" + yb
    dr["state"] = np.sign(dr["score_diff"]).map({-1: "losing", 0: "level", 1: "winning"})
    dr["pattern"] = np.where(dr["play_pattern"].isin(["From Counter", "Regular Play"]), dr["play_pattern"], "Other")
    dr["late"] = (dr["minute"] >= 75).astype(int)
    dr["cs"] = dr["competition"].astype(str) + " " + dr["season"].astype(str)

    cats = ["zone", "state", "pattern", "cs"]
    X = pd.get_dummies(dr[cats], drop_first=True).astype(float)
    X["late"] = dr["late"].astype(float)
    X = sm.add_constant(X)
    glm = sm.GLM(dr["success"], X, family=sm.families.Binomial()).fit()
    cols = X.columns

    def marginal(var, level):
        """Average predicted success if every take-on had `var` = level, other context as observed."""
        z = dr[cats].copy()
        z[var] = level
        Z = pd.get_dummies(z, drop_first=False).astype(float)
        Z["late"] = dr["late"].astype(float)
        Z = sm.add_constant(Z, has_constant="add").reindex(columns=cols, fill_value=0.0)
        return float(glm.predict(Z).mean())

    out = {"n_takeons": int(len(dr)), "base_success": float(dr["success"].mean())}

    # 1. context: zones (5 x 5 grid), pattern, state
    zrows = []
    for zz, g in dr.groupby("zone"):
        zrows.append({"zone": zz, "xb": g["xb"].iat[0], "yb": g["yb"].iat[0], "n": int(len(g)),
                      "observed": float(g["success"].mean()), "adjusted": marginal("zone", zz),
                      "xg_after_success": float(g.loc[g["success"] == 1, "poss_xg"].mean()),
                      "xg_after_fail": float(g.loc[g["success"] == 0, "poss_xg"].mean())})
    zt = pd.DataFrame(zrows)
    zt.to_csv(R / "zone_success.csv", index=False)
    out["pattern"] = {p: {"n": int((dr["pattern"] == p).sum()), "adjusted": marginal("pattern", p)}
                      for p in ["From Counter", "Regular Play", "Other"]}
    out["state"] = {s: {"n": int((dr["state"] == s).sum()), "adjusted": marginal("state", s)}
                    for s in ["losing", "level", "winning"]}
    lanes = dr.assign(lane=np.where(dr["yb"].isin(["w1", "w2"]), "wide", np.where(dr["yb"] == "c", "central", "half-space")))
    thirds = np.where(dr["x"] < 40, "defensive third", np.where(dr["x"] < 80, "middle third", "final third"))
    lanes["third"] = thirds
    out["lane_by_third"] = {f"{t} / {l}": {"n": int(len(g)), "observed": float(g["success"].mean()),
                                            "xg_after_success": float(g.loc[g["success"] == 1, "poss_xg"].mean()),
                                            "xg_after_fail": float(g.loc[g["success"] == 0, "poss_xg"].mean())}
                            for (t, l), g in lanes.groupby(["third", "lane"])}

    # 2. reliability of a season's context-adjusted rating
    s = json.load(open(R / "summary.json"))
    tau2, p = s["tau2_oe"], s["base_success"]
    noise = p * (1 - p)
    need = {f"R{int(r * 100)}": int(np.ceil(r / (1 - r) * noise / tau2)) for r in (0.5, 0.7, 0.8)}
    b = pd.read_csv(R / "benchmark_2015_16.csv")
    b = b[b["takeons"] > 0]
    rel = tau2 / (tau2 + noise / b["takeons"])
    out["reliability"] = {
        "tau_pp": float(np.sqrt(tau2) * 100), "attempts_needed": need,
        "players": int(len(b)),
        "median_attempts_season": float(b["takeons"].median()),
        "median_takeons90": float(b["takeons90"].median()),
        "share_R50": float((rel >= 0.5).mean()), "share_R70": float((rel >= 0.7).mean()),
        "messi_attempts": int(b.loc[b["player_id"] == 5503, "takeons"].iat[0]),
    }
    med90 = b["takeons90"].median()
    out["reliability"]["matches_needed_median_player_R70"] = float(need["R70"] / med90)
    out["reliability"]["matches_needed_at_4_per90_R70"] = float(need["R70"] / 4)
    json.dump(out, open(R / "practice.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    # ---- figure: adjusted success by zone (pitch heat map)
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
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 2.6, f"n={r['n']:,}", ha="center", va="center", fontsize=7,
                color="white" if dark else "#5f6470")
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal")
    ax.set_xticks([0, 40, 60, 80, 102, 120]); ax.set_yticks([])
    ax.set_xlabel("Distance from own goal line (StatsBomb yards); attack runs left to right")
    ax.set_title("Context-adjusted take-on success by pitch zone", loc="left", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "fig_context.png", dpi=200)
    plt.close(fig)

    # ---- figure: reliability curve with the distribution of season attempts
    n = np.arange(5, 601)
    curve = tau2 / (tau2 + noise / n)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(n, curve, color=BLUE, lw=2)
    for r, col in [(0.5, GREY), (0.7, ORANGE)]:
        k = need[f"R{int(r * 100)}"]
        ax.axhline(r, color=col, lw=0.8, ls="--")
        ax.plot([k], [r], "o", color=col, ms=6, mec="white", mew=1.5)
        ax.text(k + 8, r - 0.045, f"{k} attempts for reliability {r}", fontsize=8.5, color="#1c1d21")
    ax2 = ax.twinx()
    ax2.hist(b["takeons"].clip(upper=600), bins=40, color=GREEN, alpha=0.25)
    ax2.set_yticks([])
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
    ax.set_xlim(0, 600); ax.set_ylim(0, 1)
    ax.set_xlabel("Take-on attempts in the sample")
    ax.set_ylabel("Reliability of the context-adjusted rating")
    ax.text(0.99, 0.04, f"Green: season attempts of {len(b)} players, 2015/16 (median {b['takeons'].median():.0f})",
            transform=ax.transAxes, ha="right", fontsize=8, color="#128a60")
    ax.set_title("A season of take-ons rarely gives a reliable rating", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(F / "fig_reliability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
