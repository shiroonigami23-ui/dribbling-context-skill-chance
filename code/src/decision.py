"""Analyses that answer the two questions a practitioner asks of a take-on rating.

A. Does it predict?  Rate every player on the first half of the 2015/16 season
   (context model and shrinkage fitted on first-half data only) and score each
   rating on the player's second-half take-ons.  Compared: league average, raw
   success rate, context-adjusted rate without shrinkage, and the shrunk rating.

B. When is a take-on worth it?  For each pitch zone, the net value of a take-on
   (own shot xG later in the possession minus the opponent's shot xG in the next
   possession) after success and after failure, against the same net value for
   an open-play pass from the zone.  The break-even success rate is the rate at
   which attempting the take-on matches the pass on average.  Combined with a
   player's shrunk rating this gives a zone-by-zone "dribble or pass" guide.

Outputs results/decision.json, results/ratings_2015_16.csv,
figures fig_validity.png and fig_breakeven.png.
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

BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280", "#d64545"
F = Path("figures"); F.mkdir(exist_ok=True)
XB = ([0, 40, 60, 80, 102, 121], ["def", "mid1", "mid2", "final", "box"])
YB = ([-1, 18, 30, 50, 62, 81], ["w1", "h1", "c", "h2", "w2"])


def zones(x, y):
    xb = pd.cut(x, XB[0], labels=XB[1]).astype(str)
    yb = pd.cut(y, YB[0], labels=YB[1]).astype(str)
    return xb, yb


def later_own_xg(events, shots):
    """Sum of own-team shot xG later in the same possession, for each row of `events`."""
    d = events[["match_id", "possession", "index", "team"]].assign(rid=np.arange(len(events)))
    j = d.merge(shots, on=["match_id", "possession", "team"], suffixes=("", "_s"))
    j = j[j["index_s"] > j["index"]]
    return j.groupby("rid")["shot_xg"].sum().reindex(np.arange(len(events))).fillna(0).to_numpy()


def next_opp_xg(events, poss_team, poss_xg):
    """Opponent shot xG in the possession that follows (0 if the next possession is the same team's)."""
    nxt = events[["match_id", "possession", "team"]].copy()
    nxt["possession"] = nxt["possession"] + 1
    k = nxt.merge(poss_team, on=["match_id", "possession"], how="left", suffixes=("", "_n"))
    k = k.merge(poss_xg, on=["match_id", "possession", "team_n"], how="left")
    opp = (k["team_n"].notna()) & (k["team_n"] != k["team"])
    return np.where(opp, k["xg_n"].fillna(0), 0.0)


def context_glm(dr):
    cats = ["zone", "state", "pattern", "cs"]
    X = pd.get_dummies(dr[cats], drop_first=True).astype(float)
    X["late"] = dr["late"].astype(float)
    X = sm.add_constant(X)
    glm = sm.GLM(dr["success"], X, family=sm.families.Binomial()).fit()
    return glm, X.columns, cats


def predict(glm, cols, cats, d):
    Z = pd.get_dummies(d[cats], drop_first=False).astype(float)
    Z["late"] = d["late"].astype(float)
    Z = sm.add_constant(Z, has_constant="add").reindex(columns=cols, fill_value=0.0)
    return glm.predict(Z).to_numpy()


def shrink(won, exp, var, n):
    oe = (won - exp) / n
    se2 = var / n ** 2
    tau2 = max(np.var(oe, ddof=1) - se2.mean(), 1e-6)
    mu = oe.mean()
    return mu + tau2 / (tau2 + se2) * (oe - mu), tau2, se2, mu


def main():
    ev, m, mins = load()
    ev["team"] = ev["team"].astype(str)
    shots = ev[ev["type"] == "Shot"][["match_id", "possession", "index", "team", "shot_xg"]]
    first = ev.sort_values(["match_id", "index"]).groupby(["match_id", "possession"], observed=True)["team"].first()
    poss_team = first.rename("team_n").reset_index()
    poss_xg = shots.groupby(["match_id", "possession", "team"])["shot_xg"].sum().rename("xg_n").reset_index() \
        .rename(columns={"team": "team_n"})

    # ---------------- take-ons with context
    dr = ev[ev["type"] == "Dribble"].copy()
    dr["success"] = (dr["outcome"] == "Complete").astype(int)
    dr = add_score_state(dr, ev)
    dr["xb"], dr["yb"] = zones(dr["x"], dr["y"])
    dr["zone"] = dr["xb"] + "_" + dr["yb"]
    dr["state"] = np.sign(dr["score_diff"]).map({-1: "losing", 0: "level", 1: "winning"})
    dr["pattern"] = np.where(dr["play_pattern"].isin(["From Counter", "Regular Play"]), dr["play_pattern"], "Other")
    dr["late"] = (dr["minute"] >= 75).astype(int)
    dr["competition"], dr["season"] = dr["competition"].astype(str), dr["season"].astype(str)
    dr["cs"] = dr["competition"] + " " + dr["season"]
    dr = dr.reset_index(drop=True)
    out = {}

    # ================= A. predictive validity, first half -> second half of 2015/16
    b = dr[dr["competition"].isin(BENCH) & (dr["season"] == "2015/2016")].copy()
    md = m[["match_id", "date"]].assign(date=lambda t: pd.to_datetime(t["date"]))
    b = b.merge(md, on="match_id")
    cut = b.groupby("competition")["date"].transform(lambda s: s.drop_duplicates().median())
    b["half"] = np.where(b["date"] <= cut, 1, 2)
    h1, h2 = b[b["half"] == 1].copy(), b[b["half"] == 2].copy()
    glm1, cols1, cats = context_glm(h1)
    h1["p"] = predict(glm1, cols1, cats, h1)
    h2["p"] = predict(glm1, cols1, cats, h2)          # context model from the first half only
    g1 = h1.groupby("player_id").agg(n=("success", "size"), won=("success", "sum"), exp=("p", "sum"),
                                     var=("p", lambda p: (p * (1 - p)).sum()))
    g2 = h2.groupby("player_id").agg(n2=("success", "size"), won2=("success", "sum"), exp2=("p", "sum"))
    MIN1 = 10
    g = g1[g1["n"] >= MIN1].join(g2, how="inner")
    g["raw"] = g["won"] / g["n"]
    g["oe"] = (g["won"] - g["exp"]) / g["n"]
    g["oe_s"], tau2_1, _, mu1 = shrink(g["won"], g["exp"], g["var"], g["n"])
    base1 = h1["success"].mean()

    h2s = h2[h2["player_id"].isin(g.index)].copy()
    gg = g.reindex(h2s["player_id"])
    preds = {
        "League average (first half)": np.full(len(h2s), base1),
        "Context model only": h2s["p"].to_numpy(),
        "Raw success rate": gg["raw"].to_numpy(),
        "Context + unshrunk player rating": h2s["p"].to_numpy() + gg["oe"].to_numpy(),
        "Context + shrunk player rating": h2s["p"].to_numpy() + gg["oe_s"].to_numpy(),
    }
    y = h2s["success"].to_numpy()

    def scores(p):
        p = np.clip(p, 0.02, 0.98)
        return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "brier": float(np.mean((y - p) ** 2))}
    sc = {k: scores(v) for k, v in preds.items()}
    # player-level: how well each rating predicts second-half success above expectation (weighted by attempts)
    g["oe2"] = (g["won2"] - g["exp2"]) / g["n2"]
    w = g["n2"]

    def wcorr(a, c):
        a, c = a - np.average(a, weights=w), c - np.average(c, weights=w)
        return float(np.sum(w * a * c) / np.sqrt(np.sum(w * a * a) * np.sum(w * c * c)))

    def wmse(pred):
        return float(np.average((g["oe2"] - pred) ** 2, weights=w))
    player_level = {
        "corr_raw": wcorr(g["raw"], g["oe2"]), "corr_oe": wcorr(g["oe"], g["oe2"]),
        "mse_zero": wmse(0 * g["oe"]), "mse_oe": wmse(g["oe"]), "mse_oe_shrunk": wmse(g["oe_s"]),
        "mse_raw_minus_mean": wmse(g["raw"] - base1),
    }
    # bootstrap over players for the log-loss gain of shrunk vs raw and vs context-only
    rng = np.random.default_rng(7)
    pids = g.index.to_numpy()
    idx_by_p = h2s.groupby("player_id").indices
    ll = {k: -(y * np.log(np.clip(v, .02, .98)) + (1 - y) * np.log(1 - np.clip(v, .02, .98))) for k, v in preds.items()}
    gains = {"vs_raw": [], "vs_context": [], "vs_unshrunk": []}
    for _ in range(2000):
        s = rng.choice(pids, len(pids))
        ii = np.concatenate([idx_by_p[p] for p in s])
        L = {k: v[ii].mean() for k, v in ll.items()}
        gains["vs_raw"].append(L["Raw success rate"] - L["Context + shrunk player rating"])
        gains["vs_context"].append(L["Context model only"] - L["Context + shrunk player rating"])
        gains["vs_unshrunk"].append(L["Context + unshrunk player rating"] - L["Context + shrunk player rating"])
    out["validity"] = {
        "players": int(len(g)), "min_first_half_attempts": MIN1,
        "takeons_first_half": int(len(h1)), "takeons_second_half_scored": int(len(h2s)),
        "tau_first_half_pp": float(np.sqrt(tau2_1) * 100),
        "scores": sc, "player_level": player_level,
        "gain_ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in gains.items()},
    }

    # ================= B. dribble or pass: break-even success by zone (2015/16, four leagues)
    b_all = b.copy()
    b_all["own"] = later_own_xg(b_all, shots)
    b_all["opp"] = next_opp_xg(b_all, poss_team, poss_xg)
    b_all["net"] = b_all["own"] - b_all["opp"]

    pa = ev[(ev["type"] == "Pass") & ev["competition"].isin(BENCH) & (ev["season"].astype(str) == "2015/2016")
            & (ev["play_pattern"].isin(["Regular Play", "From Counter"])) & ev["sub_type"].isna()].copy()
    pa = pa[["match_id", "possession", "index", "team", "x", "y", "outcome"]].reset_index(drop=True)
    pa["own"] = later_own_xg(pa, shots)
    pa["opp"] = next_opp_xg(pa, poss_team, poss_xg)
    pa["net"] = pa["own"] - pa["opp"]
    pa["xb"], pa["yb"] = zones(pa["x"], pa["y"])
    pa["zone"] = pa["xb"] + "_" + pa["yb"]

    THIRD = {"def": "defensive", "mid1": "middle", "mid2": "middle", "final": "final", "box": "box"}
    LANE = {"w1": "wide", "w2": "wide", "h1": "half-space", "h2": "half-space", "c": "central"}
    for d_ in (b_all, pa):
        d_["tl"] = d_["xb"].map(THIRD) + " / " + d_["yb"].map(LANE)

    def be_table(td, pd_, w_t=None, w_p=None):
        res = {}
        for z, d in td.groupby("tl"):
            pz = pd_[pd_["tl"] == z]
            wt = np.ones(len(d)) if w_t is None else d["match_id"].map(w_t).fillna(0).to_numpy()
            wp = np.ones(len(pz)) if w_p is None else pz["match_id"].map(w_p).fillna(0).to_numpy()
            ss = d["success"].to_numpy() == 1
            if wt[ss].sum() == 0 or wt[~ss].sum() == 0 or wp.sum() == 0:
                continue
            net = d["net"].to_numpy()
            vs, vf = np.average(net[ss], weights=wt[ss]), np.average(net[~ss], weights=wt[~ss])
            vp = np.average(pz["net"].to_numpy(), weights=wp)
            succ = np.average(ss, weights=wt)
            be = np.clip((vp - vf) / (vs - vf), 0, 1) if vs > vf else np.nan
            res[z] = dict(success=succ, vs=vs, vf=vf, vp=vp, be=be, margin=succ - be,
                          ev_gap=succ * vs + (1 - succ) * vf - vp)
        return res

    base = be_table(b_all, pa)
    matches = b_all["match_id"].unique()
    boot = {z: {"be": [], "margin": [], "ev_gap": []} for z in base}
    for _ in range(400):
        ms = pd.Series(rng.choice(matches, len(matches))).value_counts()
        r_ = be_table(b_all, pa, ms, ms)
        for z, v in r_.items():
            for k in boot[z]:
                boot[z][k].append(v[k])
    rows = []
    for z, v in base.items():
        t, l = z.split(" / ")
        d = b_all[b_all["tl"] == z]
        ci = lambda k: (float(np.nanpercentile(boot[z][k], 5)), float(np.nanpercentile(boot[z][k], 95)))
        rows.append({"third": t, "lane": l, "takeons": int(len(d)), "passes": int((pa["tl"] == z).sum()),
                     "success": v["success"], "net_success": 100 * v["vs"], "net_fail": 100 * v["vf"],
                     "net_pass": 100 * v["vp"],
                     "opp_after_fail": 100 * d.loc[d["success"] == 0, "opp"].mean(),
                     "opp_after_success": 100 * d.loc[d["success"] == 1, "opp"].mean(),
                     "breakeven": v["be"], "be_lo90": ci("be")[0], "be_hi90": ci("be")[1],
                     "margin": v["margin"], "ev_gap": 100 * v["ev_gap"],
                     "ev_gap_lo90": 100 * ci("ev_gap")[0], "ev_gap_hi90": 100 * ci("ev_gap")[1]})
    at = pd.DataFrame(rows)
    order = {"defensive": 0, "middle": 1, "final": 2, "box": 3}
    lorder = {"wide": 0, "half-space": 1, "central": 2}
    at = at.sort_values(["third", "lane"], key=lambda s: s.map(order if s.name == "third" else lorder))

    rt = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"])
    q = {k: float(np.percentile(rt["oe_shrunk"], v)) for k, v in [("p10", 10), ("p50", 50), ("p90", 90)]}
    q["messi"] = float(rt.loc[rt["player_id"] == 5503, "oe_shrunk"].iat[0])
    for k, r in q.items():
        at[f"go_{k}"] = (at["success"] + r) > at["breakeven"]
    at.to_csv(R / "breakeven_third_lane.csv", index=False)
    # risk of losing the ball: opponent xG in the next possession, by third (per 100 take-ons)
    risk = {t: {"after_fail": 100 * d.loc[d["success"] == 0, "opp"].mean(),
                "after_success": 100 * d.loc[d["success"] == 1, "opp"].mean(),
                "after_pass": 100 * pa.loc[pa["tl"].str.startswith(t), "opp"].mean()}
            for t, d in b_all.assign(t=b_all["tl"].str.split(" / ").str[0]).groupby("t")}
    out["breakeven"] = {"takeons": int(len(b_all)), "passes": int(len(pa)), "rating_quantiles": q,
                        "cells_go": {k: int(at[f"go_{k}"].sum()) for k in q}, "cells_total": int(len(at)),
                        "risk_by_third": risk, "third_lane": at.to_dict(orient="records")}

    # ---------------- clean ratings table for release
    s = json.load(open(R / "summary.json"))
    tau2, pbar = s["tau2_oe"], s["base_success"]
    rt["reliability"] = tau2 / (tau2 + pbar * (1 - pbar) / rt["takeons"])
    rt["rating_lo90"] = rt["oe_shrunk"] - 1.645 * rt["oe_sd"]
    rt["rating_hi90"] = rt["oe_shrunk"] + 1.645 * rt["oe_sd"]
    keep = ["player", "competition", "season", "minutes", "takeons", "takeons_won", "succ", "exp_won", "oe",
            "oe_shrunk", "rating_lo90", "rating_hi90", "reliability"]
    rr = rt[keep].rename(columns={"succ": "success_rate", "exp_won": "expected_won", "oe": "above_expected_raw",
                                  "oe_shrunk": "rating"}).sort_values("rating", ascending=False)
    for c in ["success_rate", "above_expected_raw", "rating", "rating_lo90", "rating_hi90"]:
        rr[c] = (100 * rr[c]).round(1)
    rr["reliability"] = rr["reliability"].round(2)
    rr["expected_won"] = rr["expected_won"].round(1)
    rr["minutes"] = rr["minutes"].round(0).astype(int)
    rr.to_csv(R / "ratings_2015_16.csv", index=False)
    out["ratings_released"] = int(len(rr))
    out["ratings_ci_excludes_zero"] = int(((rr["rating_lo90"] > 0) | (rr["rating_hi90"] < 0)).sum())

    json.dump(out, open(R / "decision.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    # ---------------- figure: predictive validity
    names = list(sc)
    vals = [sc[k]["log_loss"] for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cols_ = [GREY, GREY, ORANGE, BLUE, GREEN]
    ax.barh(range(len(names)), vals, color=cols_, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    lo = min(vals) - 0.004
    ax.set_xlim(lo, max(vals) + 0.004)
    ax.set_xlabel("Log loss on second-half take-ons (lower is better)")
    ax.set_title("Which first-half number best predicts second-half take-ons?", loc="left", fontsize=11)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(F / "fig_validity.png", dpi=200); plt.close(fig)

    # ---------------- figure: break-even map on third x lane (wings pooled, drawn on both sides)
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
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal")
    ax.set_xticks([0, 40, 80, 102, 120]); ax.set_yticks([])
    ax.set_xlabel("Distance from own goal line (StatsBomb yards); attack runs left to right")
    ax.set_title("Success rate a take-on needs to match an open-play pass (blue: average beats it)",
                 loc="left", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(F / "fig_breakeven.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
