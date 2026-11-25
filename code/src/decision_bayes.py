"""Bayesian dribble-or-pass decisions for individual players, and a decision-quality score.

For player j in area c (third x lane, Study 1, StatsBomb 2015/16 benchmark):
    expected success      s_jc = s_c + theta_j,         theta_j ~ N(mu_j, sd_j^2)    (empirical-Bayes posterior)
    break-even rate       b_c  ~ N(b_hat_c, se_c^2)     (se from the match-resampled 90% interval)
    P(take-on beats pass) = Phi( (s_c + mu_j - b_hat_c) / sqrt(sd_j^2 + se_c^2) )
A decision is "confident" when this probability is above 0.8 (take on) or below 0.2 (pass).
Expected net gain of attempting instead of passing, per 100 attempts:
    100 * [ s_jc (V_s - V_f) + V_f - V_p ].
Decision quality of a player: share of his take-ons made where P(beats pass) < 0.2, i.e. where for him a pass
was better with at least 80% posterior probability; and the expected xG lost per 100 take-ons relative to
passing in those situations.
Outputs results/decision_bayes.json, results/decision_quality_2015_16.csv, figure fig_decision_quality.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from analysis import BENCH, R  # noqa: E402

F = Path("figures"); F.mkdir(exist_ok=True)
THIRD = {"def": "defensive", "mid1": "middle", "mid2": "middle", "final": "final", "box": "box"}
LANE = {"w1": "wide", "w2": "wide", "h1": "half-space", "h2": "half-space", "c": "central"}


def main():
    be = pd.read_csv(R / "breakeven_third_lane.csv")
    be["tl"] = be["third"] + " / " + be["lane"]
    be["se"] = (be["be_hi90"] - be["be_lo90"]) / (2 * 1.645)
    bm = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"])
    tk = pd.read_parquet(R / "takeons.parquet", columns=["player_id", "competition", "season", "zone", "success"])
    tk = tk[tk["competition"].astype(str).isin(BENCH) & (tk["season"].astype(str) == "2015/2016")]
    xb, yb = tk["zone"].str.split("_").str[0], tk["zone"].str.split("_").str[1]
    tk["tl"] = xb.map(THIRD) + " / " + yb.map(LANE)
    # grid of players x areas
    P = bm[["player_id", "player", "takeons", "oe_shrunk", "oe_sd"]].assign(k=1).merge(be.assign(k=1), on="k")
    z = (P["success"] + P["oe_shrunk"] - P["breakeven"]) / np.sqrt(P["oe_sd"] ** 2 + P["se"] ** 2)
    P["p_beats"] = norm.cdf(z)
    P["gain100"] = (P["success"] + P["oe_shrunk"]) * (P["net_success"] - P["net_fail"]) + P["net_fail"] - P["net_pass"]
    P["call"] = np.where(P["p_beats"] > 0.8, "take on", np.where(P["p_beats"] < 0.2, "pass", "uncertain"))
    grid = P.pivot_table(index="tl", columns="call", values="player_id", aggfunc="count", fill_value=0)
    grid = grid.div(grid.sum(1), axis=0)
    # decision quality: where did each player actually take people on?
    cnt = tk.groupby(["player_id", "tl"]).size().rename("att").reset_index()
    Q = cnt.merge(P[["player_id", "tl", "p_beats", "gain100", "call"]], on=["player_id", "tl"])
    q = Q.groupby("player_id").apply(lambda g: pd.Series({
        "attempts": g["att"].sum(),
        "share_pass_better": g.loc[g["call"] == "pass", "att"].sum() / g["att"].sum(),
        "share_take_on_better": g.loc[g["call"] == "take on", "att"].sum() / g["att"].sum(),
        "exp_gain_per100": np.average(g["gain100"], weights=g["att"]),
    }), include_groups=False).reset_index().merge(bm[["player_id", "player", "oe_shrunk"]], on="player_id")
    q["exp_gain_per100"] = q["exp_gain_per100"]          # xG per 100 take-ons vs passing in the same areas
    q.sort_values("exp_gain_per100", ascending=False).to_csv(R / "decision_quality_2015_16.csv", index=False)
    out = {
        "players": int(len(bm)), "areas": int(len(be)),
        "share_calls": P["call"].value_counts(normalize=True).to_dict(),
        "by_area": grid.round(3).to_dict(orient="index"),
        "attempt_weighted_share_pass_better": float((q["share_pass_better"] * q["attempts"]).sum() / q["attempts"].sum()),
        "attempt_weighted_share_take_on_better": float((q["share_take_on_better"] * q["attempts"]).sum() / q["attempts"].sum()),
        "median_player_share_pass_better": float(q["share_pass_better"].median()),
        "corr_skill_vs_gain": float(np.corrcoef(q["oe_shrunk"], q["exp_gain_per100"])[0, 1]),
        "top5_gain": q.nlargest(5, "exp_gain_per100")[["player", "attempts", "exp_gain_per100", "share_pass_better"]].to_dict(orient="records"),
        "bottom5_gain": q.nsmallest(5, "exp_gain_per100")[["player", "attempts", "exp_gain_per100", "share_pass_better"]].to_dict(orient="records"),
        "messi": q[q["player_id"] == 5503][["attempts", "exp_gain_per100", "share_pass_better", "share_take_on_better"]].to_dict(orient="records"),
    }
    # how much of the variation in the decision score is skill vs where players choose to dribble?
    # counterfactual: give every player the average skill (theta = 0) but keep his own area mix
    P0 = be.assign(gain0=be["success"] * (be["net_success"] - be["net_fail"]) + be["net_fail"] - be["net_pass"])
    Q0 = cnt.merge(P0[["tl", "gain0"]], on="tl")
    q0 = Q0.groupby("player_id").apply(lambda g: np.average(g["gain0"], weights=g["att"]), include_groups=False) \
        .rename("gain_mix_only")
    q = q.merge(q0, left_on="player_id", right_index=True)
    out["var_share_area_mix"] = float(np.var(q["gain_mix_only"]) / np.var(q["exp_gain_per100"]))
    out["corr_gain_vs_mix_only"] = float(np.corrcoef(q["gain_mix_only"], q["exp_gain_per100"])[0, 1])
    json.dump(out, open(R / "decision_bayes.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    # figure: skill (x) vs decision score (y), point size = attempts
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    big = q[q["attempts"] >= 60]
    ax.scatter(100 * big["oe_shrunk"], big["exp_gain_per100"], s=big["attempts"] / 4, color="#2a78d6", alpha=0.45,
               edgecolor="white", lw=0.5)
    ax.axhline(0, color="#9ca3af", lw=0.8); ax.axvline(0, color="#9ca3af", lw=0.8)
    offs = [(5, 3), (5, 4), (-45, 4), (5, 3)]
    for (_, r), o in zip(big.nlargest(4, "exp_gain_per100").iterrows(), offs):
        lab = "Messi" if r["player_id"] == 5503 else r["player"].split()[-1]
        ax.annotate(lab, (100 * r["oe_shrunk"], r["exp_gain_per100"]), fontsize=8, xytext=o, textcoords="offset points")
    ax.set_xlabel("Take-on skill: shrunk success above expectation (points)")
    ax.set_ylabel("Expected xG vs passing, per 100 take-ons")
    ax.set_title("Skill and choice: does a player's take-on habit beat passing? (2015/16, ≥60 attempts)",
                 loc="left", fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(F / "fig_decision_quality.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
