"""Study 4: what the defenders around the ball add, from StatsBomb 360 freeze frames.

415 matches (men: World Cup 2022, Euros 2020/2024, Bundesliga 2023/24, La Liga 2020/21, Ligue 1 2021/22-2022/23;
women: Women's World Cup 2023, Women's Euros 2022/2025).  For every take-on and open-play pass the freeze frame
gives: distance to the nearest opponent, opponents within 3 and 5 yards, opponents goal-side in a 30-degree cone
towards goal, and open teammates (no opponent within 5 yards, within 35 yards of the ball; "forward" if ahead).

Questions:
 (a) How much better does defender context predict take-on success than the event-only context model?
 (b) How much of the between-player spread survives once defender context is in the model?  (Tests whether
     the Study 1 skill spread was inflated by unmodelled context.)
 (c) Is a take-on worth more than a pass when no teammate is open?  Break-even rates conditioned on the number
     of open forward teammates (answers the main limitation of the Study 1 break-even analysis).
Output: results/study360.json, figures fig_360_context.png and fig_360_breakeven.png.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from analysis import R  # noqa: E402
from decision import shrink  # noqa: E402
from replicate import COLS, THIRD, cv_auc, design, fit, prep  # noqa: E402

F = Path("figures"); F.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"


def load360():
    ff = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/more/ff/*.parquet"))], ignore_index=True)
    ff = ff.sort_values(["match_id", "id"], kind="stable").reset_index(drop=True)   # row order of downloads varies
    mids = set(ff["match_id"])
    parts, metas = [], []
    # Study 1 matches with 360 (events in data/events) and the new matches (data/more/events)
    m1 = pd.read_csv("data/matches.csv").assign(gender="men")
    mm = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/more/matches_*.csv"))])
    meta = pd.concat([m1, mm])[["match_id", "competition", "season", "gender", "date"]].drop_duplicates("match_id")
    meta = meta[meta["match_id"].isin(mids)]
    for p in sorted(Path("data/events").glob("*.parquet")) + sorted(Path("data/more/events").glob("*.parquet")):
        e = pd.read_parquet(p, columns=COLS + ["id"])
        e = e[e["match_id"].isin(mids)]
        if len(e):
            parts.append(e)
    ev = pd.concat(parts, ignore_index=True).merge(meta[["match_id", "competition", "season"]], on="match_id")
    for c in ["team", "play_pattern", "player", "type", "outcome", "sub_type"]:
        ev[c] = ev[c].astype("category")
    return ev, ff, meta


def main():
    ev, ff, meta = load360()
    ids = ev[["match_id", "index", "id"]]
    dr, pa = prep(ev)
    dr = dr.merge(ff.drop(columns=["type"]), on=["match_id", "id"], how="inner")
    pa = pa.merge(ids, on=["match_id", "index"]).merge(ff.drop(columns=["type"]), on=["match_id", "id"], how="inner")
    dr = dr.merge(meta[["match_id", "gender"]], on="match_id")
    # keep frames where the area around the ball is visible (at least 3 players of each side in view)
    dr = dr[(dr["n_opp"] >= 3) & dr["d_near"].notna()].reset_index(drop=True)
    pa = pa[(pa["n_opp"] >= 3)].reset_index(drop=True)
    out = {"matches": int(dr["match_id"].nunique()), "takeons": int(len(dr)), "passes": int(len(pa)),
           "takeons_by_gender": dr["gender"].value_counts().to_dict()}

    # ---- defender features (categorical bins, so the GLM needs no functional-form choice)
    def feats(d):
        d = d.copy()
        d["near_bin"] = pd.cut(d["d_near"], [-1, 1, 2, 3, 200], labels=["<1", "1-2", "2-3", "3+"]).astype(str)
        d["opp3_c"] = d["opp3"].clip(upper=2).astype(str)
        d["cone_c"] = d["opp_cone"].clip(upper=5).astype(str)
        d["open_c"] = d["open_fwd"].clip(upper=3).astype(str)
        return d
    dr = feats(dr)
    base = ["zone", "state", "pattern", "cs"]
    rich = base + ["near_bin", "opp3_c", "cone_c", "open_c"]
    auc_b, pb = cv_auc(dr, base)
    auc_r, pr = cv_auc(dr, rich)
    gb, colsb = fit(dr, base)
    gr, colsr = fit(dr, rich)
    dr["p_base"], dr["p_rich"] = pb, pr
    out["auc_cv"] = {"event_context": auc_b, "with_defenders": auc_r}
    out["brier"] = {"event_context": float(np.mean((dr["success"] - pb) ** 2)),
                    "with_defenders": float(np.mean((dr["success"] - pr) ** 2)),
                    "constant": float(np.var(dr["success"]))}
    # descriptive effects: success by nearest-defender distance and goal-side defenders (observed)
    out["success_by_near"] = dr.groupby("near_bin")["success"].agg(["mean", "size"]).to_dict(orient="index")
    out["success_by_cone"] = dr.groupby("cone_c")["success"].agg(["mean", "size"]).to_dict(orient="index")
    out["success_by_open"] = dr.groupby("open_c")["success"].agg(["mean", "size"]).to_dict(orient="index")
    # adjusted (marginal) effects from the rich model
    def marginal(var, level):
        z = dr[rich].copy(); z[var] = level
        return float(gr.predict(design(z.assign(late=dr["late"]), rich, colsr)).mean())
    out["adjusted_near"] = {lv: marginal("near_bin", lv) for lv in ["<1", "1-2", "2-3", "3+"]}
    out["adjusted_cone"] = {lv: marginal("cone_c", lv) for lv in ["0", "1", "2", "3", "4", "5"]}
    out["adjusted_open"] = {lv: marginal("open_c", lv) for lv in ["0", "1", "2", "3"]}

    # ---- (b) skill spread with and without defender context, same take-ons, players pooled within gender
    def tau(pcol, min_att):
        g = dr.groupby(["player_id", "gender"]).agg(n=("success", "size"), won=("success", "sum"), exp=(pcol, "sum"),
                                                     var=(pcol, lambda p: (p * (1 - p)).sum()))
        g = g[g["n"] >= min_att]
        _, t2, _, _ = shrink(g["won"], g["exp"], g["var"], g["n"])
        oe = (g["won"] - g["exp"]) / g["n"]
        return float(np.sqrt(t2) * 100), int(len(g)), oe
    tb = {}
    for k in (15, 20, 30):
        t_b, nb, oe_b = tau("p_base", k)
        t_r, nr, oe_r = tau("p_rich", k)
        tb[k] = {"players": nb, "tau_event": t_b, "tau_defenders": t_r,
                 "share_variance_remaining": (t_r / t_b) ** 2 if t_b > 0 else None,
                 "corr_ratings": float(np.corrcoef(oe_b, oe_r)[0, 1])}
    out["tau"] = tb
    # bootstrap the remaining share (players resampled) at min 20 attempts
    g = dr.groupby(["player_id", "gender"])
    agg = g.agg(n=("success", "size"), won=("success", "sum"), eb=("p_base", "sum"), er=("p_rich", "sum"),
                vb=("p_base", lambda p: (p * (1 - p)).sum()), vr=("p_rich", lambda p: (p * (1 - p)).sum()))
    agg = agg[agg["n"] >= 20]
    rng = np.random.default_rng(11)
    shares = []
    for _ in range(2000):
        s = agg.iloc[rng.integers(0, len(agg), len(agg))]
        _, t2b, _, _ = shrink(s["won"], s["eb"], s["vb"], s["n"])
        _, t2r, _, _ = shrink(s["won"], s["er"], s["vr"], s["n"])
        if t2b > 1e-5:
            shares.append(t2r / t2b)
    out["share_ci"] = [float(np.percentile(shares, 5)), float(np.percentile(shares, 95))]

    # ---- (c) break-even conditioned on open forward teammates
    dr["open"] = np.where(dr["open_fwd"] >= 1, "1+ open forward", "none open forward")
    pa["open"] = np.where(pa["open_fwd"] >= 1, "1+ open forward", "none open forward")
    third = lambda xs: np.where(xs < 80, "own half to middle", np.where(xs < 102, "final third", "box"))
    dr["zone3"], pa["zone3"] = third(dr["x"]), third(pa["x"])
    rows = []
    for (z, o), d in dr.groupby(["zone3", "open"]):
        pz = pa[(pa["zone3"] == z) & (pa["open"] == o)]
        vs, vf, vp = d.loc[d["success"] == 1, "net"].mean(), d.loc[d["success"] == 0, "net"].mean(), pz["net"].mean()
        be = float(np.clip((vp - vf) / (vs - vf), 0, 1)) if vs > vf else np.nan
        # bootstrap by match
        mids = np.sort(d["match_id"].unique())
        bes = []
        for _ in range(400):
            ms = pd.Series(rng.choice(mids, len(mids))).value_counts()
            wd, wp = d["match_id"].map(ms).fillna(0).to_numpy(), pz["match_id"].map(ms).fillna(0).to_numpy()
            ss = d["success"].to_numpy() == 1
            if wd[ss].sum() == 0 or wd[~ss].sum() == 0 or wp.sum() == 0:
                continue
            a = np.average(d["net"].to_numpy()[ss], weights=wd[ss])
            b = np.average(d["net"].to_numpy()[~ss], weights=wd[~ss])
            c = np.average(pz["net"].to_numpy(), weights=wp)
            if a > b:
                bes.append(np.clip((c - b) / (a - b), 0, 1))
        rows.append({"zone": z, "open": o, "takeons": int(len(d)), "passes": int(len(pz)),
                     "share_takeons": float(len(d) / max(len(d) + len(pz), 1)),
                     "success": float(d["success"].mean()), "net_success": 100 * vs, "net_fail": 100 * vf,
                     "net_pass": 100 * vp, "breakeven": be,
                     "be_lo90": float(np.percentile(bes, 5)), "be_hi90": float(np.percentile(bes, 95))})
    bt = pd.DataFrame(rows)
    bt.to_csv(R / "breakeven_360.csv", index=False)
    out["breakeven_open"] = bt.to_dict(orient="records")
    json.dump(out, open(R / "study360.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    # ---- figure: adjusted success by nearest-defender distance and by goal-side defenders
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, key, lv, lab in [(axs[0], "adjusted_near", ["<1", "1-2", "2-3", "3+"], "Nearest opponent (yards)"),
                             (axs[1], "adjusted_cone", ["0", "1", "2", "3", "4", "5"], "Opponents goal-side (30° cone)")]:
        v = [100 * out[key][x] for x in lv]
        ax.plot(range(len(lv)), v, "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.5)
        for i, y in enumerate(v):
            ax.text(i, y + 1.2, f"{y:.0f}%", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv if key == "adjusted_near" else lv[:-1] + ["5+"])
        ax.set_xlabel(lab, fontsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axs[0].set_ylabel("Adjusted take-on success (%)")
    fig.suptitle("Defender context from 360 freeze frames", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(F / "fig_360_context.png", dpi=200); plt.close(fig)

    # ---- figure: break-even by open forward teammates
    order = ["own half to middle", "final third", "box"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.36
    for j, (o, col) in enumerate([("1+ open forward", GREY), ("none open forward", ORANGE)]):
        sub = bt[bt["open"] == o].set_index("zone").reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * w
        ax.bar(x, 100 * sub["breakeven"], w * 0.92, color=col, label="Needed: a forward teammate open" if o.startswith("1+") else "Needed: no forward teammate open")
        ax.errorbar(x, 100 * sub["breakeven"], yerr=[100 * (sub["breakeven"] - sub["be_lo90"]),
                                                     100 * (sub["be_hi90"] - sub["breakeven"])],
                    fmt="none", ecolor="#374151", lw=1, capsize=3)
        ax.plot(x, 100 * sub["success"], "D", color="#111827", ms=5, label="Average take-on success" if j == 0 else None)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["Own half / middle third", "Final third", "Penalty box"]); fig.set_size_inches(7.2, 3.8)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.set_title("Break-even success vs a pass, by whether a forward teammate is open", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(F / "fig_360_breakeven.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
