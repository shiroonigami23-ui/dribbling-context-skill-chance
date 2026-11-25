"""Studies 2 and 3: does the Study 1 picture hold in football the method never saw?

Study 2 (women): seven full league seasons (FA WSL x4, NWSL 2023, Liga F, Frauen-Bundesliga, Serie A Women)
plus Women's World Cups 2019/2023 and Women's Euros 2022/2025.  Re-estimates the context model, skill spread,
reliability, the first-half -> second-half test in every full league season, a true season-to-season test
in the WSL (2018/19 -> 2019/20 -> 2020/21), and the dribble-or-pass break-even table.

Study 3 (men, new): World Cups 2018/2022, Euros 2020/2024, Copa America 2024, AFCON 2023, ISL 2021/22 and
Bundesliga 2023/24 (Leverkusen matches).  The Study 1 context model (refitted without the competition term)
is scored on these matches without refitting (transfer), then the same re-estimates are made.

Output: results/replication.json and the tables used in the paper.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from analysis import R, add_score_state, load  # noqa: E402
from decision import later_own_xg, next_opp_xg, shrink, zones  # noqa: E402

D = Path("data/more")
COLS = ["match_id", "index", "period", "minute", "possession", "team", "play_pattern", "player_id", "player", "type",
        "x", "y", "end_x", "end_y", "outcome", "sub_type", "under_pressure", "aerial_won", "goal", "shot_xg"]
THIRD = {"def": "defensive", "mid1": "middle", "mid2": "middle", "final": "final", "box": "box"}
LANE = {"w1": "wide", "w2": "wide", "h1": "half-space", "h2": "half-space", "c": "central"}
LEAGUES_W = ["FA Women's Super League", "NWSL", "Liga F", "Frauen Bundesliga", "Serie A Women"]
FULL_MIN = 100      # a "full league season" in the open data has at least this many matches


def load_more(gender):
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(D / "matches_*.csv")))])
    m = m[m["gender"] == gender]
    tags = {f"{c}_{s}" for c, s in zip(m["comp_id"], m["season_id"])}
    ev = pd.concat([pd.read_parquet(D / "events" / f"{t}.parquet", columns=COLS) for t in sorted(tags)],
                   ignore_index=True)
    for c in ["team", "play_pattern", "player", "type", "outcome", "sub_type"]:
        ev[c] = ev[c].astype("category")
    ev = ev.merge(m[["match_id", "competition", "season"]], on="match_id")
    mins = pd.concat([pd.read_parquet(D / f"minutes_{t}.parquet") for t in sorted(tags)])
    mins = mins.merge(m[["match_id", "competition", "season"]], on="match_id")
    return ev, m, mins


def prep(ev, passes=True):
    """Take-ons with context and net value; open-play passes with net value."""
    ev = ev.copy()
    ev["team"] = ev["team"].astype(str)
    ev["competition"], ev["season"] = ev["competition"].astype(str), ev["season"].astype(str)
    shots = ev[ev["type"] == "Shot"][["match_id", "possession", "index", "team", "shot_xg"]]
    first = ev.sort_values(["match_id", "index"]).groupby(["match_id", "possession"], observed=True)["team"].first()
    poss_team = first.rename("team_n").reset_index()
    poss_xg = shots.groupby(["match_id", "possession", "team"])["shot_xg"].sum().rename("xg_n").reset_index() \
        .rename(columns={"team": "team_n"})
    dr = ev[ev["type"] == "Dribble"].copy()
    dr["success"] = (dr["outcome"] == "Complete").astype(int)
    dr = add_score_state(dr, ev).reset_index(drop=True)
    dr["xb"], dr["yb"] = zones(dr["x"], dr["y"])
    dr["zone"] = dr["xb"] + "_" + dr["yb"]
    dr["state"] = np.sign(dr["score_diff"]).map({-1: "losing", 0: "level", 1: "winning"})
    dr["pattern"] = np.where(dr["play_pattern"].isin(["From Counter", "Regular Play"]), dr["play_pattern"], "Other")
    dr["late"] = (dr["minute"] >= 75).astype(int)
    dr["cs"] = dr["competition"] + " " + dr["season"]
    dr["own"] = later_own_xg(dr, shots)
    dr["opp"] = next_opp_xg(dr, poss_team, poss_xg)
    dr["net"] = dr["own"] - dr["opp"]
    dr["tl"] = dr["xb"].map(THIRD) + " / " + dr["yb"].map(LANE)
    if not passes:
        return dr, None
    pa = ev[(ev["type"] == "Pass") & ev["play_pattern"].isin(["Regular Play", "From Counter"]) & ev["sub_type"].isna()]
    pa = pa[["match_id", "possession", "index", "team", "x", "y", "competition", "season"]].reset_index(drop=True)
    pa["own"] = later_own_xg(pa, shots)
    pa["opp"] = next_opp_xg(pa, poss_team, poss_xg)
    pa["net"] = pa["own"] - pa["opp"]
    xb, yb = zones(pa["x"], pa["y"])
    pa["tl"] = xb.map(THIRD) + " / " + yb.map(LANE)
    return dr, pa


def design(d, cats, cols=None):
    X = pd.get_dummies(d[cats], drop_first=cols is None).astype(float)
    X["late"] = d["late"].astype(float)
    X = sm.add_constant(X, has_constant="add")
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)


def fit(d, cats):
    X = design(d, cats)
    return sm.GLM(d["success"], X, family=sm.families.Binomial()).fit(), X.columns


def cv_auc(d, cats, k=5, seed=1):
    rng = np.random.default_rng(seed)
    mids = np.sort(d["match_id"].unique())     # sorted: folds must not depend on file order
    fold = dict(zip(mids, rng.integers(0, k, len(mids))))
    f = d["match_id"].map(fold).to_numpy()
    p = np.zeros(len(d))
    for i in range(k):
        g, cols = fit(d[f != i], cats)
        p[f == i] = g.predict(design(d[f == i], cats, cols))
    return float(roc_auc_score(d["success"], p)), p


def ratings(dr, mins, min_minutes=900, min_att=30, key=("player_id", "cs")):
    """Season OE per player with EB shrinkage; returns tau2 and the table."""
    key = list(key)
    g = dr.groupby(key).agg(n=("success", "size"), won=("success", "sum"), exp=("p", "sum"),
                            var=("p", lambda p: (p * (1 - p)).sum())).reset_index()
    if min_minutes:
        mm = mins.assign(cs=mins["competition"].astype(str) + " " + mins["season"].astype(str)) \
            .groupby(["player_id", "cs"])["minutes"].sum().rename("minutes").reset_index()
        g = g.merge(mm, on=["player_id", "cs"], how="left")
        g = g[g["minutes"] >= min_minutes]
    g = g[g["n"] >= min_att].copy()
    g["oe_s"], tau2, se2, mu = shrink(g["won"], g["exp"], g["var"], g["n"])
    g["oe"] = (g["won"] - g["exp"]) / g["n"]
    g["sd"] = np.sqrt(tau2 * se2 / (tau2 + se2))
    return g, tau2


def reliability(tau2, pbar):
    noise = pbar * (1 - pbar)
    return {f"R{int(r * 100)}": int(np.ceil(r / (1 - r) * noise / tau2)) for r in (0.5, 0.7, 0.8)}


def breakeven(dr, pa):
    rows = []
    for z, d in dr.groupby("tl"):
        pz = pa[pa["tl"] == z]
        vs, vf, vp = d.loc[d["success"] == 1, "net"].mean(), d.loc[d["success"] == 0, "net"].mean(), pz["net"].mean()
        be = float(np.clip((vp - vf) / (vs - vf), 0, 1)) if vs > vf else np.nan
        rows.append({"tl": z, "takeons": int(len(d)), "success": float(d["success"].mean()), "breakeven": be,
                     "margin": float(d["success"].mean() - be) if np.isfinite(be) else np.nan,
                     "net_success": 100 * vs, "net_fail": 100 * vf, "net_pass": 100 * vp})
    return pd.DataFrame(rows)


def score(y, p):
    p = np.clip(p, 0.02, 0.98)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def halves_test(dr, m, cats, min1=10):
    """First half of a league season -> second half, as in Study 1."""
    md = m[["match_id", "date"]].assign(date=lambda t: pd.to_datetime(t["date"]))
    d = dr.merge(md, on="match_id")
    cut = d.groupby("cs")["date"].transform(lambda s: s.drop_duplicates().median())
    h1, h2 = d[d["date"] <= cut].copy(), d[d["date"] > cut].copy()
    g, cols = fit(h1, cats)
    h1["p"] = g.predict(design(h1, cats, cols)); h2["p"] = g.predict(design(h2, cats, cols))
    return predictive(h1, h2, min1)


def predictive(a, b, min1=10, key="player_id"):
    ga = a.groupby(key).agg(n=("success", "size"), won=("success", "sum"), exp=("p", "sum"),
                            var=("p", lambda p: (p * (1 - p)).sum()))
    ga = ga[ga["n"] >= min1]
    ga["raw"] = ga["won"] / ga["n"]
    ga["oe"] = (ga["won"] - ga["exp"]) / ga["n"]
    ga["oe_s"], tau2, _, _ = shrink(ga["won"], ga["exp"], ga["var"], ga["n"])
    bb = b[b[key].isin(ga.index)]
    gg = ga.reindex(bb[key])
    y = bb["success"].to_numpy()
    preds = {"league_avg": np.full(len(bb), a["success"].mean()), "context": bb["p"].to_numpy(),
             "raw": gg["raw"].to_numpy(), "context_unshrunk": bb["p"].to_numpy() + gg["oe"].to_numpy(),
             "context_shrunk": bb["p"].to_numpy() + gg["oe_s"].to_numpy()}
    ll = {k: score(y, v) for k, v in preds.items()}
    # bootstrap over players: gain of shrunk over raw and over context
    rng = np.random.default_rng(3)
    idx = bb.groupby(key).indices
    pids = np.array(list(idx))
    per = {k: -(y * np.log(np.clip(v, .02, .98)) + (1 - y) * np.log(1 - np.clip(v, .02, .98))) for k, v in preds.items()}
    gr, gc = [], []
    for _ in range(1000):
        ii = np.concatenate([idx[p] for p in rng.choice(pids, len(pids))])
        gr.append(per["raw"][ii].mean() - per["context_shrunk"][ii].mean())
        gc.append(per["context"][ii].mean() - per["context_shrunk"][ii].mean())
    return {"players": int(len(ga)), "takeons_scored": int(len(bb)), "log_loss": ll,
            "gain_vs_raw_ci": [float(np.percentile(gr, 2.5)), float(np.percentile(gr, 97.5))],
            "gain_vs_context_ci": [float(np.percentile(gc, 2.5)), float(np.percentile(gc, 97.5))]}


def season_to_season(dr, cats, pairs, min1=20):
    """Rate on season t (context model fitted on t), predict season t+1 take-ons (context model from t)."""
    out = {}
    for a, b in pairs:
        A, B = dr[dr["cs"] == a].copy(), dr[dr["cs"] == b].copy()
        c2 = [c for c in cats if c != "cs"]
        g, cols = fit(A, c2)
        A["p"] = g.predict(design(A, c2, cols)); B["p"] = g.predict(design(B, c2, cols))
        out[f"{a} -> {b}"] = predictive(A, B, min1)
    return out


def summarise(dr, pa, mins, m, label, leagues=None):
    cats = ["zone", "state", "pattern", "cs"]
    glm, cols = fit(dr, cats)
    dr["p"] = glm.predict(design(dr, cats, cols))
    auc_cv, _ = cv_auc(dr, cats)
    third = dr["xb"].map(THIRD)
    res = {"label": label, "matches": int(dr["match_id"].nunique()), "takeons": int(len(dr)),
           "passes": int(len(pa)), "success": float(dr["success"].mean()), "auc_cv": auc_cv,
           "success_by_third": dr.groupby(third)["success"].mean().to_dict(),
           "success_by_pattern": dr.groupby("pattern")["success"].mean().to_dict(),
           "xg_after": {"completed": float(dr.loc[dr["success"] == 1, "own"].mean()),
                        "failed": float(dr.loc[dr["success"] == 0, "own"].mean())}}
    pool = dr if leagues is None else dr[dr["competition"].isin(leagues)]
    g, tau2 = ratings(pool, mins, min_minutes=900 if leagues is not None else 0,
                      min_att=30 if leagues is not None else 15)
    res.update({"rated_player_seasons": int(len(g)), "tau_pp": float(np.sqrt(tau2) * 100),
                "attempts_needed": reliability(tau2, dr["success"].mean())})
    att = pool.groupby(["player_id", "cs"]).size()
    res["median_attempts_player_season"] = float(att.median())
    rel = tau2 / (tau2 + dr["success"].mean() * (1 - dr["success"].mean()) / att)
    res["share_R70"] = float((rel >= 0.7).mean()); res["share_R50"] = float((rel >= 0.5).mean())
    be = breakeven(dr, pa)
    res["breakeven"] = be.to_dict(orient="records")
    q = np.percentile(g["oe_s"], [50, 90])
    res["rating_p50_p90"] = [float(q[0]), float(q[1])]
    res["cells_go_p50"] = int(((be["success"] + q[0]) > be["breakeven"]).sum())
    res["cells_go_p90"] = int(((be["success"] + q[1]) > be["breakeven"]).sum())
    res["cells_go_max"] = int(((be["success"] + g["oe_s"].max()) > be["breakeven"]).sum())
    return res, g


def main():
    out = {}
    cats = ["zone", "state", "pattern", "cs"]

    # ---------------- Study 2: women
    ev, m, mins = load_more("women")
    dr, pa = prep(ev)
    del ev
    res, g = summarise(dr, pa, mins, m, "women", leagues=LEAGUES_W)
    nm = m.groupby(["competition", "season"]).size()
    full = [f"{c} {s}" for (c, s), n in nm.items() if n >= FULL_MIN and c in LEAGUES_W]
    res["full_league_seasons"] = full
    res["halves"] = halves_test(dr[dr["cs"].isin(full)], m, cats)
    wsl = "FA Women's Super League"
    res["season_to_season"] = season_to_season(dr, cats, [(f"{wsl} 2018/2019", f"{wsl} 2019/2020"),
                                                          (f"{wsl} 2019/2020", f"{wsl} 2020/2021")])
    # pooled WSL season-to-season (stack both pairs)
    res["by_competition"] = dr.groupby("competition").agg(matches=("match_id", "nunique"), takeons=("success", "size"),
                                                          success=("success", "mean")).reset_index() \
        .to_dict(orient="records")
    out["women"] = res
    breakeven(dr, pa).to_csv(R / "breakeven_women.csv", index=False)
    g.to_csv(R / "ratings_women.csv", index=False)
    dr[["match_id", "index", "player_id", "success", "p", "own", "opp", "net", "zone", "tl"]] \
        .to_parquet(R / "takeons_women.parquet", index=False)
    del dr, pa

    # ---------------- Study 3: men, new competitions
    ev1, m1, mins1 = load()
    d1, _ = prep(ev1.assign(competition=ev1["competition"].astype(str), season=ev1["season"].astype(str)), passes=False)
    del ev1
    c2 = ["zone", "state", "pattern"]
    g1, cols1 = fit(d1, c2)
    ev, m, mins = load_more("men")
    dr, pa = prep(ev)
    del ev
    p_tr = g1.predict(design(dr, c2, cols1)).to_numpy()
    y = dr["success"].to_numpy()
    auc_tr = float(roc_auc_score(y, p_tr))
    # calibration slope and intercept of the transferred model
    lp = np.log(p_tr / (1 - p_tr))
    cal = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    dec = pd.qcut(p_tr, 10, labels=False)
    calib = pd.DataFrame({"pred": p_tr, "obs": y, "d": dec}).groupby("d").mean()
    res, g = summarise(dr, pa, mins, m, "men_new", leagues=None)
    res["transfer"] = {"auc_transfer": auc_tr, "auc_refit_cv": res["auc_cv"],
                       "mean_pred": float(p_tr.mean()), "mean_obs": float(y.mean()),
                       "cal_intercept": float(cal.params[0]), "cal_slope": float(cal.params[1]),
                       "max_decile_gap": float((calib["obs"] - calib["pred"]).abs().max())}
    res["by_competition"] = dr.groupby("competition").agg(matches=("match_id", "nunique"), takeons=("success", "size"),
                                                          success=("success", "mean")).reset_index() \
        .to_dict(orient="records")
    isl = dr[dr["competition"] == "Indian Super league"]
    res["halves_isl"] = halves_test(isl, m, cats)
    gi, ti = ratings(isl, mins, 900, 30)
    res["isl_tau_pp"] = float(np.sqrt(ti) * 100); res["isl_rated"] = int(len(gi))
    out["men_new"] = res
    calib.to_csv(R / "transfer_calibration.csv")
    breakeven(dr, pa).to_csv(R / "breakeven_men_new.csv", index=False)
    json.dump(out, open(R / "replication.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
