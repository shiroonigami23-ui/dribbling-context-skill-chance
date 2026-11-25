"""Dribbling study on StatsBomb open data: definitions, expected threat, context-adjusted success, career, cohorts.

Pitch: StatsBomb 120 x 80, every team attacking towards x = 120. Goal centre (120, 40).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta as beta_dist

D = Path("data"); R = Path("results"); R.mkdir(exist_ok=True)
SEED = 2026
BENCH = {"La Liga", "Premier League", "Serie A", "Ligue 1"}   # complete 2015/16 seasons
MESSI = 5503

COHORT = {  # StatsBomb player ids (checked against the data)
    5503: "Lionel Messi", 25879: "Ronaldinho", 4320: "Neymar", 15516: "Thierry Henry", 3621: "Eden Hazard",
    5477: "Ousmane Dembélé", 2995: "Ángel Di María", 3814: "Riyad Mahrez", 5216: "Andrés Iniesta",
}


def load():
    cols = ["match_id", "index", "period", "minute", "possession", "team", "play_pattern", "player_id", "player", "type",
            "x", "y", "end_x", "end_y", "outcome", "sub_type", "under_pressure", "aerial_won", "goal", "shot_xg"]
    parts = []
    for p in sorted((D / "events").glob("*.parquet")):
        e = pd.read_parquet(p, columns=cols)
        for c in ["team", "play_pattern", "player", "type", "outcome", "sub_type"]:
            e[c] = e[c].astype("category")
        for c in ["x", "y", "end_x", "end_y"]:
            e[c] = e[c].astype("float32")
        parts.append(e)
    ev = pd.concat(parts, ignore_index=True)
    for c in ["team", "play_pattern", "player", "type", "outcome", "sub_type"]:
        ev[c] = ev[c].astype("category")
    m = pd.read_csv(D / "matches.csv")
    ev = ev.merge(m[["match_id", "competition", "season"]].astype({"competition": "category", "season": "category"}),
                  on="match_id")
    mins = pd.read_parquet(D / "minutes.parquet").merge(m[["match_id", "competition", "season"]], on="match_id")
    return ev, m, mins


# ------------------------------------------------------------------ definitions
def dist_goal(x, y):
    return np.hypot(120 - x, 40 - y)


def in_box(x, y):
    return (x >= 102) & (y >= 18) & (y <= 62)


def progressive(x, y, ex, ey):
    """Moves the ball >= 10 yards closer to the goal centre, or into the penalty area from outside;
    actions starting in the team's own 40% (x < 48) are excluded."""
    closer = dist_goal(x, y) - dist_goal(ex, ey)
    return (x >= 48) & ((closer >= 10) | (in_box(ex, ey) & ~in_box(x, y)))


# ------------------------------------------------------------------ expected threat (Singh 2018)
NX, NY = 16, 12


def cell(x, y):
    cx = np.clip((np.asarray(x, float) / 120 * NX).astype(int), 0, NX - 1)
    cy = np.clip((np.asarray(y, float) / 80 * NY).astype(int), 0, NY - 1)
    return cx * NY + cy


def fit_xt(ev, iters=50):
    shots = ev[ev["type"] == "Shot"]
    moves = ev[ev["type"].isin(["Pass", "Carry"]) & ev["x"].notna() & ev["end_x"].notna()]
    moves = moves[~((moves["type"] == "Pass") & moves["sub_type"].isin(["Corner", "Free Kick", "Throw-in", "Kick Off", "Goal Kick"]))]
    ok = moves[(moves["type"] == "Carry") | moves["outcome"].isna()]
    n = NX * NY
    cs, cm = np.bincount(cell(shots["x"], shots["y"]), minlength=n), np.bincount(cell(moves["x"], moves["y"]), minlength=n)
    goals = np.bincount(cell(shots[shots["goal"]]["x"], shots[shots["goal"]]["y"]), minlength=n)
    tot = cs + cm
    s = np.divide(cs, tot, out=np.zeros(n), where=tot > 0); mv = np.divide(cm, tot, out=np.zeros(n), where=tot > 0)
    g = np.divide(goals, cs, out=np.zeros(n), where=cs > 0)
    T = np.zeros((n, n))
    np.add.at(T, (cell(ok["x"], ok["y"]), cell(ok["end_x"], ok["end_y"])), 1)
    T = T / np.maximum(cm, 1)[:, None]          # failed moves leave the chain (value 0)
    xt = np.zeros(n)
    for _ in range(iters):
        xt = s * g + mv * (T @ xt)
    return xt


# ------------------------------------------------------------------ game state
def add_score_state(dr, ev):
    """Goal difference for the acting team before each take-on (goals from shots only)."""
    g = ev[ev["goal"]][["match_id", "index", "team"]].astype({"team": str}).rename(columns={"index": "g_index", "team": "g_team"})
    d = dr[["match_id", "index", "team"]].astype({"team": str}).assign(rid=np.arange(len(dr)))
    j = d.merge(g, on="match_id")
    j = j[j["g_index"] < j["index"]]
    j["d"] = np.where(j["team"] == j["g_team"], 1, -1)
    dr = dr.copy()
    dr["score_diff"] = j.groupby("rid")["d"].sum().reindex(np.arange(len(dr))).fillna(0).to_numpy()
    return dr


# ------------------------------------------------------------------ main
def main():
    ev, m, mins = load()

    bench = ev[ev["competition"].isin(BENCH) & (ev["season"] == "2015/2016")]
    xt = fit_xt(bench)
    np.save(R / "xt_grid.npy", xt.reshape(NX, NY))

    # ---- take-ons with value and context
    dr = ev[ev["type"] == "Dribble"].copy()
    dr["success"] = (dr["outcome"] == "Complete").astype(int)
    dr = add_score_state(dr, ev)
    # value of a take-on: xG of the team's later shots in the same possession (counts what the dribble led to)
    sh = ev[ev["type"] == "Shot"][["match_id", "possession", "index", "team", "shot_xg"]].astype({"team": str})
    d = dr[["match_id", "possession", "index", "team"]].astype({"team": str}).assign(rid=np.arange(len(dr)))
    j = d.merge(sh, on=["match_id", "possession", "team"], suffixes=("", "_s"))
    j = j[j["index_s"] > j["index"]]
    dr["poss_xg"] = j.groupby("rid")["shot_xg"].sum().reindex(np.arange(len(dr))).fillna(0).to_numpy()
    dr["xt_value"] = dr["poss_xg"]          # kept name for downstream code
    dr["zone"] = pd.cut(dr["x"], [0, 40, 60, 80, 102, 121], labels=["def", "mid1", "mid2", "final", "box"]).astype(str) + \
        "_" + pd.cut(dr["y"], [-1, 18, 30, 50, 62, 81], labels=["w1", "h1", "c", "h2", "w2"]).astype(str)
    dr["state"] = np.sign(dr["score_diff"]).map({-1: "losing", 0: "level", 1: "winning"})
    dr["pattern"] = np.where(dr["play_pattern"].isin(["From Counter", "Regular Play"]), dr["play_pattern"], "Other")
    dr["late"] = (dr["minute"] >= 75).astype(int)
    dr["cs"] = dr["competition"].astype(str) + " " + dr["season"].astype(str)

    # ---- expected success model (context only, no player terms)
    X = pd.get_dummies(dr[["zone", "state", "pattern", "cs"]], drop_first=True).astype(float)
    # StatsBomb flags every take-on as under pressure, so that flag carries no information and is not used
    X["late"] = dr["late"].astype(float)
    X = sm.add_constant(X)
    glm = sm.GLM(dr["success"], X, family=sm.families.Binomial()).fit()
    dr["p_exp"] = glm.predict(X)
    coef = {"late": float(glm.params["late"])}
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(dr["success"], dr["p_exp"])

    # ---- minutes and per-90 tables
    pl = mins.groupby(["player_id", "competition", "season"]).agg(minutes=("minutes", "sum"), player=("player", "first"),
                                                                 matches=("match_id", "nunique")).reset_index()

    def per_player(e, keys):
        t = e["type"].astype(str)
        f = pd.DataFrame({k: e[k] for k in keys})
        f["takeons"] = t == "Dribble"
        f["takeons_won"] = f["takeons"] & (e["outcome"].astype(str) == "Complete")
        f["carries"] = t == "Carry"
        f["prog_carries"] = f["carries"] & progressive(e["x"], e["y"], e["end_x"], e["end_y"])
        f["receipts"] = t == "Ball Receipt*"
        f["receipts_up"] = f["receipts"] & e["under_pressure"]
        f["miscontrols"] = t == "Miscontrol"
        f["miscontrols_up"] = f["miscontrols"] & e["under_pressure"]
        car = (t == "Carry") & e["end_x"].notna()
        xs = np.where(car, xt[cell(e["x"].fillna(0), e["y"].fillna(0))], 0)
        xe = np.where(car, xt[cell(e["end_x"].fillna(0), e["end_y"].fillna(0))], 0)
        f["carry_xt"] = np.where(car, xe - xs, 0.0)
        f["aerial_won"] = e["aerial_won"]
        f["aerial_lost"] = (t == "Duel") & (e["sub_type"].astype(str) == "Aerial Lost")
        f["dispossessed"] = t == "Dispossessed"
        f["fouls_won"] = t == "Foul Won"
        out = f.groupby(keys, observed=True).sum()
        ints = [c for c in out.columns if c != "carry_xt"]
        out[ints] = out[ints].astype(int)
        return out.reset_index()

    tab = per_player(ev, ["player_id", "competition", "season"])
    tab[["competition", "season"]] = tab[["competition", "season"]].astype(str)
    pl[["competition", "season"]] = pl[["competition", "season"]].astype(str)
    tab = tab.merge(pl, on=["player_id", "competition", "season"])
    dr[["competition", "season"]] = dr[["competition", "season"]].astype(str)
    val = dr.groupby(["player_id", "competition", "season"]).agg(xt_takeons=("xt_value", "sum"), exp_won=("p_exp", "sum"),
                                                                 var_won=("p_exp", lambda p: (p * (1 - p)).sum())).reset_index()
    tab = tab.merge(val, on=["player_id", "competition", "season"], how="left").fillna({"xt_takeons": 0, "exp_won": 0, "var_won": 0})
    tab["p90"] = 90 / tab["minutes"].clip(lower=1)
    tab.to_csv(R / "player_season_table.csv", index=False)

    # ---- benchmark 2015/16, four complete leagues, >= 900 minutes
    b = tab[tab["competition"].isin(BENCH) & (tab["season"] == "2015/2016") & (tab["minutes"] >= 900)].copy()
    b["takeons90"] = b["takeons"] * b["p90"]; b["won90"] = b["takeons_won"] * b["p90"]
    b["prog90"] = b["prog_carries"] * b["p90"]; b["xt90"] = b["xt_takeons"] * b["p90"]
    b["carry_xt90"] = b["carry_xt"] * b["p90"]
    b["succ"] = b["takeons_won"] / b["takeons"].replace(0, np.nan)
    # context-adjusted success above expectation, empirical-Bayes shrunk (players with >= 30 attempts)
    q = b[b["takeons"] >= 30].copy()
    q["oe"] = (q["takeons_won"] - q["exp_won"]) / q["takeons"]
    q["se2"] = q["var_won"] / q["takeons"] ** 2
    tau2 = max(q["oe"].var() - q["se2"].mean(), 1e-6)
    q["oe_shrunk"] = q["oe"].mean() + tau2 / (tau2 + q["se2"]) * (q["oe"] - q["oe"].mean())
    q["oe_sd"] = np.sqrt(tau2 * q["se2"] / (tau2 + q["se2"]))
    b = b.merge(q[["player_id", "competition", "oe", "oe_shrunk", "oe_sd"]], on=["player_id", "competition"], how="left")
    b.to_csv(R / "benchmark_2015_16.csv", index=False)

    def pct(col, pid=MESSI):
        r = b[b["player_id"] == pid]
        return float((b[col] < r[col].iloc[0]).mean() * 100) if len(r) and col in b else None

    def rank(col, pid=MESSI, pool=b):
        pool = pool.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
        i = pool.index[pool["player_id"] == pid]
        return (int(i[0]) + 1, len(pool)) if len(i) else None

    vs = dr.groupby("success")["poss_xg"].mean()
    summary_value = dict(xg_after_success=float(vs.get(1, np.nan)), xg_after_fail=float(vs.get(0, np.nan)))
    summary = dict(value=summary_value,
        n_events=int(len(ev)), n_matches=int(ev["match_id"].nunique()), n_takeons=int(len(dr)),
        model_auc=float(auc), model_coef=coef, base_success=float(dr["success"].mean()),
        bench_players=int(len(b)), bench_players_oe=int(q.shape[0]), tau2_oe=float(tau2),
        messi_2015_16={k: (float(b.loc[b["player_id"] == MESSI, k].iloc[0]) if (b["player_id"] == MESSI).any() else None)
                       for k in ["minutes", "takeons90", "won90", "succ", "prog90", "xt90", "oe", "oe_shrunk", "oe_sd"]},
        messi_pct={c: pct(c) for c in ["takeons90", "won90", "succ", "prog90", "xt90", "carry_xt90"]},
        messi_rank={c: rank(c) for c in ["takeons90", "won90", "prog90", "xt90", "carry_xt90", "oe_shrunk"]},
    )
    top = {c: b.sort_values(c, ascending=False).head(10)[["player", "competition", "minutes", c]].to_dict("records")
           for c in ["won90", "xt90", "oe_shrunk", "prog90", "carry_xt90"]}
    summary["top10"] = top

    # ---- Messi's career in La Liga (Barcelona matches) + PSG
    mc = tab[tab["player_id"] == MESSI].copy()
    mc = mc[mc["competition"].isin(["La Liga", "Ligue 1"])].sort_values("season")
    mdr = dr[dr["player_id"] == MESSI]
    rows = []
    rng = np.random.default_rng(SEED)
    for (comp, ssn), d in mdr.groupby(["competition", "season"]):
        n, k = len(d), int(d["success"].sum())
        lo, hi = beta_dist.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
        mt = mc[(mc["competition"] == comp) & (mc["season"] == ssn)]
        oe = (k - d["p_exp"].sum()) / n
        boots = [((s := d.sample(n, replace=True, random_state=int(rng.integers(1e9))))["success"].sum() - s["p_exp"].sum()) / n
                 for _ in range(500)]
        rows.append(dict(competition=comp, season=ssn, minutes=float(mt["minutes"].iloc[0]) if len(mt) else np.nan,
                         attempts=n, won=k, success=k / n, lo=lo, hi=hi, oe=oe, oe_lo=np.percentile(boots, 2.5),
                         oe_hi=np.percentile(boots, 97.5), xt=float(d["xt_value"].sum())))
    career = pd.DataFrame(rows)
    career["att90"] = career["attempts"] / career["minutes"] * 90
    career["won90"] = career["won"] / career["minutes"] * 90
    career.to_csv(R / "messi_career.csv", index=False)

    # ---- cross-era cohort (same provider), per player over all of his matches in the data
    co = tab[tab["player_id"].isin(COHORT)].copy()
    co["label"] = co["player_id"].map(COHORT)
    agg = co.groupby(["label", "competition"]).agg(seasons=("season", lambda s: f"{min(s)}–{max(s)}"), minutes=("minutes", "sum"),
                                                   matches=("matches", "sum"), takeons=("takeons", "sum"),
                                                   won=("takeons_won", "sum"), prog=("prog_carries", "sum"),
                                                   xt=("xt_takeons", "sum"), exp_won=("exp_won", "sum"), cxt=("carry_xt", "sum"),
                                                   aw=("aerial_won", "sum"), al=("aerial_lost", "sum"),
                                                   rec=("receipts", "sum"), mis=("miscontrols", "sum")).reset_index()
    agg = agg[agg["minutes"] >= 900]
    for c, src in [("att90", "takeons"), ("won90", "won"), ("prog90", "prog"), ("xt90", "xt"), ("cxt90", "cxt")]:
        agg[c] = agg[src] / agg["minutes"] * 90
    agg["succ"] = agg["won"] / agg["takeons"]; agg["oe"] = (agg["won"] - agg["exp_won"]) / agg["takeons"]
    agg["mis100"] = 100 * agg["mis"] / agg["rec"]
    agg.to_csv(R / "cohort.csv", index=False)

    # single historical matches (anecdotal)
    hist = []
    for comp, ssn, who in [("La Liga", "1973/1974", "Johan Cruyff"), ("Serie A", "1986/1987", "Diego Armando Maradona")]:
        e = ev[(ev["competition"] == comp) & (ev["season"] == ssn) & ev["player"].str.contains(who.split()[-1], na=False)]
        mm = mins[(mins["competition"] == comp) & (mins["season"] == ssn) & mins["player"].str.contains(who.split()[-1], na=False)]
        hist.append(dict(player=who, season=ssn, minutes=float(mm["minutes"].sum()),
                         takeons=int((e["type"] == "Dribble").sum()),
                         won=int(((e["type"] == "Dribble") & (e["outcome"] == "Complete")).sum()),
                         carries=int((e["type"] == "Carry").sum())))
    pd.DataFrame(hist).to_csv(R / "historical_single_matches.csv", index=False)

    # ---- duel-seeking vs structured carrying (measured version of the DFD/SED spectrum)
    b["dfd_share"] = b["takeons"] / (b["takeons"] + b["prog_carries"]).replace(0, np.nan)
    b[["player_id", "player", "competition", "minutes", "takeons90", "prog90", "dfd_share", "succ", "xt90", "carry_xt90"]].to_csv(R / "spectrum_2015_16.csv", index=False)

    # ---- first touch and aerials, Messi vs attacking players in the benchmark
    att = b[b["prog90"] > b["prog90"].median()]
    summary["first_touch"] = dict(
        messi_mis100=float(100 * b.loc[b.player_id == MESSI, "miscontrols"].iloc[0] / b.loc[b.player_id == MESSI, "receipts"].iloc[0]),
        messi_mis100_up=float(100 * b.loc[b.player_id == MESSI, "miscontrols_up"].iloc[0] / max(b.loc[b.player_id == MESSI, "receipts_up"].iloc[0], 1)),
        pool_mis100_median=float((100 * att["miscontrols"] / att["receipts"]).median()))
    summary["aerials"] = dict(
        messi_aerials90=float((b.loc[b.player_id == MESSI, ["aerial_won", "aerial_lost"]].sum(axis=1) * b.loc[b.player_id == MESSI, "p90"]).iloc[0]),
        messi_aerial_win=float(b.loc[b.player_id == MESSI, "aerial_won"].iloc[0] / max(b.loc[b.player_id == MESSI, ["aerial_won", "aerial_lost"]].sum(axis=1).iloc[0], 1)),
        pool_aerials90_median=float(((att["aerial_won"] + att["aerial_lost"]) * att["p90"]).median()),
        pool_aerial_win_median=float((att["aerial_won"] / (att["aerial_won"] + att["aerial_lost"]).replace(0, np.nan)).median()))
    dr.to_parquet(R / "takeons.parquet", index=False)
    json.dump(summary, open(R / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "top10"}, indent=1, default=float))
    for c, t in top.items():
        print(c, [(r["player"], round(r[c], 3)) for r in t])
    print(career.round(3).to_string())
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
