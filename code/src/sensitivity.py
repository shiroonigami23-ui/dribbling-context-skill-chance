"""Does Messi's 2015/16 context-adjusted take-on rank survive different modelling choices?"""
import json
import numpy as np, pandas as pd, statsmodels.api as sm

dr = pd.read_parquet("results/takeons.parquet")
mins = pd.read_parquet("data/minutes.parquet").merge(pd.read_csv("data/matches.csv")[["match_id", "competition", "season"]], on="match_id")
BENCH = {"La Liga", "Premier League", "Serie A", "Ligue 1"}


def run(drop=(), min_att=30, leagues=BENCH, interactions=False):
    X = pd.get_dummies(dr[["zone", "state", "pattern", "cs"]], drop_first=True).astype(float)
    if "late" not in drop:
        X["late"] = dr["late"].astype(float)
    if interactions:   # zone x game-state interaction
        for st in ["losing", "winning"]:
            X[f"final_x_{st}"] = (dr["zone"].str.startswith(("final", "box")) & (dr["state"] == st)).astype(float)
    if "zone" in drop:
        X = X[[c for c in X.columns if not c.startswith("zone_")]]
    p = sm.GLM(dr["success"], sm.add_constant(X), family=sm.families.Binomial()).fit().predict(sm.add_constant(X))
    d = dr.assign(p=p)
    d = d[d["competition"].isin(leagues) & (d["season"] == "2015/2016")]
    g = d.groupby("player_id").agg(n=("success", "size"), k=("success", "sum"), e=("p", "sum"), v=("p", lambda q: (q * (1 - q)).sum()))
    mm = mins[mins["competition"].isin(leagues) & (mins["season"] == "2015/2016")].groupby("player_id")["minutes"].sum()
    g = g.join(mm).query("minutes >= 900 and n >= @min_att")
    g["oe"] = (g["k"] - g["e"]) / g["n"]; g["se2"] = g["v"] / g["n"] ** 2
    t2 = max(g["oe"].var() - g["se2"].mean(), 1e-6)
    g["shr"] = g["oe"].mean() + t2 / (t2 + g["se2"]) * (g["oe"] - g["oe"].mean())
    g = g.sort_values("shr", ascending=False)
    r = int(np.where(g.index == 5503)[0][0]) + 1 if 5503 in g.index else None
    return dict(rank=r, pool=len(g), messi=float(g.loc[5503, "shr"]) if 5503 in g.index else None, second=float(g["shr"].iloc[1]))


res = {"main": run(), "no game-time term": run(drop=("late",)), "no zone terms": run(drop=("zone",)),
       "final third x game state": run(interactions=True), "min 50 attempts": run(min_att=50), "min 100 attempts": run(min_att=100),
       "La Liga only": run(leagues={"La Liga"}), "Premier League + Serie A + Ligue 1 pool, Messi excluded": None}
res.pop("Premier League + Serie A + Ligue 1 pool, Messi excluded")
json.dump(res, open("results/sensitivity.json", "w"), indent=1)
for k, v in res.items():
    print(f"{k:28s} rank {v['rank']}/{v['pool']}  Messi {v['messi']:.3f}  next best {v['second']:.3f}")
