"""Robustness checks: out-of-fold context model, match-clustered uncertainty, rank stability, reliability.

Reads results/takeons.parquet and results/benchmark_2015_16.csv (written by analysis.py).
Writes results/robustness.json, results/calibration.csv, results/messi_career_clustered.csv, results/rank_boot.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

R = Path("results")
MESSI, SEED, B = 5503, 7, 2000
BENCH = ["La Liga", "Premier League", "Serie A", "Ligue 1"]
rng = np.random.default_rng(SEED)

dr = pd.read_parquet(R / "takeons.parquet")
dr[["competition", "season"]] = dr[["competition", "season"]].astype(str)


def design(d):
    X = pd.get_dummies(d[["zone", "state", "pattern", "cs"]].astype(str), drop_first=False).astype(float)
    X["late"] = d["late"].astype(float)
    return X


# ---------- 1. out-of-fold context model (5 folds grouped by match)
X = design(dr)
X = X.drop(columns=[X.filter(like="zone_").columns[0], X.filter(like="state_").columns[0],
                    X.filter(like="pattern_").columns[0], X.filter(like="cs_").columns[0]])
X = sm.add_constant(X)
y = dr["success"].to_numpy()
p_oof = np.zeros(len(dr))
for tr, te in GroupKFold(5).split(X, y, groups=dr["match_id"]):
    keep = X.columns[(X.iloc[tr].sum() > 0)]
    m = sm.GLM(y[tr], X.iloc[tr][keep], family=sm.families.Binomial()).fit()
    p_oof[te] = m.predict(X.iloc[te][keep])
dr["p_oof"] = p_oof
cal = (dr.assign(bin=pd.qcut(dr["p_oof"], 10, labels=False))
       .groupby("bin").agg(pred=("p_oof", "mean"), obs=("success", "mean"), n=("success", "size")).reset_index())
cal.to_csv(R / "calibration.csv", index=False)
fit = dict(auc_in=roc_auc_score(y, dr["p_exp"]), auc_oof=roc_auc_score(y, p_oof),
           brier_in=brier_score_loss(y, dr["p_exp"]), brier_oof=brier_score_loss(y, p_oof),
           brier_base=brier_score_loss(y, np.full(len(y), y.mean())),
           max_cal_gap=float((cal["obs"] - cal["pred"]).abs().max()))

# ---------- benchmark pool (same players as the paper: >= 900 min, >= 30 take-ons)
b = pd.read_csv(R / "benchmark_2015_16.csv")
pool = b.loc[b["takeons"] >= 30, ["player_id", "competition", "player"]]
bd = dr[(dr["season"] == "2015/2016") & dr["competition"].isin(BENCH)].merge(pool[["player_id", "competition"]])


def shrink(won, exp, var, n):
    oe, se2 = (won - exp) / n, var / n ** 2
    tau2 = max(np.var(oe, ddof=1) - se2.mean(), 1e-6)
    return oe.mean() + tau2 / (tau2 + se2) * (oe - oe.mean()), oe


def table(d, pcol):
    g = d.groupby(["player_id", "competition"]).agg(won=("success", "sum"), exp=(pcol, "sum"),
                                                    var=(pcol, lambda p: (p * (1 - p)).sum()), n=("success", "size")).reset_index()
    g["shr"], g["oe"] = shrink(g["won"], g["exp"], g["var"], g["n"])
    return g.sort_values("shr", ascending=False).reset_index(drop=True)


t_in, t_oof = table(bd, "p_exp"), table(bd, "p_oof")
messi_oof = t_oof.index[t_oof["player_id"] == MESSI][0] + 1
oof = dict(rank=int(messi_oof), pool=len(t_oof), messi=float(t_oof.loc[t_oof["player_id"] == MESSI, "shr"].iloc[0]),
           second=float(t_oof.loc[t_oof["player_id"] != MESSI, "shr"].iloc[0]),
           spearman_in_vs_oof=float(t_in.merge(t_oof, on=["player_id", "competition"])[["shr_x", "shr_y"]].corr("spearman").iloc[0, 1]))

# ---------- 2. match-clustered bootstrap of the 2015/16 ranking
bd["pk"] = bd["player_id"].astype(str) + "|" + bd["competition"]
keys = pd.Index(bd["pk"].unique()); mids = pd.Index(bd["match_id"].unique())
allm = pd.Index(dr.loc[(dr["season"] == "2015/2016") & dr["competition"].isin(BENCH), "match_id"].unique())
pi, mi = keys.get_indexer(bd["pk"]), allm.get_indexer(bd["match_id"])
pe = bd["p_exp"].to_numpy()


def mat(v):
    return sp.csr_matrix((v, (pi, mi)), shape=(len(keys), len(allm)))


Mw, Me, Mv, Mn = mat(bd["success"].to_numpy().astype(float)), mat(pe), mat(pe * (1 - pe)), mat(np.ones(len(bd)))
messi_k = [i for i, k in enumerate(keys) if k.startswith(f"{MESSI}|")][0]
ranks, top1, top3, vals = [], 0, 0, []
for _ in range(B):
    w = np.bincount(rng.integers(0, len(allm), len(allm)), minlength=len(allm)).astype(float)
    won, exp, var, n = Mw @ w, Me @ w, Mv @ w, Mn @ w
    ok = n >= 30
    shr = np.full(len(keys), np.nan)
    shr[ok], _ = shrink(won[ok], exp[ok], var[ok], n[ok])
    if not ok[messi_k]:
        continue
    r = int((shr[ok] > shr[messi_k]).sum()) + 1
    ranks.append(r); vals.append(shr[messi_k])
ranks = np.array(ranks)
pd.DataFrame({"rank": ranks, "messi_shrunk": vals}).to_csv(R / "rank_boot.csv", index=False)
boot = dict(reps=int(len(ranks)), p_rank1=float((ranks == 1).mean()), p_top3=float((ranks <= 3).mean()),
            p_top10=float((ranks <= 10).mean()), rank_q05=float(np.quantile(ranks, .05)), rank_q95=float(np.quantile(ranks, .95)),
            messi_ci=[float(np.quantile(vals, .025)), float(np.quantile(vals, .975))])

# ---------- 3. Messi career with match-clustered intervals
rows = []
for (comp, ssn), d in dr[dr["player_id"] == MESSI].groupby(["competition", "season"]):
    if len(d) < 50:
        continue
    gm = d.groupby("match_id").agg(w=("success", "sum"), e=("p_exp", "sum"), n=("success", "size"))
    oe = (gm["w"].sum() - gm["e"].sum()) / gm["n"].sum()
    idx = rng.integers(0, len(gm), (B, len(gm)))
    bs = (gm["w"].to_numpy()[idx].sum(1) - gm["e"].to_numpy()[idx].sum(1)) / gm["n"].to_numpy()[idx].sum(1)
    rows.append(dict(competition=comp, season=ssn, attempts=len(d), matches=len(gm), oe=oe,
                     lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
career = pd.DataFrame(rows)
career.to_csv(R / "messi_career_clustered.csv", index=False)
cl = dict(seasons=len(career), above=int((career["oe"] > 0).sum()), lo_above_zero=int((career["lo"] > 0).sum()),
          median_oe=float(career["oe"].median()))

# ---------- 4. reliability: random split-half by match, 2015/16 pool
r_list = []
for _ in range(200):
    half = set(rng.choice(allm, len(allm) // 2, replace=False))
    a = table(bd[bd["match_id"].isin(half)], "p_exp")
    c = table(bd[~bd["match_id"].isin(half)], "p_exp")
    m = a.merge(c, on=["player_id", "competition"])
    m = m[(m["n_x"] >= 15) & (m["n_y"] >= 15)]
    r_list.append(np.corrcoef(m["oe_x"], m["oe_y"])[0, 1])
r_half = float(np.mean(r_list))
rel = dict(split_half_r=r_half, spearman_brown=2 * r_half / (1 + r_half), splits=200, min_attempts_per_half=15)

# ---------- 5. volume vs adjusted success among the pool
vol = t_in.merge(b[["player_id", "competition", "takeons90"]])
rel["corr_volume_adjusted"] = float(vol[["takeons90", "shr"]].corr("spearman").iloc[0, 1])

out = dict(model_fit=fit, out_of_fold_ranking=oof, match_bootstrap=boot, career_clustered=cl, reliability=rel)
json.dump(out, open(R / "robustness.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# ---------- 6. empirical-Bayes posterior rank of Messi (normal approximation)
q = b.dropna(subset=["oe_shrunk"]).reset_index(drop=True)
mi_ = q.index[q["player_id"] == MESSI][0]
draws = rng.normal(q["oe_shrunk"].to_numpy()[None, :], q["oe_sd"].to_numpy()[None, :], (20000, len(q)))
pr = (draws > draws[:, [mi_]]).sum(1) + 1
second = q.drop(mi_).sort_values("oe_shrunk", ascending=False).index[0]
post = dict(p_rank1=float((pr == 1).mean()), p_top3=float((pr <= 3).mean()), p_top10=float((pr <= 10).mean()),
            rank_q05=float(np.quantile(pr, .05)), rank_q95=float(np.quantile(pr, .95)),
            p_above_second=float((draws[:, mi_] > draws[:, second]).mean()),
            p_above_zero=float((draws[:, mi_] > 0).mean()))
out["eb_posterior_rank"] = post
json.dump(out, open(R / "robustness.json", "w"), indent=1)
print(json.dumps(post, indent=1))

# ---------- 7. completions above expectation per 90 (adjusted success x volume)
v90 = q["takeons90"].to_numpy()
q["extra90"] = q["oe_shrunk"] * v90
d90 = draws * v90[None, :]
pr90 = (d90 > d90[:, [mi_]]).sum(1) + 1
srt = q.sort_values("extra90", ascending=False)
out["extra_completions_per90"] = dict(messi=float(q.loc[mi_, "extra90"]), rank=int((q["extra90"] > q.loc[mi_, "extra90"]).sum() + 1),
                                      second=float(srt.iloc[1]["extra90"]), second_player=str(srt.iloc[1]["player"]),
                                      third=float(srt.iloc[2]["extra90"]), third_player=str(srt.iloc[2]["player"]),
                                      p_rank1=float((pr90 == 1).mean()), p_top3=float((pr90 <= 3).mean()))
q[["player", "competition", "takeons", "takeons90", "oe_shrunk", "oe_sd", "extra90"]].sort_values("extra90", ascending=False).to_csv(R / "extra90_2015_16.csv", index=False)
json.dump(out, open(R / "robustness.json", "w"), indent=1)
print(json.dumps(out["extra_completions_per90"], indent=1))
