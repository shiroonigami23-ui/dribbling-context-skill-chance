"""Hierarchical (mixed-effects) model of take-on success: who matters more, the attacker or the defender?

    logit P(y_i = 1) = x_i' beta + a_{attacker(i)} + d_{defender(i)},
    a ~ N(0, s_a^2),  d ~ N(0, s_d^2),   crossed random effects, fitted with lme4::glmer (nAGQ = 0: fixed effects estimated in the penalised IRLS step, the usual fast approximation for large crossed models).

x_i is the Study 1 context (zone, score state, play pattern, late, competition).  Fitted on the two complete
men's league seasons we have from two providers: StatsBomb 2015/16 (4 leagues) and Wyscout 2017/18 (5 leagues).
Also: attacker + defending team, and attacker only (the two-stage method's analogue).

Outputs: results/hier_*.csv (random effects), results/hierarchical.json.
"""
import glob
import json
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis import BENCH, R, add_score_state  # noqa: E402
from decision import zones  # noqa: E402

COLS = ["match_id", "index", "minute", "team", "play_pattern", "player_id", "type", "x", "y", "outcome", "sub_type",
        "goal"]


def statsbomb_takeons():
    m = pd.read_csv("data/matches.csv")
    keep = m[m["competition"].isin(BENCH) & (m["season"] == "2015/2016")]
    tags = keep[["comp_id", "season_id"]].drop_duplicates()
    ev = pd.concat([pd.read_parquet(f"data/events/{c}_{s}.parquet", columns=COLS) for c, s in tags.values])
    ev = ev.merge(keep[["match_id", "competition"]], on="match_id")
    ev["team"] = ev["team"].astype(str)
    dr = ev[ev["type"] == "Dribble"].copy()
    dr = add_score_state(dr, ev)
    de = ev[ev["type"].isin(["Dribbled Past", "Duel"])][["match_id", "index", "team", "player_id", "type", "sub_type"]] \
        .rename(columns={"index": "di", "team": "dteam", "player_id": "defender_id", "type": "dtype"})
    j = dr[["match_id", "index", "team", "outcome"]].merge(de, on="match_id")
    j = j[(j["dteam"] != j["team"]) & ((j["di"] - j["index"]).abs() <= 3)]
    ok = ((j["outcome"] == "Complete") & (j["dtype"] == "Dribbled Past")) | \
         ((j["outcome"] != "Complete") & (j["dtype"] == "Duel") & (j["sub_type"] == "Tackle"))
    j = j[ok].assign(gap=(j["di"] - j["index"]).abs()).sort_values("gap").drop_duplicates(["match_id", "index"])
    dr = dr.merge(j[["match_id", "index", "defender_id", "dteam"]], on=["match_id", "index"], how="left")
    dr["provider"] = "StatsBomb 2015/16"
    return dr


def wyscout_takeons():
    ev = pd.read_parquet("data/wyscout/events.parquet",
                         columns=COLS + ["competition", "season", "defender_id"])
    ev = ev[ev["season"] == "2017/2018"]
    m = pd.read_csv("data/wyscout/matches.csv")
    dr = ev[ev["type"] == "Dribble"].copy()
    dr = add_score_state(dr, ev.assign(team=ev["team"].astype(str)))
    dr = dr.merge(m[["match_id", "home", "away"]].astype({"home": str, "away": str}), on="match_id")
    dr["dteam"] = np.where(dr["team"].astype(str) == dr["home"], dr["away"], dr["home"])
    dr["provider"] = "Wyscout 2017/18"
    return dr


def context(dr):
    dr = dr.copy()
    dr["success"] = (dr["outcome"] == "Complete").astype(int)
    xb, yb = zones(dr["x"], dr["y"])
    dr["zone"] = xb + "_" + yb
    dr["state"] = np.sign(dr["score_diff"]).map({-1: "losing", 0: "level", 1: "winning"})
    dr["pattern"] = np.where(dr["play_pattern"].isin(["From Counter", "Regular Play"]), dr["play_pattern"], "Other")
    dr["late"] = (dr["minute"] >= 75).astype(int)
    dr["attacker"] = "a" + dr["player_id"].astype(str)
    dr["defender"] = np.where(dr["defender_id"].notna(), "d" + dr["defender_id"].astype("Int64").astype(str), np.nan)
    dr["defteam"] = "t" + dr["dteam"].astype(str)
    return dr[["provider", "competition", "match_id", "success", "zone", "state", "pattern", "late", "attacker",
               "defender", "defteam"]]


RCODE = r'''
suppressMessages(library(lme4))
args <- commandArgs(TRUE); f <- args[1]; tag <- args[2]
d <- read.csv(f, stringsAsFactors = TRUE)
d <- d[!is.na(d$defender) & d$defender != "", ]
ctl <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
fx <- success ~ zone + state + pattern + late + competition
m0 <- glmer(update(fx, . ~ . + (1 | attacker)), data = d, family = binomial, control = ctl, nAGQ = 0)
m1 <- glmer(update(fx, . ~ . + (1 | attacker) + (1 | defender)), data = d, family = binomial, control = ctl, nAGQ = 0)
m2 <- glmer(update(fx, . ~ . + (1 | attacker) + (1 | defteam)), data = d, family = binomial, control = ctl, nAGQ = 0)
vc <- function(m) { v <- as.data.frame(VarCorr(m)); setNames(v$sdcor, v$grp) }
lr <- anova(m0, m1)
out <- list(n = nrow(d), attackers = nlevels(droplevels(d$attacker)), defenders = nlevels(droplevels(d$defender)),
            sd_m0 = as.list(vc(m0)), sd_m1 = as.list(vc(m1)), sd_m2 = as.list(vc(m2)),
            aic = list(m0 = AIC(m0), m1 = AIC(m1), m2 = AIC(m2)),
            lrt_defender_chisq = lr$Chisq[2], lrt_defender_p = lr$`Pr(>Chisq)`[2])
re <- ranef(m1, condVar = TRUE)
pa <- attr(re$attacker, "postVar"); pd <- attr(re$defender, "postVar")
write.csv(data.frame(attacker = rownames(re$attacker), a = re$attacker[, 1], se = sqrt(pa[1, 1, ])),
          paste0("results/hier_attacker_", tag, ".csv"), row.names = FALSE)
write.csv(data.frame(defender = rownames(re$defender), d = re$defender[, 1], se = sqrt(pd[1, 1, ])),
          paste0("results/hier_defender_", tag, ".csv"), row.names = FALSE)
re0 <- ranef(m0, condVar = TRUE)
write.csv(data.frame(attacker = rownames(re0$attacker), a = re0$attacker[, 1]),
          paste0("results/hier_attacker_m0_", tag, ".csv"), row.names = FALSE)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 8), file = paste0("results/hier_", tag, ".json"))
'''


def main():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "hier.R").write_text(RCODE)
    out = {}
    for tag, fn in [("sb", statsbomb_takeons), ("wy", wyscout_takeons)]:
        d = context(fn())
        d.to_csv(str(tmp / f"hier_{tag}.csv"), index=False)
        out[f"{tag}_paired_share"] = float(d["defender"].notna().mean())
        r = subprocess.run(["Rscript", str(tmp / "hier.R"), str(tmp / f"hier_{tag}.csv"), tag], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:]); raise SystemExit(1)
        h = json.load(open(f"results/hier_{tag}.json"))
        pbar = d["success"].mean()
        k = pbar * (1 - pbar)                       # logit -> probability scale at the mean
        sa, sd_ = h["sd_m1"]["attacker"], h["sd_m1"]["defender"]
        h["pbar"] = float(pbar)
        h["attacker_sd_pp"] = 100 * sa * k
        h["defender_sd_pp"] = 100 * sd_ * k
        h["attacker_only_sd_pp"] = 100 * h["sd_m0"]["attacker"] * k
        h["attacker_share_of_player_variance"] = sa ** 2 / (sa ** 2 + sd_ ** 2)
        # latent-scale ICC of one duel, and attempts needed for reliability 0.7 on the probability scale
        tau2 = (sa * k) ** 2
        h["attempts_R50"] = int(np.ceil(1 * k / tau2)); h["attempts_R70"] = int(np.ceil(0.7 / 0.3 * k / tau2))
        h["latent_icc_attacker"] = sa ** 2 / (sa ** 2 + sd_ ** 2 + np.pi ** 2 / 3)
        h["latent_icc_defender"] = sd_ ** 2 / (sa ** 2 + sd_ ** 2 + np.pi ** 2 / 3)
        # do the attacker ratings change once the defender is modelled?
        a1 = pd.read_csv(f"results/hier_attacker_{tag}.csv"); a0 = pd.read_csv(f"results/hier_attacker_m0_{tag}.csv")
        n_att = d.groupby("attacker").size()
        aa = a1.merge(a0, on="attacker", suffixes=("_m1", "_m0"))
        aa["n"] = aa["attacker"].map(n_att)
        big = aa[aa["n"] >= 30]
        h["corr_attacker_m0_m1_30plus"] = float(np.corrcoef(big["a_m1"], big["a_m0"])[0, 1])
        h["attackers_30plus"] = int(len(big))
        dd = pd.read_csv(f"results/hier_defender_{tag}.csv")
        n_def = d.groupby("defender").size()
        dd["n"] = dd["defender"].map(n_def)
        h["defenders_30plus"] = int((dd["n"] >= 30).sum())
        out[tag] = h
    # agreement with the two-stage (empirical Bayes) ratings of Study 1, StatsBomb 2015/16
    b = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"])
    a1 = pd.read_csv("results/hier_attacker_sb.csv")
    a1["player_id"] = a1["attacker"].str[1:].astype(int)
    j = b.merge(a1, on="player_id")
    out["sb_corr_two_stage_vs_glmm"] = float(np.corrcoef(j["oe_shrunk"], j["a"])[0, 1])
    out["sb_corr_players"] = int(len(j))
    out["sb_rank_of_top_two_stage_in_glmm"] = int(j["a"].rank(ascending=False)[j["oe_shrunk"].idxmax()])
    json.dump(out, open(R / "hierarchical.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
