"""Study 5: a second provider (Wyscout public data, 2017/18) and cross-provider tests.

(a) Replication: the Study 1 pipeline on Wyscout's five complete 2017/18 leagues (+ World Cup 2018, Euro 2016).
(b) Same matches, two providers: the 64 World Cup 2018 matches are coded by both StatsBomb and Wyscout.
    Player-match take-on attempts and completions are compared (Lin's concordance, ICC(2,1), Bland-Altman).
(c) Same players, two seasons, two providers: players rated in StatsBomb 2015/16 (Study 1 benchmark) are
    matched by name to Wyscout 2017/18.  We correlate their ratings, correct the correlation for the
    unreliability of each rating (Spearman's disattenuation), and test whether the 2015/16 StatsBomb rating
    predicts 2017/18 Wyscout take-on success out of sample.
Output: results/study5.json and tables.
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis import R  # noqa: E402
from replicate import breakeven, design, fit, halves_test, prep, score, summarise  # noqa: E402
from decision import shrink  # noqa: E402

WD = Path("data/wyscout")
LEAGUES = ["Premier League", "La Liga", "Serie A", "1. Bundesliga", "Ligue 1"]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def load_w(drop_neutral=False):
    ev = pd.read_parquet(WD / "events.parquet")
    if drop_neutral:
        ev = ev[~((ev["type"] == "Dribble") & (ev["neutral"] == 1))]
    for c in ["team", "play_pattern", "type", "outcome", "sub_type"]:
        ev[c] = ev[c].astype("category")
    m = pd.read_csv(WD / "matches.csv")
    mins = pd.read_parquet(WD / "minutes.parquet")
    return ev, m, mins


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sab = np.cov(a, b, ddof=0)[0, 1]
    return float(2 * sab / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    Y = np.column_stack([a, b]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def main():
    out = {}
    cats = ["zone", "state", "pattern", "cs"]
    # ---------------- (a) replication
    ev, m, mins = load_w()
    dr, pa = prep(ev)
    res, g = summarise(dr, pa, mins, m, "wyscout", leagues=LEAGUES)
    full = [f"{c} 2017/2018" for c in LEAGUES]
    res["halves"] = halves_test(dr[dr["cs"].isin(full)], m, cats)
    res["defender_paired"] = float(ev.loc[ev["type"] == "Dribble", "defender_id"].notna().mean())
    out["replication"] = res
    breakeven(dr, pa).to_csv(R / "breakeven_wyscout.csv", index=False)
    g.to_csv(R / "ratings_wyscout_2017_18.csv", index=False)
    dr[["match_id", "index", "player_id", "defender_id", "team", "success", "p", "zone", "state", "pattern", "cs",
        "late", "competition"]].to_parquet(R / "takeons_wyscout.parquet", index=False)
    del pa
    # ---------------- (b) World Cup 2018, both providers
    sbm = pd.read_csv("data/more/matches_43_3.csv")
    sbe = pd.read_parquet("data/more/events/43_3.parquet")
    wm = m[m["competition"] == "FIFA World Cup"].copy()
    teams = {t["wyId"]: t["name"] for t in json.load(open(Path(os.environ.get("WYSCOUT_DIR", "data/wyscout_raw")) / "teams.json"))}
    wm["h"], wm["a"] = wm["home"].astype(int).map(teams).map(norm), wm["away"].astype(int).map(teams).map(norm)
    sbm["h"], sbm["a"] = sbm["home"].map(norm), sbm["away"].map(norm)
    pairs = []
    for _, r in sbm.iterrows():
        c = wm[(wm["date"] == r["date"]) & (((wm["h"] == r["h"]) & (wm["a"] == r["a"])) | ((wm["h"] == r["a"]) & (wm["a"] == r["h"])))]
        if len(c) == 1:
            pairs.append((r["match_id"], int(c["match_id"].iat[0])))
    pmap = dict(pairs)
    # players: StatsBomb names vs Wyscout first+last
    wpl = pd.read_parquet(WD / "players.parquet")
    wpl["full"] = (wpl["first"] + " " + wpl["last"]).map(norm)
    wpl["shortn"] = wpl["short"].map(norm)
    sbt = sbe[sbe["type"] == "Dribble"].copy()
    sbt["succ"] = (sbt["outcome"] == "Complete").astype(int)
    a = sbt.groupby(["match_id", "player_id", "player"]).agg(att=("succ", "size"), won=("succ", "sum")).reset_index()
    sb_players = sbe[["match_id", "player_id", "player"]].drop_duplicates()
    wt = ev[(ev["type"] == "Dribble") & (ev["competition"] == "FIFA World Cup")].copy()
    wt["succ"] = (wt["outcome"] == "Complete").astype(int)
    b = wt.groupby(["match_id", "player_id"]).agg(att=("succ", "size"), won=("succ", "sum")).reset_index()
    w_players = ev[ev["competition"] == "FIFA World Cup"][["match_id", "player_id"]].drop_duplicates() \
        .merge(wpl[["player_id", "full", "shortn", "last"]], on="player_id")

    def match_name(sbname, cands):
        n = norm(sbname); toks = set(n.split())
        exact = cands[cands["full"] == n]
        if len(exact) == 1:
            return exact["player_id"].iat[0]
        # token overlap on last name tokens
        best, bs = None, 0
        for _, c in cands.iterrows():
            ct = set(c["full"].split())
            sc = len(toks & ct) / max(len(ct), 1)
            if sc > bs:
                best, bs = c["player_id"], sc
        return best if bs >= 0.5 else None
    link = []
    for sbmid, wmid in pairs:
        cands = w_players[w_players["match_id"] == wmid].drop_duplicates("player_id")
        for _, p in sb_players[sb_players["match_id"] == sbmid].drop_duplicates("player_id").iterrows():
            wid = match_name(p["player"], cands)
            if wid is not None:
                link.append((sbmid, p["player_id"], wmid, int(wid)))
    L = pd.DataFrame(link, columns=["sb_match", "sb_player", "w_match", "w_player"]).drop_duplicates(["sb_match", "sb_player"])
    L = L.merge(a.rename(columns={"match_id": "sb_match", "player_id": "sb_player", "att": "att_sb", "won": "won_sb"}),
                on=["sb_match", "sb_player"], how="left") \
        .merge(b.rename(columns={"match_id": "w_match", "player_id": "w_player", "att": "att_w", "won": "won_w"}),
               on=["w_match", "w_player"], how="left").fillna({"att_sb": 0, "won_sb": 0, "att_w": 0, "won_w": 0})
    L = L[(L["att_sb"] + L["att_w"]) > 0]
    tot = L.groupby("sb_player")[["att_sb", "won_sb", "att_w", "won_w"]].sum()
    tot10 = tot[(tot["att_sb"] >= 10) & (tot["att_w"] >= 10)]
    diff = L["att_w"] - L["att_sb"]
    out["wc2018_two_providers"] = {
        "matches_paired": len(pairs), "player_matches": int(len(L)), "players_linked": int(L["sb_player"].nunique()),
        "takeons_sb": int(L["att_sb"].sum()), "takeons_w": int(L["att_w"].sum()),
        "success_sb": float(L["won_sb"].sum() / L["att_sb"].sum()), "success_w": float(L["won_w"].sum() / L["att_w"].sum()),
        "attempts_pearson": float(np.corrcoef(L["att_sb"], L["att_w"])[0, 1]),
        "attempts_ccc": ccc(L["att_sb"], L["att_w"]), "attempts_icc21": icc21(L["att_sb"], L["att_w"]),
        "bland_altman_mean": float(diff.mean()), "bland_altman_loa": [float(diff.mean() - 1.96 * diff.std()),
                                                                     float(diff.mean() + 1.96 * diff.std())],
        "tournament_attempts_ccc": ccc(tot["att_sb"], tot["att_w"]),
        "tournament_success_corr_10plus": float(np.corrcoef(tot10["won_sb"] / tot10["att_sb"],
                                                            tot10["won_w"] / tot10["att_w"])[0, 1]),
        "players_10plus_both": int(len(tot10)),
    }
    L.to_csv(R / "wc2018_two_providers.csv", index=False)

    # ---------------- (c) same players, StatsBomb 2015/16 -> Wyscout 2017/18
    sb = pd.read_csv(R / "benchmark_2015_16.csv").dropna(subset=["oe_shrunk"])
    s1 = json.load(open(R / "summary.json"))
    tau2_sb, pbar_sb = s1["tau2_oe"], s1["base_success"]
    sb["rel"] = tau2_sb / (tau2_sb + pbar_sb * (1 - pbar_sb) / sb["takeons"])
    sb["n"] = sb["player"].map(norm)
    wr = g.copy()      # Wyscout 2017/18 league ratings (>= 900 min, >= 30 attempts)
    wr = wr[wr["cs"].str.endswith("2017/2018")]
    wr = wr.merge(wpl[["player_id", "full"]], on="player_id")
    tau2_w = (res["tau_pp"] / 100) ** 2
    pbar_w = res["success"]
    wr["rel"] = tau2_w / (tau2_w + pbar_w * (1 - pbar_w) / wr["n"])
    # exact normalised full-name match, unique on both sides
    sbu = sb[~sb["n"].duplicated(keep=False)]
    wru = wr.sort_values("n", ascending=False).drop_duplicates("player_id")
    wru = wru[~wru["full"].duplicated(keep=False)]
    J = sbu.merge(wru, left_on="n", right_on="full", suffixes=("_sb", "_w"))
    r_obs = float(np.corrcoef(J["oe_shrunk"], J["oe_s"])[0, 1])
    r_raw = float(np.corrcoef((J["takeons_won"] - J["exp_won"]) / J["takeons"], (J["won"] - J["exp"]) / J["n_w"])[0, 1])
    rel_sb, rel_w = float(J["rel_sb"].mean()), float(J["rel_w"].mean())
    # bootstrap over players
    rng = np.random.default_rng(5)
    bs = []
    x1 = ((J["takeons_won"] - J["exp_won"]) / J["takeons"]).to_numpy()
    x2 = ((J["won"] - J["exp"]) / J["n_w"]).to_numpy()
    for _ in range(4000):
        i = rng.integers(0, len(J), len(J))
        bs.append(np.corrcoef(x1[i], x2[i])[0, 1])
    out["sb1516_to_w1718"] = {
        "players_matched": int(len(J)), "corr_unshrunk": r_raw,
        "corr_unshrunk_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "corr_shrunk": r_obs, "mean_reliability_sb": rel_sb, "mean_reliability_w": rel_w,
        "disattenuated": float(r_raw / np.sqrt(rel_sb * rel_w)),
        "disattenuated_ci": [float(np.percentile(bs, 2.5) / np.sqrt(rel_sb * rel_w)),
                             float(np.percentile(bs, 97.5) / np.sqrt(rel_sb * rel_w))],
    }
    # predictive test on the Wyscout 2017/18 take-ons of matched players
    tk = dr[dr["player_id"].isin(J["player_id_w"]) & dr["cs"].isin(full)].copy()
    jj = J.set_index("player_id_w")
    tk["sb_shr"] = tk["player_id"].map(jj["oe_shrunk"]); tk["sb_raw"] = tk["player_id"].map(jj["succ"])
    y = tk["success"].to_numpy()
    out["sb1516_to_w1718"]["predict"] = {
        "takeons": int(len(tk)),
        "league_avg_sb": score(y, np.full(len(tk), pbar_sb)),
        "raw_sb_rate": score(y, tk["sb_raw"].to_numpy()),
        "context_w": score(y, tk["p"].to_numpy()),
        "context_w_plus_sb_shrunk": score(y, (tk["p"] + tk["sb_shr"]).to_numpy()),
    }
    # bootstrap the gain of context+shrunk over context (players resampled)
    idx = tk.groupby("player_id").indices
    pid = np.array(list(idx))
    lc = -(y * np.log(np.clip(tk["p"], .02, .98)) + (1 - y) * np.log(1 - np.clip(tk["p"], .02, .98))).to_numpy()
    q = np.clip(tk["p"] + tk["sb_shr"], .02, .98).to_numpy()
    ls = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    gains = []
    for _ in range(2000):
        ii = np.concatenate([idx[p] for p in rng.choice(pid, len(pid))])
        gains.append(lc[ii].mean() - ls[ii].mean())
    out["sb1516_to_w1718"]["predict"]["gain_ci"] = [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
    J.to_csv(R / "sb1516_w1718_linked.csv", index=False)
    del ev, dr
    # sensitivity: neutral duels dropped
    ev2, _, mins2 = load_w(drop_neutral=True)
    d2, pa2 = prep(ev2)
    r2, _ = summarise(d2, pa2, mins2, m, "wyscout_no_neutral", leagues=LEAGUES)
    out["no_neutral"] = {k: r2[k] for k in ["takeons", "success", "auc_cv", "tau_pp", "attempts_needed"]}
    del ev2, d2, pa2

    json.dump(out, open(R / "study5.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
