"""Parse the public Wyscout data set (Pappalardo et al., 2019) into the Study 1 format.

Source: top-5 European leagues 2017/18 (all matches), FIFA World Cup 2018, UEFA Euro 2016.
Conventions that differ from StatsBomb, and what we do about them:
  * Coordinates are percentages (0-100) with attack towards x = 100; converted to StatsBomb's 120 x 80 yards.
  * A take-on is a "Ground attacking duel" tagged 503/504 (take on left/right); success = tag 703 (won).
    Duels tagged 702 (neutral) are kept as failures in the main analysis and dropped in a sensitivity check.
  * Every duel is logged twice (attacking and defending).  The defender is the player of the paired
    "Ground defending duel" closest in time (within 1.5 s) for the other team.
  * There are no possession ids: a possession is a maximal run of on-ball events (passes, shots, free kicks,
    attacking duels, touches, clearances) by one team; defending duels, fouls and interruptions do not end it.
  * There is no xG: a shot model (logistic: distance, angle, header, free kick, penalty fixed at 0.76) is fitted
    on the 2017/18 league shots and applied to every shot.
  * Play pattern: "From Counter" if any event of the possession carries tag 1901 (counter-attack),
    "Other" if the possession started with a free kick, corner, throw-in or goal kick, else "Regular Play".
Outputs data/wyscout/events.parquet (same columns as Study 1 plus defender_id), matches.csv, minutes.parquet.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import os
W = Path(os.environ.get("WYSCOUT_DIR", "data/wyscout_raw"))  # filled by fetch_wyscout.py
OUT = Path("data/wyscout"); OUT.mkdir(parents=True, exist_ok=True)
FILES = {"England": ("Premier League", "2017/2018"), "Spain": ("La Liga", "2017/2018"),
         "Italy": ("Serie A", "2017/2018"), "Germany": ("1. Bundesliga", "2017/2018"),
         "France": ("Ligue 1", "2017/2018"), "World_Cup": ("FIFA World Cup", "2018"),
         "European_Championship": ("UEFA Euro", "2016")}
ONBALL = {"Pass", "Shot", "Free Kick", "Others on the ball"}
SETP = {"Free Kick", "Corner", "Throw in", "Goal kick", "Free kick cross", "Free kick shot", "Penalty"}


def tags(e):
    return {t["id"] for t in e.get("tags", [])}


def parse(name):
    comp, season = FILES[name]
    ev = json.load(open(W / f"events_{name}.json"))
    rows = []
    for e in ev:
        tg = tags(e)
        p0 = e["positions"][0] if e["positions"] else {"x": None, "y": None}
        p1 = e["positions"][1] if len(e["positions"]) > 1 else {"x": None, "y": None}
        rows.append((e["matchId"], e["id"], e["matchPeriod"], e["eventSec"], e["teamId"], e["playerId"],
                     e["eventName"], e["subEventName"], p0["x"], p0["y"], p1["x"], p1["y"],
                     int(503 in tg or 504 in tg), int(703 in tg), int(702 in tg), int(701 in tg),
                     int(101 in tg), int(102 in tg), int(403 in tg), int(1901 in tg), int(1801 in tg)))
    d = pd.DataFrame(rows, columns=["match_id", "eid", "period", "sec", "team", "player_id", "ev", "sub", "x0", "y0",
                                    "x1", "y1", "takeon_tag", "won", "neutral", "lost", "goal_tag", "own_goal",
                                    "head", "counter", "accurate"])
    d["competition"], d["season"] = comp, season
    return d


def build():
    parts = [parse(n) for n in FILES]
    d = pd.concat(parts, ignore_index=True)
    per = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}
    d["period"] = d["period"].map(per)
    d = d[d["period"] <= 4].sort_values(["match_id", "period", "sec", "eid"]).reset_index(drop=True)
    d["index"] = d.groupby("match_id").cumcount() + 1
    # minute (cumulative, as StatsBomb)
    off = {1: 0, 2: 45, 3: 90, 4: 105}
    d["minute"] = (d["period"].map(off) + d["sec"] / 60).astype(int)
    # coordinates to 120 x 80
    for a, b in [("x0", "x"), ("y0", "y"), ("x1", "end_x"), ("y1", "end_y")]:
        d[b] = (d[a] * (1.2 if a.startswith("x") else 0.8)).astype("float32")
    # ---- possessions
    onball = d["ev"].isin(ONBALL) | (d["sub"] == "Ground attacking duel")
    owner = d["team"].where(onball)
    owner = owner.groupby([d["match_id"], d["period"]]).ffill()
    owner = owner.fillna(d["team"])
    d["owner"] = owner.astype(int)
    change = (d["owner"] != d.groupby(["match_id", "period"])["owner"].shift()) | \
        (d["period"] != d.groupby("match_id")["period"].shift())
    d["possession"] = change.groupby(d["match_id"]).cumsum().astype(int)
    first = d.groupby(["match_id", "possession"])["sub"].transform("first")
    cnt = d.groupby(["match_id", "possession"])["counter"].transform("max")
    d["play_pattern"] = np.where(cnt == 1, "From Counter", np.where(first.isin(SETP), "Other", "Regular Play"))
    # ---- shots, goals and an xG model
    shot = (d["ev"] == "Shot") | d["sub"].isin(["Free kick shot", "Penalty"])
    dx, dy = (120 - d["x"]).clip(lower=0.1), (d["y"] - 40).abs()
    dist = np.hypot(dx, dy)
    ang = np.arctan2(7.32 * dx, dx ** 2 + dy ** 2 - (7.32 / 2) ** 2)
    ang = np.where(ang < 0, ang + np.pi, ang)
    X = pd.DataFrame({"dist": dist, "ang": ang, "head": d["head"], "fk": (d["sub"] == "Free kick shot").astype(int)})
    fitmask = shot & (d["sub"] != "Penalty") & d["season"].eq("2017/2018")
    xm = sm.GLM(d.loc[fitmask, "goal_tag"], sm.add_constant(X[fitmask]), family=sm.families.Binomial()).fit()
    d["shot_xg"] = np.where(shot, np.where(d["sub"] == "Penalty", 0.76, xm.predict(sm.add_constant(X))), np.nan)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(d.loc[fitmask, "goal_tag"], d.loc[fitmask, "shot_xg"])
    d["goal"] = shot & (d["goal_tag"] == 1)
    # ---- types in Study 1 vocabulary
    d["type"] = np.select([d["takeon_tag"].eq(1) & d["sub"].eq("Ground attacking duel"), shot, d["ev"].eq("Pass")],
                          ["Dribble", "Shot", "Pass"], default=d["ev"].astype(str))
    d["outcome"] = np.where(d["type"] == "Dribble", np.where(d["won"] == 1, "Complete", "Incomplete"),
                            np.where((d["type"] == "Pass") & (d["accurate"] == 0), "Incomplete", None))
    d["sub_type"] = np.where((d["type"] == "Pass") & d["sub"].isin(SETP), "Set piece", None)
    d.loc[(d["type"] == "Pass"), "sub_type"] = np.where(d.loc[d["type"] == "Pass", "sub"].isin(
        ["Simple pass", "High pass", "Head pass", "Smart pass", "Cross", "Launch", "Hand pass"]), None, "Set piece")
    # free kicks that are passes count as set-piece passes
    fkp = d["ev"].eq("Free Kick") & ~shot
    d.loc[fkp, "type"] = "Pass"; d.loc[fkp, "sub_type"] = "Set piece"
    # ---- defender of each take-on (paired defending duel)
    tk = d[d["type"] == "Dribble"][["match_id", "period", "sec", "team", "eid"]]
    dd = d[d["sub"] == "Ground defending duel"][["match_id", "period", "sec", "team", "player_id"]] \
        .rename(columns={"sec": "dsec", "team": "dteam", "player_id": "defender_id"})
    j = tk.merge(dd, on=["match_id", "period"])
    j = j[(j["dteam"] != j["team"]) & ((j["dsec"] - j["sec"]).abs() <= 1.5)]
    j["gap"] = (j["dsec"] - j["sec"]).abs()
    j = j.sort_values("gap").drop_duplicates("eid")
    d = d.merge(j[["eid", "defender_id"]], on="eid", how="left")
    d["under_pressure"] = True
    d["aerial_won"] = False
    d["player"] = d["player_id"].astype(str)
    d["team"] = d["team"].astype(str)
    keep = ["match_id", "index", "period", "minute", "possession", "team", "play_pattern", "player_id", "player",
            "type", "x", "y", "end_x", "end_y", "outcome", "sub_type", "under_pressure", "aerial_won", "goal",
            "shot_xg", "competition", "season", "defender_id", "neutral"]
    d[keep].to_parquet(OUT / "events.parquet", index=False)
    return d, {"shot_model_auc": float(auc), "shots": int(fitmask.sum()), "coef": xm.params.to_dict()}


def matches_and_minutes():
    ms, mins = [], []
    players = {p["wyId"]: p for p in json.load(open(W / "players.json"))}
    for name, (comp, season) in FILES.items():
        for m in json.load(open(W / f"matches_{name}.json")):
            dur = {"Regular": 90, "ExtraTime": 120, "Penalties": 120}.get(m.get("duration", "Regular"), 90)
            teams = list(m["teamsData"].values())
            home = next(t for t in teams if t["side"] == "home"); away = next(t for t in teams if t["side"] == "away")
            ms.append(dict(match_id=m["wyId"], competition=comp, season=season, date=m["dateutc"][:10],
                           home=str(home["teamId"]), away=str(away["teamId"]), home_score=home["score"],
                           away_score=away["score"], label=m["label"]))
            for t in teams:
                f = t.get("formation") or {}
                on = {p["playerId"]: 0.0 for p in f.get("lineup", [])}
                off = {}
                for s in (f.get("substitutions") or []):
                    if isinstance(s, dict):
                        off[s["playerOut"]] = s["minute"]; on[s["playerIn"]] = s["minute"]
                for pid, a in on.items():
                    b = off.get(pid, dur)
                    pl = players.get(pid, {})
                    mins.append(dict(match_id=m["wyId"], team=str(t["teamId"]), player_id=pid,
                                     player=f'{pl.get("firstName", "")} {pl.get("lastName", "")}'.strip(),
                                     minutes=max(b - a, 0), competition=comp, season=season))
    pd.DataFrame(ms).to_csv(OUT / "matches.csv", index=False)
    pd.DataFrame(mins).to_parquet(OUT / "minutes.parquet", index=False)
    pl = pd.DataFrame([dict(player_id=k, first=v.get("firstName", ""), last=v.get("lastName", ""),
                            short=v.get("shortName", ""), birth=v.get("birthDate"),
                            role=(v.get("role") or {}).get("name")) for k, v in players.items()])
    pl.to_parquet(OUT / "players.parquet", index=False)


if __name__ == "__main__":
    matches_and_minutes()
    d, info = build()
    tk = d[d["type"] == "Dribble"]
    info.update(events=int(len(d)), matches=int(d["match_id"].nunique()), takeons=int(len(tk)),
                success=float((tk["outcome"] == "Complete").mean()), neutral=float(tk["neutral"].mean()),
                defender_paired=float(tk["defender_id"].notna().mean()),
                by_comp=tk.groupby("competition").size().to_dict())
    json.dump(info, open(OUT / "parse_info.json", "w"), indent=1, default=float)
    Path("results").mkdir(exist_ok=True)
    json.dump(info, open("results/wyscout_parse_info.json", "w"), indent=1, default=float)
    print(json.dumps(info, indent=1, default=float))
