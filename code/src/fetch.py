"""Download StatsBomb open-data events for the selected competitions and keep only what the study uses.

One compact parquet file per competition-season in data/events/. Also writes data/matches.csv and
data/minutes.parquet (minutes played per player per match, from lineups + substitutions + red cards).
"""
import concurrent.futures as cf
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
SEASONS = [(11, s) for s in [37, 38, 39, 40, 41, 21, 22, 23, 24, 25, 26, 27, 2, 1, 4, 42, 90, 278]] + \
          [(2, 27), (12, 27), (9, 27), (7, 27), (7, 108), (7, 235), (12, 86), (2, 44)]
KEEP = {"Dribble", "Carry", "Pass", "Shot", "Ball Receipt*", "Miscontrol", "Dispossessed", "Duel", "Dribbled Past",
        "Foul Won", "Clearance", "Ball Recovery", "Interception", "Block", "Goal Keeper", "Foul Committed"}
OUT = Path("data"); (OUT / "events").mkdir(parents=True, exist_ok=True)


def get(url):
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception:
            continue
    raise RuntimeError(url)


def loc(e, k="location"):
    v = e.get(k)
    return (v[0], v[1]) if v else (None, None)


def flatten(mid, ev):
    rows, mins = [], []
    end = {}
    for e in ev:
        t = e["type"]["name"]
        if t == "Half End":
            end[e["period"]] = max(end.get(e["period"], 0), e["minute"] + e["second"] / 60)
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                mins.append(dict(match_id=mid, team=e["team"]["name"], player_id=p["player"]["id"],
                                 player=p["player"]["name"], position=p["position"]["name"], on=0.0, off=None))
        if t == "Substitution":
            m = e["minute"] + e["second"] / 60
            for r in mins:
                if r["player_id"] == e["player"]["id"] and r["off"] is None:
                    r["off"] = m
            rep = e["substitution"]["replacement"]
            mins.append(dict(match_id=mid, team=e["team"]["name"], player_id=rep["id"], player=rep["name"],
                             position=e["position"]["name"], on=m, off=None))
        if t in ("Bad Behaviour", "Foul Committed"):
            card = (e.get("bad_behaviour") or e.get("foul_committed") or {}).get("card", {}).get("name", "")
            if card in ("Red Card", "Second Yellow") and "player" in e:
                for r in mins:
                    if r["player_id"] == e["player"]["id"] and r["off"] is None:
                        r["off"] = e["minute"] + e["second"] / 60
        if t not in KEEP or "player" not in e:
            continue
        x, y = loc(e)
        sub = e.get(t.lower().replace("*", "").replace(" ", "_"), {}) or {}
        ex = ey = None
        if "end_location" in sub:
            ex, ey = sub["end_location"][0], sub["end_location"][1]
        rows.append(dict(
            match_id=mid, id=e["id"], index=e["index"], period=e["period"], minute=e["minute"], second=e["second"],
            possession=e["possession"], possession_team=e["possession_team"]["name"], team=e["team"]["name"],
            play_pattern=e["play_pattern"]["name"], player_id=e["player"]["id"], player=e["player"]["name"],
            position=(e.get("position") or {}).get("name"), type=t, x=x, y=y, end_x=ex, end_y=ey,
            outcome=(sub.get("outcome") or {}).get("name"), sub_type=(sub.get("type") or {}).get("name"),
            under_pressure=bool(e.get("under_pressure", False)), duration=e.get("duration"),
            nutmeg=bool(sub.get("nutmeg", False)), overrun=bool(sub.get("overrun", False)),
            body_part=(sub.get("body_part") or {}).get("name"), aerial_won=bool(sub.get("aerial_won", False)),
            shot_xg=sub.get("statsbomb_xg") if t == "Shot" else None,
            goal=(t == "Shot" and (sub.get("outcome") or {}).get("name") == "Goal")))
    total = max(end.values()) if end else 90.0   # StatsBomb minutes are cumulative across periods
    for r in mins:
        r["off"] = total if r["off"] is None else r["off"]
        r["minutes"] = max(r["off"] - r["on"], 0.0)
    return rows, mins


def season(cs):
    c, s = cs
    ms = get(f"{BASE}matches/{c}/{s}.json")
    rows, mins = [], []
    with cf.ThreadPoolExecutor(12) as ex:
        for r, mn in ex.map(lambda m: flatten(m["match_id"], get(f"{BASE}events/{m['match_id']}.json")), ms):
            rows += r; mins += mn
    pd.DataFrame(rows).to_parquet(OUT / "events" / f"{c}_{s}.parquet", index=False)
    meta = [dict(match_id=m["match_id"], competition=m["competition"]["competition_name"], season=m["season"]["season_name"],
                 date=m["match_date"], home=m["home_team"]["home_team_name"], away=m["away_team"]["away_team_name"],
                 home_score=m["home_score"], away_score=m["away_score"], comp_id=c, season_id=s) for m in ms]
    return meta, mins, (c, s, len(ms), len(rows))


if __name__ == "__main__":
    allm, allmin = [], []
    for cs in SEASONS:
        meta, mins, info = season(cs)
        allm += meta; allmin += mins
        print(info, flush=True)
    pd.DataFrame(allm).to_csv(OUT / "matches.csv", index=False)
    pd.DataFrame(allmin).to_parquet(OUT / "minutes.parquet", index=False)
