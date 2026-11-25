"""Download the StatsBomb open data that Study 1 does not use, and 360 freeze-frame features.

Studies 2-4 of the v3 paper:
  * women's football: seven full league seasons and four tournaments,
  * men's football not in Study 1: World Cups, Euros, Copa America, AFCON and a full ISL season,
  * 360 freeze frames for every match that has them (including the Study 1 matches with 360 data).

Events go to data/more/events/<comp>_<season>.parquet (same columns as fetch.py), matches to
data/more/matches.csv, minutes to data/more/minutes.parquet.  For every take-on and every pass in a
360 match, the freeze frame is reduced to a few numbers (defenders near the ball, defenders goal-side,
open teammates, visible area) in data/more/ff360.parquet; raw 360 files are not kept.
"""
import concurrent.futures as cf
import json
import math
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch import BASE, flatten  # noqa: E402

OUT = Path("data/more"); (OUT / "events").mkdir(parents=True, exist_ok=True)
(OUT / "ff").mkdir(exist_ok=True)

WOMEN = [(37, 4), (37, 42), (37, 90), (37, 281), (49, 3), (49, 107), (182, 281), (135, 281), (131, 281),
         (72, 30), (72, 107), (53, 106), (53, 315)]
MEN = [(43, 3), (43, 106), (55, 43), (55, 282), (223, 282), (1267, 107), (1238, 108), (9, 281)]
STUDY1_360 = [(11, 90), (7, 108), (7, 235)]          # events already in Study 1; fetch 360 features only


def get(url, tries=5):
    for _ in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=180))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            continue
    raise RuntimeError(url)


def poly_area(v):
    xs, ys = v[0::2], v[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2 if n >= 3 else 0.0


def ff_features(ev, ff):
    """Freeze-frame features for take-ons and passes."""
    want = {e["id"]: e for e in ev if e["type"]["name"] in ("Dribble", "Pass") and e.get("location")}
    rows = []
    for f in ff:
        e = want.get(f["event_uuid"])
        if e is None:
            continue
        fr = f.get("freeze_frame") or []
        actor = next((p["location"] for p in fr if p.get("actor")), None)
        if actor is None:
            continue                     # orientation cannot be checked without the actor
        ex, ey = e["location"][:2]
        # some frames (notably failed take-ons) are stored in the opposite orientation: (120 - x, 80 - y).
        # Re-orient so the frame matches the event, judged by where the actor is.
        flipped = math.hypot(actor[0] - (120 - ex), actor[1] - (80 - ey)) < math.hypot(actor[0] - ex, actor[1] - ey)
        if flipped:
            fr = [dict(p, location=[120 - p["location"][0], 80 - p["location"][1]]) for p in fr]
            actor = [120 - actor[0], 80 - actor[1]]
            va = f.get("visible_area") or []
            f = dict(f, visible_area=[120 - v if i % 2 == 0 else 80 - v for i, v in enumerate(va)])
        ax, ay = actor
        opp = [p["location"] for p in fr if not p.get("teammate") and not p.get("keeper")]
        mates = [p["location"] for p in fr if p.get("teammate") and not p.get("actor") and not p.get("keeper")]
        dist = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])
        d_opp = sorted(dist((ax, ay), o) for o in opp)
        gx, gy = 120.0, 40.0
        gvec = (gx - ax, gy - ay)
        gnorm = math.hypot(*gvec) or 1.0

        def in_cone(o, deg=30):
            v = (o[0] - ax, o[1] - ay)
            n = math.hypot(*v)
            if n == 0 or n > gnorm:
                return False
            return (v[0] * gvec[0] + v[1] * gvec[1]) / (n * gnorm) > math.cos(math.radians(deg))

        def open_mate(m):
            return dist((ax, ay), m) <= 35 and (min((dist(m, o) for o in opp), default=99) > 5)

        rows.append(dict(
            id=e["id"], type=e["type"]["name"], flipped=flipped, actor_gap=math.hypot(ax - ex, ay - ey),
            d_near=d_opp[0] if d_opp else None,
            opp3=sum(d <= 3 for d in d_opp), opp5=sum(d <= 5 for d in d_opp), opp10=sum(d <= 10 for d in d_opp),
            opp_cone=sum(in_cone(o) for o in opp),
            open_mates=sum(open_mate(m) for m in mates),
            open_fwd=sum(open_mate(m) and m[0] > ax + 2 for m in mates),
            n_opp=len(opp), n_mates=len(mates), vis_area=poly_area(f.get("visible_area") or []),
        ))
    return rows


def match(m, keep_events, has360):
    mid = m["match_id"]
    ev = get(f"{BASE}events/{mid}.json")
    rows, mins = flatten(mid, ev) if keep_events else ([], [])
    ffr = []
    if has360:
        try:
            ff = get(f"{BASE}three-sixty/{mid}.json", tries=8)
        except RuntimeError:
            ff = None
            print("360 missing", mid, flush=True)
        if ff:
            ffr = [dict(match_id=mid, **r) for r in ff_features(ev, ff)]
    return rows, mins, ffr


def season(c, s, gender, keep_events=True):
    tag = f"{c}_{s}"
    done = OUT / "events" / f"{tag}.parquet" if keep_events else OUT / "ff" / f"{tag}.parquet"
    if done.exists():
        print("skip", tag, flush=True)
        return
    ms = get(f"{BASE}matches/{c}/{s}.json")
    rows, mins, ffr = [], [], []
    with cf.ThreadPoolExecutor(12) as ex:
        futs = [ex.submit(match, m, keep_events, m.get("match_status_360") == "available") for m in ms]
        for f in cf.as_completed(futs):
            r, mn, fr = f.result()
            rows += r; mins += mn; ffr += fr
    if ffr:
        pd.DataFrame(ffr).to_parquet(OUT / "ff" / f"{tag}.parquet", index=False)
    if keep_events:
        meta = [dict(match_id=m["match_id"], competition=m["competition"]["competition_name"],
                     season=m["season"]["season_name"], date=m["match_date"], home=m["home_team"]["home_team_name"],
                     away=m["away_team"]["away_team_name"], home_score=m["home_score"], away_score=m["away_score"],
                     comp_id=c, season_id=s, gender=gender,
                     has360=m.get("match_status_360") == "available") for m in ms]
        pd.DataFrame(meta).to_csv(OUT / f"matches_{tag}.csv", index=False)
        pd.DataFrame(mins).to_parquet(OUT / f"minutes_{tag}.parquet", index=False)
        pd.DataFrame(rows).to_parquet(done, index=False)
    print(tag, gender, len(ms), "matches", len(rows), "events", len(ffr), "360 rows", flush=True)


def redo_360():
    """Recompute only the 360 features for every competition with 360 data."""
    for c, s in MEN + STUDY1_360 + WOMEN:
        season(c, s, None, keep_events=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "360":
        redo_360()
        sys.exit()
    for c, s in MEN:
        season(c, s, "men")
    for c, s in STUDY1_360:
        season(c, s, "men", keep_events=False)
    for c, s in WOMEN:
        season(c, s, "women")
