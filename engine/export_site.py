"""Export JSON for the website: one file per upcoming FBS-vs-FBS game plus a week index.

Usage: python -m engine.export_site [season] [week]
  Defaults to the current season and the next week with unplayed games.
Writes site/data/week.json and site/data/games/<gameId>.json, plus site/data/league.json and
site/data/teams/<team>.json for the build-your-own matchup page.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.model import (LABELS, METRICS, COUNT_COL, Blend, Matchup, Params, PointsModel, baseline_for, fbs_teams,
                          higher_is_better, load, matchup_features, matchup_units, percentile_refs, profiles_asof,
                          project, results_ratings, unit_ingredients)

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "site" / "data"
RESULTS = ROOT / "data" / "backtest" / "results.json"
BEST_A = None
if RESULTS.exists():
    _r = json.loads(RESULTS.read_text())
    BEST_A = {k: _r["params"][k] for k in ("a_opp", "a_rec", "a_set", "a_ros")}


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(float(x)) or math.isinf(float(x))) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def load_tuned() -> tuple[Params, PointsModel, dict | None]:
    """Site params: backtest-tuned settings, with the slider strengths at the site defaults."""
    if RESULTS.exists():
        r = json.loads(RESULTS.read_text())
        a = r.get("site_default", {}).get("a", {})
        return Params(**{**r["params"], **a}), PointsModel(**r["points_model"]), r
    return Params(), PointsModel(), None


def load_blend() -> Blend | None:
    """Blend of the play-by-play margin with the results rating (engine/backtest.fit_blend)."""
    if RESULTS.exists():
        b = json.loads(RESULTS.read_text()).get("blend")
        if b:
            return Blend(**b["coef"])
    return None


def pick_week(season: int) -> tuple[str, int]:
    """(seasonType, week) of the next unplayed FBS-vs-FBS game from now on."""
    _, g = load()
    fbs = fbs_teams(season)
    s = g[(g.season == season) & (g.completed != True) & g.homeTeam.isin(fbs) & g.awayTeam.isin(fbs)]  # noqa: E712
    upcoming = s[s.date >= pd.Timestamp.now("UTC") - pd.Timedelta(hours=12)].sort_values("date")
    if upcoming.empty:
        upcoming = s.sort_values("date")
    r = upcoming.iloc[0]
    return str(r.seasonType), int(r.week)


def recent_form(team: str, season: int, date: pd.Timestamp, n: int = 5) -> list[dict]:
    _, g = load()
    s = g[(g.season == season) & (g.completed == True) & (g.date < date) &  # noqa: E712
          ((g.homeTeam == team) | (g.awayTeam == team))].sort_values("date", ascending=False).head(n)
    out = []
    for r in s.itertuples(index=False):
        home = r.homeTeam == team
        pts, opp_pts = (r.homePoints, r.awayPoints) if home else (r.awayPoints, r.homePoints)
        out.append({"date": r.date.strftime("%m/%d"), "opp": r.awayTeam if home else r.homeTeam,
                    "site": "neutral" if r.neutralSite else ("home" if home else "away"),
                    "result": "W" if pts > opp_pts else ("L" if pts < opp_pts else "T"),
                    "pts": int(pts), "opp_pts": int(opp_pts)})
    return out


def unit_payload(info: dict, season_opps: set[str], common: set[str]) -> dict:
    r = info["rows"]
    games = []
    for row in r.itertuples(index=False):
        games.append({
            "week": int(row.week), "date": row.date.strftime("%m/%d"), "opponent": row.opponent,
            "site": row.site, "pts": row.pts, "opp_pts": row.opp_pts, "qb": row.qb if isinstance(row.qb, str) else None,
            "opp_qb": row.opp_qb if isinstance(row.opp_qb, str) else None,
            "dist": row.dist, "weeks_ago": row.weeks_ago, "site_base": row.site_base,
            "wx_d2": None if row.wx_d2 is None or (isinstance(row.wx_d2, float) and math.isnan(row.wx_d2)) else row.wx_d2,
            "qb_match": row.qb_match, "fcs": not bool(row.opp_fbs), "common": row.opponent in common,
            "rb_played": bool(row.rb_played), "wr_played": bool(row.wr_played),
            "lu": {pos: {"share": mm["share"].get(int(row.gameId), 0.0), "pool": mm["pool"].get(int(row.gameId), 0.0)}
                   for pos, mm in (info.get("km") or {}).items()},
            "plays": row.plays, "rushes": row.rushes, "dropbacks": row.dropbacks, "opp_level": row.opp_level,
            "m": {m: getattr(row, m) for m in METRICS},
        })
    qbs = sorted({g["qb"] for g in games if g["qb"]})
    km = {pos: {k: v for k, v in mm.items() if k not in ("share", "pool")} for pos, mm in (info.get("km") or {}).items()}
    return {"expected_qb": info["expected_qb"], "qbs": qbs, "key": info.get("key"), "km": km,
            "baseline": info["baseline"].to_dict(), "games": games}


def export_game(row, p: Params, pm: PointsModel) -> dict:
    season = int(row.season)
    weather = None if pd.isna(row.temp) and not bool(row.indoors) else {
        "indoors": bool(row.indoors), "temp": row.temp, "wind": row.wind, "precip": row.precip}
    m = Matchup(season, row.date, row.homeTeam, row.awayTeam, bool(row.neutralSite), weather)
    units = matchup_units(m)
    feats = matchup_features(m, units, p)
    bl = load_blend()
    proj = project(m, feats, pm, bl)
    eq = project(m, matchup_features(m, units, p.equal()), pm, bl)
    rr = results_ratings(season, m.date.isoformat())["rating"]
    refs = percentile_refs(season, m.date)
    prof = profiles_asof(season, m.date.isoformat())

    opps = {t: set(units[(t, "offense")]["rows"].opponent) for t in (m.home, m.away)}
    common = (opps[m.home] & opps[m.away]) - {m.home, m.away}

    def plays_avg(t, side):
        r = units[(t, side)]["rows"]
        return float(r.plays.mean()) if not r.empty else 65.0

    return _clean({
        "game": {"id": int(row.gameId), "season": season, "week": int(row.week), "start": row.date.isoformat(),
                 "home": m.home, "away": m.away, "neutral": m.neutral, "venue": row.venue,
                 "home_conf": row.homeConference, "away_conf": row.awayConference,
                 "weather": weather, "spread_home": row.spread_home, "total": row.total},
        "common_opponents": sorted(common),
        "form": {m.home: recent_form(m.home, season, m.date), m.away: recent_form(m.away, season, m.date)},
        "units": {f"{t}|{s}": unit_payload(units[(t, s)], opps[t], common)
                  for t in (m.home, m.away) for s in ("offense", "defense")},
        "refs": {f"{side}|{mt}": refs[(side, mt)].tolist() for side in ("offense", "defense") for mt in METRICS},
        "lg_epa": prof.attrs.get("lg_epa", 0.0),
        "plays": {m.home: (plays_avg(m.home, "offense") + plays_avg(m.away, "defense")) / 2,
                  m.away: (plays_avg(m.away, "offense") + plays_avg(m.home, "defense")) / 2},
        "params": asdict(p), "points_model": asdict(pm),
        "best_a": {k: v for k, v in (BEST_A or {}).items()},
        "results": {m.home: float(rr.get(m.home, 0.0)), m.away: float(rr.get(m.away, 0.0))},
        "blend": asdict(bl) if bl else None,
        "projection": proj, "projection_equal": eq,
    })


def _team_slug(team: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in team.lower()).strip("-")


def export_teams(season: int, p: Params, pm: PointsModel) -> int:
    """Data for user-built matchups: one file per FBS team plus league.json.

    Everything in a unit's weighting ingredients is team-specific except the opponent similarity
    (dist) and the site match, which depend on the matchup. Those are recomputed in the browser
    (site/custom.js) from the scaled similarity vectors in league.json, mirroring
    model.unit_ingredients. Profiles are as of now. Each past game carries its weather so the
    browser can apply a weather scenario the user picks (model._wx_dist2).
    """
    from engine.model import SIM_FEATURES, SITE_MATCH, feature_scales, key_player_models, key_players, last_qb

    asof = pd.Timestamp.now("UTC").floor("h")
    prof = profiles_asof(season, asof.isoformat())
    if prof.empty:
        print("  no games yet this season; skipping team export")
        return 0
    _, g = load()
    fbs = sorted(fbs_teams(season))
    scales = feature_scales()
    vec = {}
    for r in prof.itertuples(index=False):
        v = [getattr(r, c) / scales[c] for c, _ in SIM_FEATURES]
        vec[f"{r.team}|{r.side}"] = None if any(pd.isna(x) for x in v) else v
    s = g[g.season == season]
    conf = {**dict(zip(s.awayTeam, s.awayConference)), **dict(zip(s.homeTeam, s.homeConference))}

    out_dir = SITE_DATA / "teams"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for team in fbs:
        try:
            qb = last_qb(team, season, asof)
            key = key_players(team, season, asof)
            km = key_player_models(team, season, asof)
            units = {}
            for side in ("offense", "defense"):
                off = side == "offense"
                # opponent/site are placeholders: dist and site_base are recomputed per matchup in the browser
                rows = unit_ingredients(team, side, season, asof, team, "home", None,
                                        qb if off else None, key if off else None)
                info = {"rows": rows, "baseline": baseline_for(team, side, season, prof),
                        "expected_qb": qb if off else None, "key": key if off else None, "km": km if off else {}}
                up = unit_payload(info, set(), set()) if not rows.empty else {
                    "expected_qb": info["expected_qb"], "qbs": [], "key": info["key"], "km": {},
                    "baseline": info["baseline"].to_dict(), "games": []}
                for gm, r in zip(up["games"], rows.itertuples(index=False)):
                    for k in ("dist", "site_base", "common", "wx_d2"):
                        gm.pop(k, None)
                    # past-game conditions, so the browser can redo the weather match for a chosen scenario
                    gm["wx"] = {"indoors": bool(r.indoors) if not pd.isna(r.indoors) else False,
                                "temp": r.temp, "wind": r.wind, "precip": r.precip}
                units[side] = up
        except Exception as e:  # one bad team shouldn't block the rest
            print(f"  skip team {team}: {e}")
            continue
        slug = _team_slug(team)
        data = {"team": team, "slug": slug, "conference": conf.get(team), "asof": asof.isoformat(),
                "form": recent_form(team, season, asof, n=5), "units": units}
        (out_dir / f"{slug}.json").write_text(json.dumps(_clean(data), separators=(",", ":")))
        index.append({"team": team, "slug": slug, "conference": conf.get(team),
                      "games": len(units["offense"]["games"])})
    keep = {f"{x['slug']}.json" for x in index}
    for f in out_dir.glob("*.json"):
        if f.name not in keep:
            f.unlink()
    refs = percentile_refs(season, asof)
    league = {
        "season": season, "asof": asof.isoformat(), "teams": index,
        "vec": vec, "sim_weights": [w for _, w in SIM_FEATURES],
        "site_match": {f"{a}|{b}": v for (a, b), v in SITE_MATCH.items()},
        "fbs": fbs,
        "refs": {f"{side}|{mt}": refs[(side, mt)].tolist() for side in ("offense", "defense") for mt in METRICS},
        "lg_epa": prof.attrs.get("lg_epa", 0.0),
        "params": asdict(p), "points_model": asdict(pm), "best_a": {k: v for k, v in (BEST_A or {}).items()},
        "results": {t: float(v) for t, v in results_ratings(season, asof.isoformat())["rating"].items()},
        "blend": asdict(load_blend()) if load_blend() else None,
    }
    (SITE_DATA / "league.json").write_text(json.dumps(_clean(league), separators=(",", ":")))
    print(f"exported {len(index)} teams for custom matchups (as of {asof:%Y-%m-%d %H:%M} UTC)")
    return len(index)


def export_colors(season: int) -> None:
    """site/data/colors.json: {team: [primary, alternate]} from data/derived/team_colors_<season>.csv.
    The matchup page picks readable, distinct shades from these (site/colors.js)."""
    path = ROOT / "data" / "derived" / f"team_colors_{season}.csv"
    if not path.exists():
        print(f"  no {path.name}; matchup pages keep the default colors")
        return
    df = pd.read_csv(path).fillna("")
    norm = lambda c: ("#" + c.lstrip("#").lower()) if len(c.lstrip("#")) == 6 else None
    out = {r.team: [norm(str(r.color)), norm(str(r.alt_color))] for r in df.itertuples(index=False)}
    (SITE_DATA / "colors.json").write_text(json.dumps(out, separators=(",", ":"), sort_keys=True))
    print(f"exported colors for {len(out)} teams")


def export_rankings(season: int, p: Params, pm: PointsModel) -> None:
    """site/data/rankings.json: every FBS team vs every other, neutral site, good weather (engine/rankings.py)."""
    from engine.rankings import GOOD_WEATHER, power_ratings
    asof = pd.Timestamp.now("UTC").floor("h")
    if profiles_asof(season, asof.isoformat()).empty:
        return
    df, _, _ = power_ratings(season, asof, p, pm, load_blend())
    _, g = load()
    s = g[(g.season == season) & (g.completed == True)]  # noqa: E712
    conf = {**dict(zip(s.awayTeam, s.awayConference)), **dict(zip(s.homeTeam, s.homeConference))}
    def record(t):
        h, a = s[s.homeTeam == t], s[s.awayTeam == t]
        w = int((h.homePoints > h.awayPoints).sum() + (a.awayPoints > a.homePoints).sum())
        return [w, len(h) + len(a) - w]
    teams = [{"rank": int(r.rank), "team": r.team, "conference": conf.get(r.team), "record": record(r.team),
              "rating": r.rating, "pbp_rating": r.pbp_rating, "results_rating": r.results_rating,
              "pts_for": r.pts_for, "pts_against": r.pts_against, "win_pct": r.win_pct,
              "favored": int(r.favored), "opponents": int(r.opponents)} for r in df.itertuples(index=False)]
    bl = load_blend()
    out = {"season": season, "asof": asof.isoformat(), "weather": GOOD_WEATHER, "teams": teams,
           "blend": asdict(bl) if bl else None}
    (SITE_DATA / "rankings.json").write_text(json.dumps(_clean(out), separators=(",", ":")))
    print(f"ranked {len(teams)} teams; #1 {teams[0]['team']} ({teams[0]['rating']:+.1f})")


def main(argv):
    u, g = load()
    season = int(argv[0]) if argv else int(g[g.completed == True].season.max())  # noqa: E712
    if len(argv) > 1:
        stype, week = (argv[2] if len(argv) > 2 else "regular"), int(argv[1])
    else:
        stype, week = pick_week(season)
    p, pm, results = load_tuned()
    fbs = fbs_teams(season)
    gs = g[(g.season == season) & (g.week == week) & (g.seasonType == stype) &
           g.homeTeam.isin(fbs) & g.awayTeam.isin(fbs)].sort_values("date")
    (SITE_DATA / "games").mkdir(parents=True, exist_ok=True)
    index = []
    for row in gs.itertuples(index=False):
        try:
            data = export_game(row, p, pm)
        except Exception as e:  # keep going; one bad game shouldn't block the week
            print(f"  skip {row.awayTeam} at {row.homeTeam}: {e}")
            continue
        (SITE_DATA / "games" / f"{row.gameId}.json").write_text(json.dumps(data, separators=(",", ":")))
        pr = data["projection"]
        index.append({"id": int(row.gameId), "start": data["game"]["start"], "home": row.homeTeam,
                      "away": row.awayTeam, "neutral": bool(row.neutralSite),
                      "home_pts": pr["home_pts"], "away_pts": pr["away_pts"], "home_wp": pr["home_wp"],
                      "spread_home": data["game"]["spread_home"], "home_conf": row.homeConference,
                      "away_conf": row.awayConference})
        print(f"  {row.awayTeam} at {row.homeTeam}: {pr['away_pts']:.0f}-{pr['home_pts']:.0f}")
    meta = {"season": season, "week": week, "season_type": stype, "generated": pd.Timestamp.now("UTC").isoformat(),
            "labels": LABELS, "metrics": METRICS, "count_col": COUNT_COL,
            "higher_is_better": {f"{s}|{m}": higher_is_better(s, m) for s in ("offense", "defense") for m in METRICS},
            "backtest": results}
    (SITE_DATA / "week.json").write_text(json.dumps(_clean({"meta": meta, "games": index}), separators=(",", ":")))
    # Drop game files from earlier weeks
    keep = {f"{x['id']}.json" for x in index}
    for f in (SITE_DATA / "games").glob("*.json"):
        if f.name not in keep:
            f.unlink()
    print(f"exported {len(index)} games for {season} {stype} week {week}")
    export_teams(season, p, pm)
    export_colors(season)
    export_rankings(season, p, pm)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
