"""Build derived data for one or more seasons.

Usage: python -m pipeline.build_season 2026 [2025 ...]
Writes (derived numbers only, safe to publish):
  data/derived/unit_games_<year>.csv  one row per game x team x side (offense/defense)
  data/derived/games_<year>.csv       schedule incl. upcoming games, weather, consensus spread
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.cfbd_client import CFBDClient, CFBDError
from pipeline.ingest import season_games, week_player_passing, week_plays
from pipeline.unit_stats import unit_game_stats

OUT = Path("data/derived")


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def season_weather(client: CFBDClient, year: int) -> pd.DataFrame:
    rows = []
    for st in ("regular", "postseason"):
        try:
            data = client.get("/games/weather", year=year, seasonType=st)
        except CFBDError as e:
            print(f"  weather {year} {st}: {e}")
            continue
        if data and st == "regular":
            print(f"  weather fields: {sorted(data[0].keys())}")
        for w in data:
            rows.append({
                "gameId": _first(w, "id", "gameId"),
                "indoors": bool(_first(w, "gameIndoors", "indoors") or False),
                "temp": _first(w, "temperature"),
                "wind": _first(w, "windSpeed"),
                "precip": _first(w, "precipitation"),
                "condition": _first(w, "weatherCondition", "condition"),
            })
    return pd.DataFrame(rows, columns=["gameId", "indoors", "temp", "wind", "precip", "condition"])


def season_spreads(client: CFBDClient, year: int) -> pd.DataFrame:
    """Consensus (mean across books) home spread; negative = home favored."""
    rows = []
    for st in ("regular", "postseason"):
        try:
            data = client.get("/lines", year=year, seasonType=st)
        except CFBDError as e:
            print(f"  lines {year} {st}: {e}")
            continue
        for g in data:
            sp = [l.get("spread") for l in g.get("lines", []) if isinstance(l.get("spread"), (int, float))]
            tot = [l.get("overUnder") for l in g.get("lines", []) if isinstance(l.get("overUnder"), (int, float))]
            rows.append({"gameId": g.get("id"),
                         "spread_home": float(np.mean(sp)) if sp else np.nan,
                         "total": float(np.mean(tot)) if tot else np.nan})
    return pd.DataFrame(rows, columns=["gameId", "spread_home", "total"])


def build(client: CFBDClient, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = season_games(client, year)
    weather = season_weather(client, year)
    spreads = season_spreads(client, year)
    sched = (games.rename(columns={"id": "gameId"})
             .merge(weather, on="gameId", how="left")
             .merge(spreads, on="gameId", how="left"))

    done = games[games["completed"] == True]  # noqa: E712
    weeks = done[["seasonType", "week"]].drop_duplicates().sort_values(["seasonType", "week"])
    plays, qbs = [], []
    for st, wk in weeks.itertuples(index=False):
        plays += week_plays(client, year, int(wk), st)
        qbs.append(week_player_passing(client, year, int(wk), st))
    units = unit_game_stats(pd.DataFrame(plays))

    g = sched[sched["completed"] == True]  # noqa: E712
    home = g.assign(team=g.homeTeam, opponent=g.awayTeam, pts=g.homePoints, opp_pts=g.awayPoints,
                    site=np.where(g.neutralSite, "neutral", "home"))
    away = g.assign(team=g.awayTeam, opponent=g.homeTeam, pts=g.awayPoints, opp_pts=g.homePoints,
                    site=np.where(g.neutralSite, "neutral", "away"))
    ctx = pd.concat([home, away])[["gameId", "season", "week", "seasonType", "startDate", "team",
                                    "site", "pts", "opp_pts", "indoors", "temp", "wind", "precip"]]
    units = units.merge(ctx, on=["gameId", "team"], how="left")

    qb = pd.concat(qbs, ignore_index=True) if qbs else pd.DataFrame(columns=["gameId", "team", "qb", "qb_id"])
    units = units.merge(qb[["gameId", "team", "qb", "qb_id"]], on=["gameId", "team"], how="left")
    opp_qb = qb.rename(columns={"team": "opponent", "qb": "opp_qb", "qb_id": "opp_qb_id"})
    units = units.merge(opp_qb[["gameId", "opponent", "opp_qb", "opp_qb_id"]], on=["gameId", "opponent"], how="left")
    units = units.sort_values(["team", "side", "startDate"]).reset_index(drop=True)
    return units, sched


def main(argv: list[str]) -> int:
    now = pd.Timestamp.now("UTC")
    current = now.year if now.month >= 7 else now.year - 1   # season year (Jan bowls belong to prior year)
    years = [int(a) for a in argv] or [current]
    client = CFBDClient()
    info = client.get("/info", refresh=True)
    print(f"CFBD tier: {info.get('tierName')} (limit {info.get('monthlyLimit')}, remaining {info.get('remainingCalls')})")
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        units, sched = build(client, y)
        units.round(4).to_csv(OUT / f"unit_games_{y}.csv", index=False)
        sched.to_csv(OUT / f"games_{y}.csv", index=False)
        wx = sched["temp"].notna().mean() if "temp" in sched else 0
        print(f"{y}: {len(units)} unit-game rows, {len(sched)} games, weather coverage {wx:.0%}, "
              f"spread coverage {sched['spread_home'].notna().mean():.0%}")
    print(f"API calls used: {client.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
