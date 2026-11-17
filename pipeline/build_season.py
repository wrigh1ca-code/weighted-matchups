"""Build derived per-game unit stats for one or more seasons.

Usage: python -m pipeline.build_season 2026 [2025 ...]
Writes data/derived/unit_games_<year>.csv (derived numbers only, safe to publish).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline.cfbd_client import CFBDClient
from pipeline.ingest import season_games, week_player_passing, week_plays
from pipeline.unit_stats import unit_game_stats

OUT = Path("data/derived")


def build(client: CFBDClient, year: int) -> pd.DataFrame:
    games = season_games(client, year)
    done = games[games["completed"] == True]  # noqa: E712
    weeks = (done[["seasonType", "week"]].drop_duplicates().sort_values(["seasonType", "week"]))

    plays, qbs = [], []
    for st, wk in weeks.itertuples(index=False):
        plays += week_plays(client, year, int(wk), st)
        qbs.append(week_player_passing(client, year, int(wk), st))
    units = unit_game_stats(pd.DataFrame(plays))

    # Game context: week, date, site from this team's point of view, final score
    g = done.rename(columns={"id": "gameId"})
    home = g.assign(team=g.homeTeam, opponent=g.awayTeam, pts=g.homePoints, opp_pts=g.awayPoints,
                    site=g.neutralSite.map({True: "neutral", False: "home"}))
    away = g.assign(team=g.awayTeam, opponent=g.homeTeam, pts=g.awayPoints, opp_pts=g.homePoints,
                    site=g.neutralSite.map({True: "neutral", False: "away"}))
    ctx = pd.concat([home, away])[["gameId", "season", "week", "seasonType", "startDate", "team",
                                    "site", "pts", "opp_pts"]]
    units = units.merge(ctx, on=["gameId", "team"], how="left")

    qb = pd.concat(qbs, ignore_index=True) if qbs else pd.DataFrame(columns=["gameId", "team", "qb"])
    # Offense rows get the team's own QB; defense rows get the opposing QB they faced
    off_qb = qb.rename(columns={"qb": "qb", "qb_id": "qb_id", "qb_att": "qb_att"})
    units = units.merge(off_qb[["gameId", "team", "qb", "qb_id"]], on=["gameId", "team"], how="left")
    opp_qb = qb.rename(columns={"team": "opponent", "qb": "opp_qb", "qb_id": "opp_qb_id"})
    units = units.merge(opp_qb[["gameId", "opponent", "opp_qb", "opp_qb_id"]], on=["gameId", "opponent"], how="left")
    return units.sort_values(["team", "side", "week"]).reset_index(drop=True)


def main(argv: list[str]) -> int:
    years = [int(a) for a in argv] or [2026]
    client = CFBDClient()
    info = client.get("/info", refresh=True)
    print(f"CFBD tier: {info.get('tierName')} (limit {info.get('monthlyLimit')}, remaining {info.get('remainingCalls')})")
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        df = build(client, y)
        path = OUT / f"unit_games_{y}.csv"
        df.round(4).to_csv(path, index=False)
        print(f"{y}: {len(df)} unit-game rows, {df['team'].nunique()} teams -> {path}")
    print(f"API calls used: {client.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
