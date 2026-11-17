"""Download a season from CFBD (cached under data/raw/, never committed)."""
from __future__ import annotations

import pandas as pd

from pipeline.cfbd_client import CFBDClient


def season_games(client: CFBDClient, year: int) -> pd.DataFrame:
    rows = []
    for st in ("regular", "postseason"):
        rows += client.get("/games", year=year, seasonType=st)
    g = pd.DataFrame(rows)
    keep = ["id", "season", "week", "seasonType", "startDate", "neutralSite", "conferenceGame",
            "venue", "venueId", "homeTeam", "homeId", "homeClassification", "homeConference", "homePoints",
            "awayTeam", "awayId", "awayClassification", "awayConference", "awayPoints", "completed"]
    return g[[c for c in keep if c in g.columns]]


def week_plays(client: CFBDClient, year: int, week: int, season_type: str = "regular") -> list[dict]:
    return client.get("/plays", year=year, week=week, seasonType=season_type)


def week_player_passing(client: CFBDClient, year: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Passer with the most attempts for each team-game (our 'starting QB' tag)."""
    data = client.get("/games/players", year=year, week=week, seasonType=season_type)
    rows = []
    for game in data:
        for team in game.get("teams", []):
            for cat in team.get("categories", []):
                if cat.get("name") != "passing":
                    continue
                atts: dict[str, dict] = {}
                for typ in cat.get("types", []):
                    if typ.get("name") != "C/ATT":
                        continue
                    for a in typ.get("athletes", []):
                        try:
                            att = int(str(a.get("stat", "0/0")).split("/")[1])
                        except (IndexError, ValueError):
                            att = 0
                        atts[a.get("name")] = {"id": a.get("id"), "att": att}
                if atts:
                    name, best = max(atts.items(), key=lambda kv: kv[1]["att"])
                    rows.append({"gameId": game.get("id"), "team": team.get("team"),
                                 "qb": name, "qb_id": best["id"], "qb_att": best["att"]})
    return pd.DataFrame(rows)
