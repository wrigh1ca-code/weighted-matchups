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


def _num(x) -> float:
    try:
        return float(str(x).split("/")[0])
    except (TypeError, ValueError):
        return 0.0


def week_player_passing(client: CFBDClient, year: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Per team-game: starting QB (most attempts) plus every rusher and receiver with their volume.

    rush_list = "Name:carries|..."  rec_list = "Name:receptions:yards|..."
    (Box scores list only players who recorded a stat, so absence ~ did not play or no touches.)
    """
    data = client.get("/games/players", year=year, week=week, seasonType=season_type)
    rows = []
    for game in data:
        for team in game.get("teams", []):
            stats: dict[str, dict] = {}
            for cat in team.get("categories", []):
                cname = cat.get("name")
                for typ in cat.get("types", []):
                    tname = typ.get("name")
                    for a in typ.get("athletes", []):
                        nm = (a.get("name") or "").strip()
                        if not nm or nm.lower() == "team":
                            continue
                        d = stats.setdefault(nm, {"id": a.get("id")})
                        stat = a.get("stat", "0")
                        if cname == "passing" and tname == "C/ATT":
                            try:
                                d["att"] = int(str(stat).split("/")[1])
                            except (IndexError, ValueError):
                                d["att"] = 0
                        elif cname == "rushing" and tname == "CAR":
                            d["car"] = _num(stat)
                        elif cname == "receiving" and tname == "REC":
                            d["rec"] = _num(stat)
                        elif cname == "receiving" and tname == "YDS":
                            d["rec_yds"] = _num(stat)
            passers = {k: v for k, v in stats.items() if v.get("att", 0) > 0}
            qb, qb_id, qb_att = None, None, 0
            if passers:
                qb, best = max(passers.items(), key=lambda kv: kv[1]["att"])
                qb_id, qb_att = best["id"], best["att"]
            rush = "|".join(f"{k.replace('|', ' ').replace(':', ' ')}:{int(v['car'])}"
                            for k, v in stats.items() if v.get("car", 0) > 0)
            rec = "|".join(f"{k.replace('|', ' ').replace(':', ' ')}:{int(v.get('rec', 0))}:{int(v.get('rec_yds', 0))}"
                           for k, v in stats.items() if v.get("rec", 0) > 0)
            rows.append({"gameId": game.get("id"), "team": team.get("team"), "qb": qb, "qb_id": qb_id,
                         "qb_att": qb_att, "rush_list": rush, "rec_list": rec})
    return pd.DataFrame(rows)
