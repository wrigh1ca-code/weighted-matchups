"""Save each team's primary and alternate colors from CFBD (/teams/fbs).

Colors only change between seasons, so this spends one API call per season: it skips the call
when data/derived/team_colors_<year>.csv already exists (pass --refresh to force).
    python -m pipeline.team_colors [year] [--refresh]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.cfbd_client import CFBDClient

OUT = Path("data/derived")


def main(argv: list[str]) -> int:
    refresh = "--refresh" in argv
    years = [int(a) for a in argv if a.isdigit()]
    today = date.today()
    year = years[0] if years else (today.year if today.month >= 3 else today.year - 1)
    path = OUT / f"team_colors_{year}.csv"
    if path.exists() and not refresh:
        print(f"{path} exists; no API call")
        return 0
    client = CFBDClient()
    teams = client.get("/teams/fbs", year=year, refresh=True)
    rows = [{"team": t.get("school"), "color": t.get("color"), "alt_color": t.get("alternateColor")}
            for t in teams if t.get("school")]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("team").to_csv(path, index=False)
    print(f"saved colors for {len(rows)} teams to {path} (API calls: {client.calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
