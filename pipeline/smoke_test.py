"""Smoke test: confirm the key works and learn the shape of each endpoint we need.

Prints only field names, counts and small derived summaries: never raw records
(workflow logs on a public repo are public).
"""
from __future__ import annotations

import sys

from pipeline.cfbd_client import CFBDClient, CFBDError

YEAR = 2026
WEEK = 11          # BYU vs Baylor (Nov 15)
NEXT_WEEK = 12     # BYU at Kansas


def keys_of(rows):
    if isinstance(rows, list) and rows:
        return sorted(rows[0].keys()) if isinstance(rows[0], dict) else type(rows[0]).__name__
    if isinstance(rows, dict):
        return sorted(rows.keys())
    return repr(rows)[:80]


def probe(client, label, path, **params):
    print(f"\n=== {label}: GET {path} {params}")
    try:
        data = client.get(path, refresh=True, **params)
    except CFBDError as e:
        print(f"  FAILED: {e}")
        return None
    n = len(data) if isinstance(data, list) else 1
    print(f"  records: {n}")
    print(f"  fields:  {keys_of(data)}")
    return data


def main() -> int:
    client = CFBDClient(use_cache=False)

    info = probe(client, "Account info", "/info")
    if isinstance(info, dict):
        safe = {k: v for k, v in info.items() if k in ("patronLevel", "tierName", "monthlyLimit", "remainingCalls")}
        print(f"  account: {safe}")

    games = probe(client, "Games (FBS, week 11)", "/games", year=YEAR, week=WEEK, classification="fbs")
    if games:
        g = [x for x in games if "BYU" in (x.get("homeTeam"), x.get("awayTeam"))]
        if g:
            x = g[0]
            print(f"  BYU game: {x.get('awayTeam')} {x.get('awayPoints')} at {x.get('homeTeam')} {x.get('homePoints')} neutral={x.get('neutralSite')}")

    upcoming = probe(client, "Games (week 12)", "/games", year=YEAR, week=NEXT_WEEK, team="BYU")
    if upcoming:
        x = upcoming[0]
        print(f"  next: {x.get('awayTeam')} at {x.get('homeTeam')} start={x.get('startDate')}")

    plays = probe(client, "Plays (BYU, week 11)", "/plays", year=YEAR, week=WEEK, team="BYU")
    if plays:
        ppa = [p.get("ppa") for p in plays if isinstance(p.get("ppa"), (int, float))]
        off = [p for p in plays if p.get("offense") == "BYU" and isinstance(p.get("ppa"), (int, float))]
        print(f"  plays with ppa: {len(ppa)} of {len(plays)}")
        if off:
            print(f"  BYU offense: {len(off)} plays, mean PPA {sum(p['ppa'] for p in off)/len(off):+.3f}")
        types = {}
        for p in plays:
            types[p.get("playType")] = types.get(p.get("playType"), 0) + 1
        print(f"  play types: {dict(sorted(types.items(), key=lambda kv: -kv[1])[:12])}")

    probe(client, "Weather (week 11)", "/games/weather", year=YEAR, week=WEEK)
    probe(client, "Lines (BYU, week 12)", "/lines", year=YEAR, week=NEXT_WEEK, team="BYU")
    probe(client, "Advanced game stats (Kansas)", "/stats/game/advanced", year=YEAR, team="Kansas")
    probe(client, "Game player stats (BYU wk 11)", "/games/players", year=YEAR, week=WEEK, team="BYU")

    print(f"\nAPI calls used by this run: {client.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
