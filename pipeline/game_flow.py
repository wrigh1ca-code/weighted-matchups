"""Game flow: how each game went, not just how it ended.

For every completed game, from the play-by-play:
  avg_lead      the home team's lead averaged over the 60 minutes of regulation (time-weighted, points)
  control       "game control": the home team's in-game win probability averaged over regulation
                (time-weighted, 0-1), from a win-probability model with no pregame information
  final_margin  home points - away points

The win-probability model is a logistic regression fit on the season's own plays: score margin, time
left, possession and field position, scaled by the square root of the time left (so a 7-point lead
matters more late than early). It has no intercept and no team strength, so both teams start at 50%:
control measures what happened on the field, not who was expected to win.

    python -m pipeline.game_flow --backfill     # compute any season from 2024 on that is missing
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/derived")
REG = 3600.0          # seconds of regulation
EPS = 0.01            # keeps 1/sqrt(time left) finite at 0:00


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def play_states(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per play with the game state from the home team's point of view."""
    p = plays.copy()
    clock = p["clock"].apply(lambda c: (c or {}) if isinstance(c, dict) else {})
    p["sec_in_period"] = clock.apply(lambda c: _num(c.get("minutes")) * 60 + _num(c.get("seconds")))
    p["period"] = pd.to_numeric(p["period"], errors="coerce")
    p = p[p.period.notna() & p.sec_in_period.notna() & p.gameId.notna()]
    reg = p.period <= 4
    p["t_left"] = np.where(reg, (4 - p.period) * 900 + p.sec_in_period.clip(0, 900), 0.0)
    home_off = p["offense"] == p["home"]
    os_, ds_ = pd.to_numeric(p["offenseScore"], errors="coerce"), pd.to_numeric(p["defenseScore"], errors="coerce")
    p["home_score"] = np.where(home_off, os_, ds_)
    p["away_score"] = np.where(home_off, ds_, os_)
    p["lead"] = p.home_score - p.away_score
    p["pos"] = np.where(home_off, 1.0, -1.0)
    ytg = pd.to_numeric(p.get("yardsToGoal"), errors="coerce").clip(0, 100)
    p["field"] = p.pos * (100 - ytg.fillna(75)) / 100           # +1 = home at the goal line, -1 = away there
    order = [c for c in ("gameId", "period", "driveNumber", "playNumber", "id") if c in p]
    p = p.sort_values(["gameId", "period", "t_left"] + order[2:], ascending=[True, True, False] + [True] * (len(order) - 2))
    return p[p.lead.notna()][["gameId", "home", "away", "period", "t_left", "lead", "pos", "field"]]


def wp_features(s: pd.DataFrame) -> np.ndarray:
    r = 1 / np.sqrt(s.t_left.to_numpy(float) / REG + EPS)
    return np.column_stack([s.lead.to_numpy(float) * r, s.field.to_numpy(float) * r, s.pos.to_numpy(float) * r,
                            s.lead.to_numpy(float)])


def fit_wp(states: pd.DataFrame, home_won: pd.Series) -> np.ndarray:
    """Logistic regression (no intercept) by Newton's method, on regulation plays of decided games."""
    s = states[states.period <= 4].merge(home_won.rename("y"), left_on="gameId", right_index=True)
    s = s[s.y.notna()]
    X, y = wp_features(s), s.y.to_numpy(float)
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        z = np.clip(X @ beta, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        W = pr * (1 - pr) + 1e-9
        step = np.linalg.solve((X.T * W) @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - pr))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def wp(states: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(wp_features(states) @ beta, -30, 30)))


def game_flow(plays: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """games: CFBD /games rows (id, homeTeam, awayTeam, homePoints, awayPoints, completed)."""
    g = games[games.completed == True].rename(columns={"id": "gameId"})  # noqa: E712
    g = g[g.homePoints.notna() & g.awayPoints.notna()]
    final = (g.set_index("gameId").homePoints - g.set_index("gameId").awayPoints)
    states = play_states(plays[plays.gameId.isin(g.gameId)])
    home_won = (final > 0).astype(float).where(final != 0)
    beta = fit_wp(states, home_won)
    s = states[states.period <= 4].copy()
    s["wp"] = wp(s, beta)
    # time-weight: each state lasts until the next play (or the end of regulation)
    nxt = s.groupby("gameId").t_left.shift(-1).fillna(0.0)
    s["dt"] = (s.t_left - nxt).clip(lower=0)
    # the opening stretch before the first snap counts as 0-0, 50%
    first = s.groupby("gameId").t_left.first()
    agg = s.assign(wl=s.dt * s.lead, ww=s.dt * s.wp).groupby("gameId")[["wl", "ww", "dt"]].sum()
    lead0 = (REG - first.reindex(agg.index)).clip(lower=0)
    total = agg.dt + lead0
    out = pd.DataFrame({
        "gameId": agg.index,
        "avg_lead": (agg.wl / total).round(3),
        "control": ((agg.ww + 0.5 * lead0) / total).round(4),
        "final_margin": final.reindex(agg.index).to_numpy(),
        "plays": s.groupby("gameId").size().reindex(agg.index).to_numpy(),
        "coverage": (agg.dt / REG).clip(upper=1).round(3),
    }).reset_index(drop=True)
    out = out.merge(g[["gameId", "homeTeam", "awayTeam"]], on="gameId", how="left")
    return out, beta


def _season_year(today: date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def backfill(first: int = 2024) -> int:
    """Compute game_flow_<year>.csv for each season from `first` that doesn't have it yet."""
    from pipeline.cfbd_client import CFBDClient
    from pipeline.ingest import season_games, week_plays
    client = None
    for y in range(first, _season_year(date.today()) + 1):
        path = OUT / f"game_flow_{y}.csv"
        if path.exists():
            print(f"{path.name} exists; skipped")
            continue
        client = client or CFBDClient()
        games = season_games(client, y)
        done = games[games["completed"] == True]  # noqa: E712
        weeks = done[["seasonType", "week"]].drop_duplicates().sort_values(["seasonType", "week"])
        plays = []
        for st, wk in weeks.itertuples(index=False):
            plays += week_plays(client, y, int(wk), st)
        flow, beta = game_flow(pd.DataFrame(plays), games)
        OUT.mkdir(parents=True, exist_ok=True)
        flow.to_csv(path, index=False)
        print(f"{y}: game flow for {len(flow)} games (median coverage {flow.coverage.median():.2f}); "
              f"WP coefficients {np.round(beta, 4).tolist()}")
    if client:
        print(f"API calls used: {client.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(backfill() if "--backfill" in sys.argv else 0)
