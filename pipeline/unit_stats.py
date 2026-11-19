"""Turn CFBD play-by-play into per-game stats for every offense and defense.

One row per (game, team, side): side = "offense" (what the team did with the ball)
or "defense" (what its opponents did against it). The weighting engine later
averages these rows across games, weighting each by plays x game weight.

Metric definitions (all per scrimmage play unless noted):
- epa_play          mean CFBD PPA (their EPA)
- sr                success rate: 1st down gains >=50% of distance, 2nd >=70%, 3rd/4th 100%
- early_epa         EPA/play on 1st and 2nd down
- late_sr           success rate on 3rd and 4th down
- pass_epa, rush_epa, pass_sr
- yds_db            yards per dropback (sacks included)
- explosive         share of plays with rush >=12 yds or pass >=16 yds
- rush_expl, pass_expl
- stuff             share of rushes gaining <=0 yds
- opportunity       share of rushes gaining >=4 yds
- line_yards        Football Outsiders-style line yards per rush
- havoc             (sacks + TFL + interceptions + fumbles) / plays  [approximate: no PBU data]
- non_expl_epa      EPA/play on non-explosive plays
Garbage time is removed with the usual margin cutoffs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PASS_TYPES = {
    "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
    "Pass Interception Return", "Interception Return Touchdown", "Pass Interception",
    "Interception", "Pass",
}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
FUMBLE_TYPES = {"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"}
INT_TYPES = {"Pass Interception Return", "Interception Return Touchdown", "Pass Interception", "Interception"}

METRICS = [
    "epa_play", "sr", "early_epa", "late_sr", "pass_epa", "rush_epa", "pass_sr", "yds_db",
    "explosive", "rush_expl", "pass_expl", "stuff", "opportunity", "line_yards", "havoc",
    "non_expl_epa",
]


def classify(plays: pd.DataFrame) -> pd.DataFrame:
    """Keep scrimmage plays and tag each as pass or rush."""
    p = plays.copy()
    text = p["playText"].fillna("").str.lower()
    is_pass = p["playType"].isin(PASS_TYPES)
    is_rush = p["playType"].isin(RUSH_TYPES)
    fum = p["playType"].isin(FUMBLE_TYPES)
    fum_pass = fum & text.str.contains(r"\bpass|sacked|scramble", regex=True)
    is_pass = is_pass | fum_pass
    is_rush = is_rush | (fum & ~fum_pass)
    p = p[is_pass | is_rush].copy()
    p["is_pass"] = is_pass[p.index]
    p["is_rush"] = ~p["is_pass"]
    p["is_sack"] = p["playType"].eq("Sack") | text[p.index].str.contains("sacked")
    p["is_int"] = p["playType"].isin(INT_TYPES)
    p["is_fumble"] = p["playType"].isin(FUMBLE_TYPES)
    p = p[p["down"].between(1, 4) & p["ppa"].notna()]
    return p


def drop_garbage_time(p: pd.DataFrame) -> pd.DataFrame:
    margin = (p["offenseScore"] - p["defenseScore"]).abs()
    cutoff = p["period"].map({1: 99, 2: 38, 3: 28, 4: 22}).fillna(99)
    return p[margin <= cutoff]


def add_flags(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    y, d, dist = p["yardsGained"], p["down"], p["distance"].clip(lower=1)
    need = np.where(d == 1, 0.5, np.where(d == 2, 0.7, 1.0)) * dist
    p["success"] = (y >= need).astype(float)
    p["expl"] = ((p["is_rush"] & (y >= 12)) | (p["is_pass"] & (y >= 16))).astype(float)
    p["stuffed"] = (p["is_rush"] & (y <= 0)).astype(float)
    p["opp"] = (p["is_rush"] & (y >= 4)).astype(float)
    ly = np.where(y < 0, 1.2 * y, np.where(y <= 4, y, np.where(y <= 10, 4 + 0.5 * (y - 4), 7)))
    p["line_y"] = np.where(p["is_rush"], ly, np.nan)
    p["havoc_ev"] = (p["is_sack"] | p["is_int"] | p["is_fumble"] | (p["is_rush"] & (y < 0))).astype(float)
    return p


def _agg(g: pd.DataFrame) -> pd.Series:
    rush, pas = g[g["is_rush"]], g[g["is_pass"]]
    early, late = g[g["down"] <= 2], g[g["down"] >= 3]
    nonexp = g[g["expl"] == 0]

    def m(s):
        return float(s.mean()) if len(s) else np.nan

    return pd.Series({
        "plays": len(g), "rushes": len(rush), "dropbacks": len(pas),
        "epa_play": m(g["ppa"]), "sr": m(g["success"]),
        "early_epa": m(early["ppa"]), "late_sr": m(late["success"]),
        "pass_epa": m(pas["ppa"]), "rush_epa": m(rush["ppa"]), "pass_sr": m(pas["success"]),
        "yds_db": m(pas["yardsGained"]),
        "explosive": m(g["expl"]), "rush_expl": m(rush["expl"]), "pass_expl": m(pas["expl"]),
        "stuff": m(rush["stuffed"]), "opportunity": m(rush["opp"]), "line_yards": m(rush["line_y"]),
        "havoc": m(g["havoc_ev"]), "non_expl_epa": m(nonexp["ppa"]),
    })


def unit_game_stats(plays: pd.DataFrame) -> pd.DataFrame:
    """plays: CFBD /plays records for any set of games. Returns offense + defense rows."""
    p = add_flags(drop_garbage_time(classify(plays)))
    off = p.groupby(["gameId", "offense", "defense"]).apply(_agg, include_groups=False).reset_index()
    off = off.rename(columns={"offense": "team", "defense": "opponent"})
    off["side"] = "offense"
    dfn = off.rename(columns={"team": "opponent", "opponent": "team"}).copy()
    dfn["side"] = "defense"
    cols = ["gameId", "team", "opponent", "side", "plays", "rushes", "dropbacks"] + METRICS
    return pd.concat([off[cols], dfn[cols]], ignore_index=True)


# ---------------------------------------------------------------- player attribution

import re

_NAME = r"([A-Z][\w'.\-]*(?:\s+(?:[A-Z][\w'.\-]*|de|la|le|van|von|St\.))*(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?)"
RUSH_RES = [re.compile(r"^" + _NAME + r"\s+(?:run|rush|rushed|scramble[sd]?)\b"),
            re.compile(r"^" + _NAME + r"\s+\d+\s+(?:Yd|Yard|yd|yard)s?\s+(?:Run|Rush|run|rush)")]
TARGET_RES = [re.compile(r"pass (?:complete|completed|incomplete|caught|intended|thrown) (?:to|for) " + _NAME),
              re.compile(r"^" + _NAME + r"\s+\d+\s+(?:Yd|Yard|yd|yard)s?\s+(?:pass|Pass) from")]
PASSER_RES = [re.compile(r"^" + _NAME + r"\s+(?:pass|sacked|scramble)"),
              re.compile(r"(?:pass|Pass) from " + _NAME)]


def _first(res, text):
    for r in res:
        m = r.search(text)
        if m:
            return m.group(1).strip()
    return None


def player_game_stats(plays: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (game, offense, player, kind) totals: kind = 'rush' (ball carrier), 'target' (intended
    receiver) or 'pass' (passer/dropback). Same scrimmage + garbage-time filters as unit stats."""
    p = add_flags(drop_garbage_time(classify(plays)))
    text = p["playText"].fillna("")
    rusher = np.where(p["is_rush"], text.map(lambda t: _first(RUSH_RES, t)), None)
    target = np.where(p["is_pass"] & ~p["is_sack"], text.map(lambda t: _first(TARGET_RES, t)), None)
    passer = np.where(p["is_pass"], text.map(lambda t: _first(PASSER_RES, t)), None)
    rows, cov, misses = [], {}, {}
    for kind, who, mask in (("rush", rusher, p["is_rush"].to_numpy()),
                            ("target", target, (p["is_pass"] & ~p["is_sack"]).to_numpy()),
                            ("pass", passer, p["is_pass"].to_numpy())):
        d = pd.DataFrame({"gameId": p["gameId"].to_numpy(), "team": p["offense"].to_numpy(),
                          "player": who, "epa": p["ppa"].to_numpy(), "success": p["success"].to_numpy(),
                          "yards": p["yardsGained"].to_numpy()})[mask]
        cov[kind] = float(d["player"].notna().mean()) if len(d) else 0.0
        misses[kind] = text.to_numpy()[mask][pd.isna(d["player"]).to_numpy()][:4].tolist()
        d = d.dropna(subset=["player"])
        g = d.groupby(["gameId", "team", "player"]).agg(n=("epa", "size"), epa_sum=("epa", "sum"),
                                                         succ_sum=("success", "sum"), yds=("yards", "sum")).reset_index()
        g["kind"] = kind
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    return out, {"coverage": cov, "unparsed_examples": misses}
