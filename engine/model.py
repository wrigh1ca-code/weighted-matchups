"""Weighted matchup engine.

For a target game, each of the four units (home offense/defense, away offense/defense)
gets a profile built from that unit's past games this season. Every past game g gets

    w_g = sim_g^a_opp * rec_g^a_rec * set_g^a_set * ros_g^a_ros

where each base factor is in (0, 1]:
  sim  similarity of the opponent unit faced in g to the opponent unit faced in the target game
       (unit-on-unit: a defense is compared on the offenses it faced, and vice versa)
  rec  0.5 ** (weeks_ago / half_life)
  set  site match x weather match
  ros  offense only: did the expected starting QB start game g

Weighted metric = sum(w * n * x) / sum(w * n), n = the metric's play count.
Confidence: n_eff = (sum w)^2 / sum w^2, lam = n_eff / (n_eff + k).
Final = lam * weighted + (1 - lam) * baseline, baseline = the unit's prior-season average
(league average if none). With every exponent at 0 this is the equal-weight model.

The base ingredients (distance, weeks_ago, site match, weather distance, QB match) are
stored per past game so the website can redo the weighting live with slider exponents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "derived"

METRICS = [
    "epa_play", "sr", "early_epa", "late_sr", "rush_epa", "stuff", "line_yards", "opportunity",
    "explosive", "pass_expl", "rush_expl", "non_expl_epa", "pass_epa", "yds_db", "pass_sr", "havoc",
]
LABELS = {
    "epa_play": "EPA/Play", "sr": "Success Rate", "early_epa": "Early Downs EPA", "late_sr": "Late Downs SR",
    "rush_epa": "Rush EPA", "stuff": "Stuff %", "line_yards": "Line Yards", "opportunity": "Opportunity %",
    "explosive": "Explosive %", "pass_expl": "Pass Expl %", "rush_expl": "Rush Expl %",
    "non_expl_epa": "Non-Expl EPA", "pass_epa": "Pass EPA", "yds_db": "Yds/DB", "pass_sr": "Pass SR",
    "havoc": "Havoc %",
}
COUNT_COL = {m: "plays" for m in METRICS}
COUNT_COL.update({m: "rushes" for m in ("rush_epa", "stuff", "line_yards", "opportunity", "rush_expl")})
COUNT_COL.update({m: "dropbacks" for m in ("pass_epa", "pass_sr", "yds_db", "pass_expl")})
# For an offense, higher is better except these; for a defense (stats allowed) it flips.
OFFENSE_LOWER_BETTER = {"stuff", "havoc"}

# Similarity features: (column, weight). epa_adj is opponent-adjusted EPA/play (strength).
SIM_FEATURES = [("epa_adj", 2.0), ("sr", 1.0), ("pass_rate", 1.0), ("explosive", 1.0),
                ("rush_epa", 0.75), ("pass_epa", 0.75)]

SITE_MATCH = {  # (target site, past-game site) -> base factor
    ("home", "home"): 1.0, ("away", "away"): 1.0, ("neutral", "neutral"): 1.0,
    ("home", "away"): 0.5, ("away", "home"): 0.5,
    ("neutral", "away"): 0.8, ("away", "neutral"): 0.8,
    ("neutral", "home"): 0.6, ("home", "neutral"): 0.6,
}


@dataclass
class Params:
    a_opp: float = 1.0
    a_rec: float = 1.0
    a_set: float = 1.0
    a_ros: float = 1.0
    half_life: float = 6.0      # weeks
    sim_bw: float = 1.0         # similarity bandwidth h
    k: float = 3.0              # confidence shrinkage constant
    qb_miss: float = 0.35       # roster base factor when a different QB started
    adj: float = 1.0            # opponent adjustment strength for EPA used in projections
    skill_miss: float = 0.7     # roster base factor per key RB/WR whose status differs

    def equal(self) -> "Params":
        return Params(0, 0, 0, 0, self.half_life, self.sim_bw, self.k, self.qb_miss, self.adj, self.skill_miss)


# ---------------------------------------------------------------- data

@lru_cache(maxsize=1)
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    units, games = [], []
    for f in sorted(DATA.glob("unit_games_*.csv")):
        units.append(pd.read_csv(f))
    for f in sorted(DATA.glob("games_*.csv")):
        games.append(pd.read_csv(f))
    u = pd.concat(units, ignore_index=True)
    g = pd.concat(games, ignore_index=True)
    u["date"] = pd.to_datetime(u["startDate"], utc=True)
    g["date"] = pd.to_datetime(g["startDate"], utc=True)
    u["pass_rate"] = u["dropbacks"] / u["plays"].where(u["plays"] > 0)
    u = u[u["plays"] >= 10].reset_index(drop=True)
    return u, g


def fbs_teams(season: int) -> set[str]:
    _, g = load()
    s = g[g.season == season]
    return set(s.loc[s.homeClassification == "fbs", "homeTeam"]) | set(s.loc[s.awayClassification == "fbs", "awayTeam"])


def wavg(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = {}
    for c in cols:
        n = df[COUNT_COL.get(c, "plays")] if c in COUNT_COL else df["plays"]
        ok = df[c].notna() & (n > 0)
        out[c] = float((df.loc[ok, c] * n[ok]).sum() / n[ok].sum()) if ok.any() else np.nan
    return pd.Series(out)


PROFILE_COLS = METRICS + ["pass_rate"]


def group_wavg(df: pd.DataFrame, cols: list[str], keys=("team", "side")) -> pd.DataFrame:
    """Vectorized play-weighted means per group (same result as wavg per group)."""
    keys = list(keys)
    out = {}
    for c in cols:
        n = df[COUNT_COL.get(c, "plays")]
        ok = df[c].notna() & (n > 0)
        num = (df[c] * n).where(ok, 0.0)
        den = n.where(ok, 0.0)
        tmp = pd.DataFrame({"num": num, "den": den})
        for k in keys:
            tmp[k] = df[k]
        agg = tmp.groupby(keys)[["num", "den"]].sum()
        out[c] = agg["num"] / agg["den"].where(agg["den"] > 0)
    return pd.DataFrame(out)


@lru_cache(maxsize=4096)
def profiles_asof(season: int, date_iso: str) -> pd.DataFrame:
    """Plain season-to-date averages per (team, side) before `date`, plus opponent-adjusted EPA."""
    u, _ = load()
    d = pd.Timestamp(date_iso)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    cur = u[(u.season == season) & (u.date < d)]
    if cur.empty:
        return pd.DataFrame()
    prof = group_wavg(cur, PROFILE_COLS)
    prof["games"] = cur.groupby(["team", "side"]).size()
    prof["plays_pg"] = cur.groupby(["team", "side"])["plays"].mean()
    prof = prof.reset_index()
    fbs = fbs_teams(season)
    lg = prof[(prof.side == "offense") & prof.team.isin(fbs)]["epa_play"].mean()
    # one-pass opponent adjustment of EPA/play
    allowed = prof[prof.side == "defense"].set_index("team")["epa_play"]
    gained = prof[prof.side == "offense"].set_index("team")["epa_play"]
    c = cur[["team", "opponent", "side", "epa_play", "plays"]].copy()
    c["opp_level"] = np.where(c.side == "offense", c.opponent.map(allowed), c.opponent.map(gained))
    c["adj"] = c["epa_play"] - (c["opp_level"].fillna(lg) - lg)
    c["adj_n"] = c["adj"] * c["plays"]
    sums = c.groupby(["team", "side"])[["adj_n", "plays"]].sum()
    adj = (sums["adj_n"] / sums["plays"]).rename("epa_adj").reset_index()
    prof = prof.merge(adj, on=["team", "side"], how="left")
    prof["is_fbs"] = prof.team.isin(fbs)
    prof.attrs["lg_epa"] = lg
    return prof


@lru_cache(maxsize=64)
def season_final(season: int) -> pd.DataFrame:
    """Full-season plain averages (used as next season's baseline)."""
    u, _ = load()
    cur = u[u.season == season]
    if cur.empty:
        return pd.DataFrame()
    return group_wavg(cur, METRICS)


@lru_cache(maxsize=1)
def feature_scales() -> dict[str, float]:
    u, _ = load()
    p = profiles_asof(int(u.season.max()) - 1, "2100-01-01T00:00:00+00:00")
    p = p[p.is_fbs]
    return {c: float(p[c].std()) for c, _ in SIM_FEATURES}


# ---------------------------------------------------------------- per-unit ingredients

def _wx_dist2(t, gw) -> float | None:
    """Squared weather distance between target conditions t and past game gw (dicts)."""
    def norm(x):
        if x is None:
            return None
        if bool(x.get("indoors")):
            return (70.0, 0.0, 0.0)
        if x.get("temp") is None or pd.isna(x.get("temp")):
            return None
        return (float(x["temp"]), float(x.get("wind") or 0), float(x.get("precip") or 0))
    a, b = norm(t), norm(gw)
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) / 25) ** 2 + ((a[1] - b[1]) / 12) ** 2 + 0.5 * float((a[2] > 0) != (b[2] > 0))


def parse_list(s) -> dict[str, list[int]]:
    """'Name:a:b|Name2:a' -> {'Name': [a, b], ...}"""
    out = {}
    if not isinstance(s, str) or not s:
        return out
    for part in s.split("|"):
        bits = part.split(":")
        name = bits[0].strip()
        if not name or name.lower() == "team":
            continue
        try:
            out[name] = [int(x) for x in bits[1:]]
        except ValueError:
            continue
    return out


def key_players(team: str, season: int, date: pd.Timestamp) -> dict:
    """Season-to-date top RB (carries, non-QB) and top WR (receiving yards) and whether each
    appeared in the team's most recent game (the default expectation for this week)."""
    u, _ = load()
    r = u[(u.team == team) & (u.side == "offense") & (u.season == season) & (u.date < date)].sort_values("date")
    if r.empty or "rush_list" not in r:
        return {"rb": None, "wr": None, "rb_expected": True, "wr_expected": True}
    qbs = set(r.qb.dropna())
    car, yds = {}, {}
    for rl, cl in zip(r.rush_list, r.rec_list):
        for n, v in parse_list(rl).items():
            if n not in qbs:
                car[n] = car.get(n, 0) + v[0]
        for n, v in parse_list(cl).items():
            yds[n] = yds.get(n, 0) + (v[1] if len(v) > 1 else 0)
    rb = max(car, key=car.get) if car else None
    wr = max(yds, key=yds.get) if yds else None
    last = r.iloc[-1]
    on_field = set(parse_list(last.rush_list)) | set(parse_list(last.rec_list))
    return {"rb": rb, "wr": wr, "rb_expected": rb in on_field if rb else True,
            "wr_expected": wr in on_field if wr else True,
            "rb_carries": car.get(rb, 0), "wr_yards": yds.get(wr, 0)}


def unit_ingredients(team: str, side: str, season: int, date: pd.Timestamp, opponent: str,
                     site: str, weather: dict | None, expected_qb: str | None,
                     key: dict | None = None) -> pd.DataFrame:
    """Past games this season for (team, side) with the raw weighting ingredients."""
    u, _ = load()
    rows = u[(u.team == team) & (u.side == side) & (u.season == season) & (u.date < date)].copy()
    if rows.empty:
        return rows
    prof = profiles_asof(season, date.isoformat())
    scales = feature_scales()
    other = "offense" if side == "defense" else "defense"
    # The opponent unit this team faces in the target game, and those it faced before
    pidx = prof.set_index(["team", "side"])

    def vec(t):
        if (t, other) not in pidx.index:
            return None
        r = pidx.loc[(t, other)]
        return np.array([r[c] / scales[c] for c, _ in SIM_FEATURES], dtype=float)

    wts = np.array([w for _, w in SIM_FEATURES])
    target_vec = vec(opponent)
    dists = []
    for opp in rows.opponent:
        v = vec(opp)
        if target_vec is None or v is None or np.isnan(v).any() or np.isnan(target_vec).any():
            dists.append(3.0)
        else:
            dists.append(float(np.sqrt(np.sum(wts * (v - target_vec) ** 2) / wts.sum())))
    rows["dist"] = dists
    rows["weeks_ago"] = (date - rows.date).dt.total_seconds() / (7 * 86400)
    rows["site_base"] = [SITE_MATCH.get((site, s), 0.6) for s in rows.site]
    rows["wx_d2"] = [_wx_dist2(weather, {"indoors": i, "temp": t, "wind": w, "precip": p})
                     for i, t, w, p in zip(rows.indoors, rows.temp, rows.wind, rows.precip)]
    if side == "offense" and expected_qb:
        rows["qb_match"] = (rows.qb == expected_qb).astype(float)
    else:
        rows["qb_match"] = 1.0
    # Key skill players: did each play in game g, and does that match this week's expectation?
    rows["rb_played"], rows["wr_played"] = True, True
    rows["rb_match"], rows["wr_match"] = 1.0, 1.0
    if side == "offense" and key and "rush_list" in rows:
        present = [set(parse_list(a)) | set(parse_list(b)) for a, b in zip(rows.rush_list, rows.rec_list)]
        for pos in ("rb", "wr"):
            name = key.get(pos)
            if name:
                played = np.array([name in ps for ps in present])
                rows[f"{pos}_played"] = played
                rows[f"{pos}_match"] = (played == bool(key.get(f"{pos}_expected", True))).astype(float)
    rows["opp_fbs"] = rows.opponent.isin(fbs_teams(season))
    # Opponent strength as of now: the opposite unit's season-to-date EPA/play (league avg if thin)
    lg = prof.attrs.get("lg_epa", 0.0)
    lvl = prof[(prof.side == other) & (prof.games >= 2)].set_index("team")["epa_play"]
    rows["opp_level"] = rows.opponent.map(lvl).fillna(lg)
    rows["lg_epa"] = lg
    return rows


def base_factors(rows: pd.DataFrame, p: Params) -> pd.DataFrame:
    f = pd.DataFrame(index=rows.index)
    f["sim"] = np.exp(-(rows["dist"] ** 2) / (2 * p.sim_bw ** 2)).clip(1e-4, 1)
    f["rec"] = np.power(0.5, rows["weeks_ago"] / p.half_life).clip(1e-4, 1)
    wx = rows["wx_d2"].apply(lambda d: 1.0 if d is None or pd.isna(d) else math.exp(-d / 2))
    f["set"] = (rows["site_base"] * wx).clip(1e-4, 1)
    f["ros"] = (np.where(rows["qb_match"] >= 1, 1.0, p.qb_miss)
                * np.where(rows["rb_match"] >= 1, 1.0, p.skill_miss)
                * np.where(rows["wr_match"] >= 1, 1.0, p.skill_miss))
    return f


def game_weights(rows: pd.DataFrame, p: Params) -> np.ndarray:
    f = base_factors(rows, p)
    return (f.sim ** p.a_opp * f.rec ** p.a_rec * f.set ** p.a_set * f.ros ** p.a_ros).to_numpy()


def baseline_for(team: str, side: str, season: int, prof: pd.DataFrame) -> pd.Series:
    prev = season_final(season - 1)
    if not prev.empty and (team, side) in prev.index:
        return prev.loc[(team, side), METRICS]
    lg = prof[prof.is_fbs & (prof.side == side)][METRICS].mean() if not prof.empty else pd.Series(np.nan, index=METRICS)
    return lg


def unit_profile(rows: pd.DataFrame, p: Params, baseline: pd.Series) -> dict:
    if rows.empty:
        return {"final": baseline.to_dict(), "weighted": baseline.to_dict(), "season": baseline.to_dict(),
                "n_eff": 0.0, "lam": 0.0, "weights": np.array([]), "epa_proj": baseline["epa_play"]}
    w = game_weights(rows, p)
    out_w, out_s = {}, {}
    for m in METRICS:
        n = rows[COUNT_COL[m]].to_numpy(dtype=float)
        x = rows[m].to_numpy(dtype=float)
        ok = ~np.isnan(x) & (n > 0)
        out_w[m] = float(np.sum(w[ok] * n[ok] * x[ok]) / np.sum(w[ok] * n[ok])) if ok.any() and np.sum(w[ok] * n[ok]) > 0 else np.nan
        out_s[m] = float(np.sum(n[ok] * x[ok]) / np.sum(n[ok])) if ok.any() else np.nan
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
    # Opponent-adjusted EPA/play (for projections): remove how good/bad each opponent unit was
    x_adj = rows["epa_play"].to_numpy(dtype=float) - p.adj * (rows["opp_level"].to_numpy(dtype=float) - rows["lg_epa"].to_numpy(dtype=float))
    n_pl = rows["plays"].to_numpy(dtype=float)
    adj_w = float(np.sum(w * n_pl * x_adj) / np.sum(w * n_pl))
    lam = n_eff / (n_eff + p.k)
    final = {m: (lam * out_w[m] + (1 - lam) * baseline[m]) if not (np.isnan(out_w[m]) or pd.isna(baseline[m]))
             else (out_w[m] if not np.isnan(out_w[m]) else baseline[m]) for m in METRICS}
    b_epa = baseline["epa_play"]
    epa_proj = lam * adj_w + (1 - lam) * b_epa if not pd.isna(b_epa) else adj_w
    return {"final": final, "weighted": out_w, "season": out_s, "n_eff": n_eff, "lam": lam, "weights": w,
            "epa_proj": epa_proj}


# ---------------------------------------------------------------- matchup

@dataclass
class Matchup:
    season: int
    date: pd.Timestamp
    home: str
    away: str
    neutral: bool = False
    weather: dict | None = None
    expected_qb: dict = field(default_factory=dict)   # team -> QB name

    def site(self, team: str) -> str:
        if self.neutral:
            return "neutral"
        return "home" if team == self.home else "away"


def last_qb(team: str, season: int, date: pd.Timestamp) -> str | None:
    u, _ = load()
    r = u[(u.team == team) & (u.side == "offense") & (u.season == season) & (u.date < date)]
    r = r.dropna(subset=["qb"])
    return None if r.empty else str(r.sort_values("date").qb.iloc[-1])


def matchup_units(m: Matchup) -> dict:
    """Ingredients for all four units of a matchup. Keys: (team, side)."""
    out = {}
    prof = profiles_asof(m.season, m.date.isoformat())
    for team, opp in ((m.home, m.away), (m.away, m.home)):
        qb = m.expected_qb.get(team) or last_qb(team, m.season, m.date)
        key = key_players(team, m.season, m.date)
        for side in ("offense", "defense"):
            off = side == "offense"
            rows = unit_ingredients(team, side, m.season, m.date, opp, m.site(team), m.weather,
                                    qb if off else None, key if off else None)
            out[(team, side)] = {"rows": rows, "baseline": baseline_for(team, side, m.season, prof),
                                 "expected_qb": qb if off else None, "key": key if off else None}
    return out


def matchup_features(m: Matchup, units: dict, p: Params) -> dict:
    """Expected EPA/play for each offense vs the opposing defense, plus plays and confidence."""
    prof = profiles_asof(m.season, m.date.isoformat())
    lg = prof.attrs.get("lg_epa", 0.0) if not prof.empty else 0.0
    res = {"lg_epa": lg}
    profs = {k: unit_profile(v["rows"], p, v["baseline"]) for k, v in units.items()}
    for team, opp in ((m.home, m.away), (m.away, m.home)):
        off, dfn = profs[(team, "offense")], profs[(opp, "defense")]
        e = off["epa_proj"] + dfn["epa_proj"] - lg
        po = units[(team, "offense")]["rows"]["plays"].mean() if not units[(team, "offense")]["rows"].empty else 65
        pd_ = units[(opp, "defense")]["rows"]["plays"].mean() if not units[(opp, "defense")]["rows"].empty else 65
        res[team] = {"e": e, "plays": (po + pd_) / 2, "n_eff": min(off["n_eff"], dfn["n_eff"])}
    res["profiles"] = profs
    return res


# ---------------------------------------------------------------- points model

@dataclass
class PointsModel:
    a: float = 27.0      # intercept (points at league-average matchup, ~65 plays)
    b: float = 60.0      # points per unit of expected EPA/play (at ~65 plays)
    c: float = 0.30      # points per extra projected play
    hfa: float = 1.25    # half of home-field edge, added to home and removed from away
    sd0: float = 13.5    # margin SD at full confidence
    sd_k: float = 2.0    # SD inflation: sd = sd0 * sqrt(1 + sd_k / n_eff)

    def points(self, e: float, plays: float, loc: int) -> float:
        return self.a + self.b * e + self.c * (plays - 65) + self.hfa * loc

    def win_prob(self, margin: float, n_eff: float) -> tuple[float, float]:
        sd = self.sd0 * math.sqrt(1 + self.sd_k / max(n_eff, 0.5))
        z = margin / sd
        return 0.5 * (1 + math.erf(z / math.sqrt(2))), sd


def project(m: Matchup, feats: dict, pm: PointsModel) -> dict:
    loc_h = 0 if m.neutral else 1
    ph = pm.points(feats[m.home]["e"], feats[m.home]["plays"], loc_h)
    pa = pm.points(feats[m.away]["e"], feats[m.away]["plays"], -loc_h)
    n = min(feats[m.home]["n_eff"], feats[m.away]["n_eff"])
    wp, sd = pm.win_prob(ph - pa, n)
    return {"home_pts": ph, "away_pts": pa, "home_wp": wp, "margin_sd": sd, "n_eff": n}


# ---------------------------------------------------------------- percentiles

def percentile_refs(season: int, date: pd.Timestamp) -> dict:
    """Sorted FBS reference values per (side, metric) for converting to 0-100 'better than' ranks."""
    prof = profiles_asof(season, date.isoformat())
    if prof.empty or prof[prof.is_fbs].groupby("side").size().min() < 50:
        prof = profiles_asof(season - 1, "2100-01-01T00:00:00+00:00")
    refs = {}
    for side in ("offense", "defense"):
        p = prof[prof.is_fbs & (prof.side == side)]
        for m in METRICS:
            refs[(side, m)] = np.sort(p[m].dropna().to_numpy())
    return refs


def higher_is_better(side: str, metric: str) -> bool:
    good_high = metric not in OFFENSE_LOWER_BETTER
    return good_high if side == "offense" else not good_high


def to_pct(value: float, side: str, metric: str, refs: dict) -> float:
    arr = refs[(side, metric)]
    if value is None or np.isnan(value) or len(arr) == 0:
        return np.nan
    frac = np.searchsorted(arr, value, side="left") / len(arr)
    return float(100 * (frac if higher_is_better(side, metric) else 1 - frac))
