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
    skill_miss: float = 1.0     # (legacy) RB/WR game-weight penalty; superseded by the lineup adjustment
    lineup: float = 1.0         # strength of the key-player lineup adjustment (0 = off)
    k_rb: float = 100.0         # shrinkage (touches) for a key RB's edge over the rest of the pool
    k_wr: float = 80.0          # shrinkage (targets) for a key WR's edge

    def equal(self) -> "Params":
        """Equal weight: every game counts the same and no lineup adjustment."""
        return Params(0, 0, 0, 0, self.half_life, self.sim_bw, self.k, self.qb_miss, self.adj, self.skill_miss,
                      0.0, self.k_rb, self.k_wr)


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


# Opponent strength used to adjust each game's EPA. "ridge" solves every offense and defense
# together (EPA/play = league + offense + defense + home edge, weighted by plays, shrunk toward
# average by OPP_RIDGE plays), so an opponent's level already accounts for its own schedule.
# "raw" is the original one-pass method: the opponent's unadjusted season EPA/play.
OPP_METHOD = "ridge"
OPP_RIDGE = 400.0        # shrinkage of each team's offense/defense toward its conference (plays)
OPP_RIDGE_CONF = 50.0    # shrinkage of each conference's offense/defense toward the league (plays)


@lru_cache(maxsize=512)
def opp_ratings(season: int, date_iso: str, lam: float | None = None, lam_conf: float | None = None) -> dict:
    """{'offense': Series, 'defense': Series}: schedule-adjusted EPA/play level of every unit
    (league average + conference strength + its own strength) from all games (FBS and FCS) before
    `date`. Conference terms pool every nonconference game a conference plays, so a conference's
    level rests on dozens of games rather than each team's three or four."""
    lam = OPP_RIDGE if lam is None else lam
    lam_conf = OPP_RIDGE_CONF if lam_conf is None else lam_conf
    u, _ = load()
    d = pd.Timestamp(date_iso)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    c = u[(u.season == season) & (u.date < d) & (u.side == "offense") & u.epa_play.notna() & (u.plays > 0)]
    if c.empty:
        return {"offense": pd.Series(dtype=float), "defense": pd.Series(dtype=float), "lg": 0.0, "hfa": 0.0}
    fbs = fbs_teams(season)
    teams = sorted(set(c.team) | set(c.opponent))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    w = c.plays.to_numpy(float)
    y = c.epa_play.to_numpy(float)
    lg = float(np.average(y[c.team.isin(fbs).to_numpy()], weights=w[c.team.isin(fbs).to_numpy()]))
    conf = team_conferences(season)
    confs = sorted({conf.get(t, "FCS" if t not in fbs else "FBS Independents") for t in teams})
    cx = {cf: i for i, cf in enumerate(confs)}
    tc = [cx[conf.get(t, "FCS" if t not in fbs else "FBS Independents")] for t in teams]
    m = len(confs)
    # columns: team offense (n), team defense (n), conference offense (m), conference defense (m), home edge
    k = 2 * n + 2 * m + 1
    X = np.zeros((len(c), k))
    rows = np.arange(len(c))
    ti = np.array([ix[t] for t in c.team]); oi = np.array([ix[o] for o in c.opponent])
    X[rows, ti] = 1.0
    X[rows, n + oi] = 1.0
    X[rows, 2 * n + np.array(tc)[ti]] = 1.0
    X[rows, 2 * n + m + np.array(tc)[oi]] = 1.0
    X[:, k - 1] = np.where(c.site.to_numpy() == "home", 0.5, np.where(c.site.to_numpy() == "away", -0.5, 0.0))
    XtW = X.T * w
    A = XtW @ X
    reg = np.concatenate([np.full(2 * n, lam), np.full(2 * m, lam_conf), [1e-6]])   # home edge not shrunk
    beta = np.linalg.solve(A + np.diag(reg), XtW @ (y - lg))
    off = beta[:n] + beta[2 * n:2 * n + m][tc]
    dfn = beta[n:2 * n] + beta[2 * n + m:2 * n + 2 * m][tc]
    return {"offense": pd.Series(lg + off, index=teams), "defense": pd.Series(lg + dfn, index=teams),
            "conf_offense": pd.Series(beta[2 * n:2 * n + m], index=confs),
            "conf_defense": pd.Series(beta[2 * n + m:2 * n + 2 * m], index=confs),
            "lg": lg, "hfa": float(beta[k - 1])}


# Results rating: what actually happened on the scoreboard, adjusted for schedule the same way.
RES_CAP = 28.0          # margins beyond this count as this (no credit for running up the score)
RES_RIDGE = 1.0         # shrinkage of each team toward its conference (in games)
RES_RIDGE_CONF = 0.5    # shrinkage of each conference toward the league (in games)


@lru_cache(maxsize=512)
def results_ratings(season: int, date_iso: str, cap: float | None = None, lam: float | None = None,
                    lam_conf: float | None = None) -> dict:
    """{'rating': Series team -> points better than an average FBS team on a neutral field, 'hfa': float}
    from final scores of every game (FBS and FCS) before `date`: capped margin = home rating -
    away rating + home edge, solved for all teams and conferences at once (ridge)."""
    cap = RES_CAP if cap is None else cap
    lam = RES_RIDGE if lam is None else lam
    lam_conf = RES_RIDGE_CONF if lam_conf is None else lam_conf
    _, g = load()
    d = pd.Timestamp(date_iso)
    if d.tzinfo is None:
        d = d.tz_localize("UTC")
    c = g[(g.season == season) & (g.date < d) & (g.completed == True) & g.homePoints.notna() & g.awayPoints.notna()]  # noqa: E712
    if c.empty:
        return {"rating": pd.Series(dtype=float), "hfa": 0.0, "games": pd.Series(dtype=float)}
    fbs = fbs_teams(season)
    teams = sorted(set(c.homeTeam) | set(c.awayTeam))
    ix = {t: i for i, t in enumerate(teams)}
    conf = team_conferences(season)
    cname = lambda t: conf.get(t, "FCS" if t not in fbs else "FBS Independents")
    confs = sorted({cname(t) for t in teams})
    cx = {cf: i for i, cf in enumerate(confs)}
    tc = np.array([cx[cname(t)] for t in teams])
    n, m = len(teams), len(confs)
    hi = np.array([ix[t] for t in c.homeTeam]); ai = np.array([ix[t] for t in c.awayTeam])
    X = np.zeros((len(c), n + m + 1))
    r = np.arange(len(c))
    X[r, hi] += 1.0; X[r, ai] -= 1.0
    np.add.at(X, (r, n + tc[hi]), 1.0); np.add.at(X, (r, n + tc[ai]), -1.0)
    X[:, n + m] = np.where(c.neutralSite.fillna(False).astype(bool).to_numpy(), 0.0, 1.0)
    y = np.clip((c.homePoints - c.awayPoints).to_numpy(float), -cap, cap)
    reg = np.concatenate([np.full(n, lam), np.full(m, lam_conf), [1e-6]])
    beta = np.linalg.solve(X.T @ X + np.diag(reg), X.T @ y)
    rating = pd.Series(beta[:n] + beta[n:n + m][tc], index=teams)
    rating -= rating[[t for t in teams if t in fbs]].mean()          # 0 = average FBS team
    games = pd.Series(np.bincount(np.concatenate([hi, ai]), minlength=n), index=teams)
    return {"rating": rating, "hfa": float(beta[n + m]), "games": games}


@lru_cache(maxsize=8)
def team_conferences(season: int) -> dict:
    """Team -> conference for the season (FBS conferences; FCS teams are left out)."""
    _, g = load()
    s = g[g.season == season]
    out = {}
    for t, cf in list(zip(s.awayTeam, s.awayConference)) + list(zip(s.homeTeam, s.homeConference)):
        if isinstance(cf, str) and cf:
            out[t] = cf
    fbs = fbs_teams(season)
    return {t: cf for t, cf in out.items() if t in fbs}


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
    if OPP_METHOD == "ridge":
        r = opp_ratings(season, date.isoformat())
        # shift to this module's league average so (opp_level - lg) is the opponent's strength
        lvl = r[other] - r["lg"] + lg
    else:
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
    """Last season's numbers (blended in while this season's sample is small). The display metrics
    are last season's plain averages; `epa_base`, used for projections, is last season's
    schedule-adjusted EPA/play level (opp_ratings) on this season's league scale."""
    prev = season_final(season - 1)
    if not prev.empty and (team, side) in prev.index:
        b = prev.loc[(team, side), METRICS].copy()
    else:
        b = prof[prof.is_fbs & (prof.side == side)][METRICS].mean() if not prof.empty else pd.Series(np.nan, index=METRICS)
    b["epa_base"] = b["epa_play"]
    if OPP_METHOD == "ridge" and not prof.empty:
        r = opp_ratings(season - 1, f"{season}-03-01T00:00:00+00:00")
        if team in r[side].index:
            b["epa_base"] = float(r[side][team] - r["lg"] + prof.attrs.get("lg_epa", r["lg"]))
    return b


def unit_profile(rows: pd.DataFrame, p: Params, baseline: pd.Series, raw: pd.DataFrame | None = None) -> dict:
    """rows may be lineup-adjusted; `raw` (unadjusted) supplies the plain season average."""
    raw = rows if raw is None else raw
    if rows.empty:
        return {"final": baseline.to_dict(), "weighted": baseline.to_dict(), "season": baseline.to_dict(),
                "n_eff": 0.0, "lam": 0.0, "weights": np.array([]), "epa_proj": baseline.get("epa_base", baseline["epa_play"])}
    w = game_weights(rows, p)
    out_w, out_s = {}, {}
    for m in METRICS:
        n = rows[COUNT_COL[m]].to_numpy(dtype=float)
        x = rows[m].to_numpy(dtype=float)
        ok = ~np.isnan(x) & (n > 0)
        out_w[m] = float(np.sum(w[ok] * n[ok] * x[ok]) / np.sum(w[ok] * n[ok])) if ok.any() and np.sum(w[ok] * n[ok]) > 0 else np.nan
        xr = raw[m].to_numpy(dtype=float)
        okr = ~np.isnan(xr) & (n > 0)
        out_s[m] = float(np.sum(n[okr] * xr[okr]) / np.sum(n[okr])) if okr.any() else np.nan
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
    # Opponent-adjusted EPA/play (for projections): remove how good/bad each opponent unit was
    x_adj = rows["epa_play"].to_numpy(dtype=float) - p.adj * (rows["opp_level"].to_numpy(dtype=float) - rows["lg_epa"].to_numpy(dtype=float))
    n_pl = rows["plays"].to_numpy(dtype=float)
    adj_w = float(np.sum(w * n_pl * x_adj) / np.sum(w * n_pl))
    lam = n_eff / (n_eff + p.k)
    final = {m: (lam * out_w[m] + (1 - lam) * baseline[m]) if not (np.isnan(out_w[m]) or pd.isna(baseline[m]))
             else (out_w[m] if not np.isnan(out_w[m]) else baseline[m]) for m in METRICS}
    b_epa = baseline.get("epa_base", baseline["epa_play"])
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
        km = key_player_models(team, m.season, m.date)
        for side in ("offense", "defense"):
            off = side == "offense"
            rows = unit_ingredients(team, side, m.season, m.date, opp, m.site(team), m.weather,
                                    qb if off else None, key if off else None)
            out[(team, side)] = {"rows": rows, "baseline": baseline_for(team, side, m.season, prof),
                                 "expected_qb": qb if off else None, "key": key if off else None,
                                 "km": km if off else {}}
    return out


def matchup_features(m: Matchup, units: dict, p: Params) -> dict:
    """Expected EPA/play for each offense vs the opposing defense, plus plays and confidence."""
    prof = profiles_asof(m.season, m.date.isoformat())
    lg = prof.attrs.get("lg_epa", 0.0) if not prof.empty else 0.0
    res = {"lg_epa": lg}
    def prof_for(v):
        km = dict(v.get("km") or {})
        if p.lineup and km:
            km = {pos: {**mm, "d_epa": mm["d_epa_raw"] * mm["touches"] / (mm["touches"] + (p.k_rb if pos == "rb" else p.k_wr)),
                        "d_sr": mm["d_sr_raw"] * mm["touches"] / (mm["touches"] + (p.k_rb if pos == "rb" else p.k_wr))}
                  for pos, mm in km.items()}
            status = {pos: mm["expected"] for pos, mm in km.items()}
            status.update(v.get("status") or {})
            adj = lineup_adjust(v["rows"], km, status, p.lineup)
            return unit_profile(adj, p, v["baseline"], raw=v["rows"])
        return unit_profile(v["rows"], p, v["baseline"])
    profs = {k: prof_for(v) for k, v in units.items()}
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


@dataclass
class Blend:
    """Final margin = a * play-by-play margin + b * (results rating gap) + h * home (0/1).
    Total points stay the play-by-play total; only the split changes. Fit on past seasons."""
    a: float = 1.0
    b: float = 0.0
    h: float = 0.0
    sd0: float = 15.4
    sd_k: float = 0.4

    def apply(self, ph: float, pa: float, loc: int, rdiff: float, n_eff: float) -> tuple[float, float, float, float]:
        margin = self.a * (ph - pa) + self.b * rdiff + self.h * loc
        total = ph + pa
        sd = self.sd0 * math.sqrt(1 + self.sd_k / max(n_eff, 0.5))
        wp = 0.5 * (1 + math.erf(margin / sd / math.sqrt(2)))
        return (total + margin) / 2, (total - margin) / 2, wp, sd


def project(m: Matchup, feats: dict, pm: PointsModel, bl: Blend | None = None) -> dict:
    loc_h = 0 if m.neutral else 1
    ph = pm.points(feats[m.home]["e"], feats[m.home]["plays"], loc_h)
    pa = pm.points(feats[m.away]["e"], feats[m.away]["plays"], -loc_h)
    n = min(feats[m.home]["n_eff"], feats[m.away]["n_eff"])
    out = {"pbp_home_pts": ph, "pbp_away_pts": pa}
    if bl is not None:
        rr = results_ratings(m.season, m.date.isoformat())["rating"]
        r_h, r_a = float(rr.get(m.home, 0.0)), float(rr.get(m.away, 0.0))
        ph, pa, wp, sd = bl.apply(ph, pa, loc_h, r_h - r_a, n)
        out.update({"results_home": r_h, "results_away": r_a})
    else:
        wp, sd = pm.win_prob(ph - pa, n)
    out.update({"home_pts": ph, "away_pts": pa, "home_wp": wp, "margin_sd": sd, "n_eff": n})
    return out


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


# ---------------------------------------------------------------- key players (lineup adjustment)
#
# For each offense, the top non-QB ball carrier (RB) and top target (WR) by volume. We measure each
# player's own opponent-adjusted EPA and success per touch against the rest of the team's pool
# (other non-QB carries / other targets), shrink that gap toward zero by sample size, and then
# adjust every past game as if this week's expected lineup had played:
#   extra touches for the player in game g = (expected share - actual share in g) x pool touches in g
#   metric_g += extra touches x player edge / (rushes, dropbacks or plays in g)

K_SHRINK = {"rb": 100.0, "wr": 80.0}


@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(DATA.glob("player_games_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=["gameId", "team", "key", "player", "n", "epa_sum", "succ_sum", "yds", "kind"])
    return pd.concat(frames, ignore_index=True)


def _display_name(key: str, team_rows: pd.DataFrame, fallback: str) -> str:
    """Prefer the box-score full name (e.g. 'LJ Martin') over play-text abbreviations ('L.Martin')."""
    from pipeline.unit_stats import name_key
    for col in ("rush_list", "rec_list"):
        for s in team_rows.get(col, pd.Series(dtype=str)).dropna():
            for n in parse_list(s):
                if name_key(n) == key:
                    return n
    return fallback


def key_player_models(team: str, season: int, date: pd.Timestamp, k: dict | None = None) -> dict:
    """{'rb': {...}, 'wr': {...}} for a team's offense using games before `date`."""
    k = k or K_SHRINK
    u, _ = load()
    games = u[(u.team == team) & (u.side == "offense") & (u.season == season) & (u.date < date)].sort_values("date")
    if games.empty:
        return {}
    pl = load_players()
    pl = pl[(pl.team == team) & pl.gameId.isin(games.gameId)]
    if pl.empty:
        return {}
    prof = profiles_asof(season, date.isoformat())
    lg = prof.attrs.get("lg_epa", 0.0)
    if OPP_METHOD == "ridge":
        r = opp_ratings(season, date.isoformat())
        lvl = r["defense"] - r["lg"] + lg
    else:
        lvl = prof[(prof.side == "defense") & (prof.games >= 2)].set_index("team")["epa_play"]
    opp_delta = (games.set_index("gameId").opponent.map(lvl).fillna(lg) - lg)
    qb_keys = set(pl[(pl.kind == "pass")].groupby("key").n.sum().loc[lambda s: s >= 10].index)
    last_game = games.gameId.iloc[-1]
    out = {}
    for pos, kind, vol in (("rb", "rush", "n"), ("wr", "target", "yds")):
        pool = pl[(pl.kind == kind) & ~pl.key.isin(qb_keys if pos == "rb" else set())].copy()
        if pool.empty:
            continue
        key = pool.groupby("key")[vol].sum().idxmax()
        pool["adj_epa"] = pool.epa_sum - pool.n * pool.gameId.map(opp_delta).fillna(0)
        me, rest = pool[pool.key == key], pool[pool.key != key]
        n_me, n_rest = me.n.sum(), rest.n.sum()
        if n_me < 5 or n_rest < 5:
            continue
        e_me, e_rest = me.adj_epa.sum() / n_me, rest.adj_epa.sum() / n_rest
        s_me, s_rest = me.succ_sum.sum() / n_me, rest.succ_sum.sum() / n_rest
        shrink = n_me / (n_me + k[pos])
        per_game = pool.groupby("gameId").n.sum().rename("pool").to_frame().join(me.set_index("gameId").n.rename("mine")).fillna(0)
        per_game["share"] = per_game.mine / per_game.pool.where(per_game.pool > 0)
        played = per_game[per_game.mine > 0]
        if pos == "rb" and len(played) >= 2:          # healthy workload: his two biggest-share games
            top = played.sort_values("share", ascending=False).head(2)
        else:
            top = played
        s_star = float(top.mine.sum() / top.pool.sum()) if len(top) else 0.0
        out[pos] = {
            "key": key, "name": _display_name(key, games, me.player.iloc[0]),
            "touches": int(n_me), "epa_me": float(e_me), "epa_rest": float(e_rest),
            "sr_me": float(s_me), "sr_rest": float(s_rest), "shrink": float(shrink),
            "d_epa": float((e_me - e_rest) * shrink), "d_sr": float((s_me - s_rest) * shrink),
            "d_epa_raw": float(e_me - e_rest), "d_sr_raw": float(s_me - s_rest),
            "s_star": s_star, "expected": bool(per_game.mine.get(last_game, 0) > 0),
            "games_played": int((per_game.mine > 0).sum()), "games": int(len(games)),
            "share": {int(g): float(r.share) if not pd.isna(r.share) else 0.0 for g, r in per_game.iterrows()},
            "pool": {int(g): float(r.pool) for g, r in per_game.iterrows()},
        }
    return out


def lineup_adjust(rows: pd.DataFrame, km: dict, status: dict, strength: float = 1.0) -> pd.DataFrame:
    """Return offense rows with metrics adjusted to this week's expected lineup.
    status: {'rb': bool expected to play, 'wr': bool}."""
    if rows.empty or not km or strength == 0:
        return rows
    r = rows.copy()
    for pos, base_col, den_col in (("rb", "rush", "rushes"), ("wr", "pass", "dropbacks")):
        m = km.get(pos)
        if not m:
            continue
        s_e = m["s_star"] if status.get(pos, m["expected"]) else 0.0
        share = r.gameId.map(lambda g: m["share"].get(int(g), 0.0)).astype(float)
        pool = r.gameId.map(lambda g: m["pool"].get(int(g), 0.0)).astype(float)
        extra = (s_e - share) * pool * strength
        de, ds = extra * m["d_epa"], extra * m["d_sr"]
        den = r[den_col].where(r[den_col] > 0)
        r[f"{base_col}_epa"] = r[f"{base_col}_epa"] + (de / den).fillna(0)
        if pos == "wr":
            r["pass_sr"] = r["pass_sr"] + (ds / den).fillna(0)
        r["epa_play"] = r["epa_play"] + (de / r.plays).fillna(0)
        r["sr"] = r["sr"] + (ds / r.plays).fillna(0)
    return r
