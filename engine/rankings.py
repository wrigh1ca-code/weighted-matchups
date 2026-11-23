"""Power rankings: every FBS team against every other on a neutral field in good weather.

Each pairing is projected with the same weighted model and site settings as the matchup pages
(engine/model.py, export_site.load_tuned). A team's rating is its average projected margin
against all other FBS teams: "points better than the average FBS team on a neutral field".

A full round robin is ~9,400 games, too slow through model.project one at a time (~0.5 s each).
On a neutral field in fixed weather every weighting factor of a unit is the same whatever the
opponent except opponent similarity, so each unit's fixed part is built once and the similarity
to every possible opponent is done as one array operation. check() confirms the result matches
model.project exactly.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from engine.model import (SIM_FEATURES, Matchup, Params, PointsModel, baseline_for, fbs_teams, feature_scales,
                          key_player_models, key_players, last_qb, lineup_adjust, load, matchup_features,
                          matchup_units, profiles_asof, project, unit_ingredients)

GOOD_WEATHER = {"indoors": False, "temp": 75, "wind": 6, "precip": 0}   # same as the builder's "Sunny and warm"


def _unit_arrays(rows: pd.DataFrame, p: Params, vecs: dict, other: str) -> dict | None:
    """Opponent-independent part of a unit's game weights, plus what the projection needs."""
    if rows.empty:
        return None
    rec = np.clip(np.power(0.5, rows["weeks_ago"].to_numpy(float) / p.half_life), 1e-4, 1)
    wx = rows["wx_d2"].apply(lambda d: 1.0 if d is None or pd.isna(d) else math.exp(-d / 2)).to_numpy(float)
    st = np.clip(rows["site_base"].to_numpy(float) * wx, 1e-4, 1)
    ros = (np.where(rows["qb_match"] >= 1, 1.0, p.qb_miss)
           * np.where(rows["rb_match"] >= 1, 1.0, p.skill_miss)
           * np.where(rows["wr_match"] >= 1, 1.0, p.skill_miss))
    fixed = rec ** p.a_rec * st ** p.a_set * ros ** p.a_ros
    nf = len(SIM_FEATURES)
    V = np.array([vecs.get((o, other), np.full(nf, np.nan)) for o in rows.opponent], dtype=float).reshape(-1, nf)
    x_adj = rows["epa_play"].to_numpy(float) - p.adj * (rows["opp_level"].to_numpy(float) - rows["lg_epa"].to_numpy(float))
    return {"fixed": fixed, "V": V, "x_adj": x_adj, "plays": rows["plays"].to_numpy(float),
            "mean_plays": float(rows["plays"].mean())}


def _epa_vs_all(ua: dict | None, baseline: pd.Series, targets: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """epa_proj and n_eff of one unit against each target opponent vector (rows of `targets`)."""
    T = len(targets)
    b = baseline["epa_play"]
    if ua is None:
        return np.full(T, b), np.zeros(T)
    wts = np.array([w for _, w in SIM_FEATURES])
    diff = ua["V"][None, :, :] - targets[:, None, :]                       # T x games x features
    d = np.sqrt(np.sum(wts * diff ** 2, axis=2) / wts.sum())
    bad = np.isnan(ua["V"]).any(axis=1)[None, :] | np.isnan(targets).any(axis=1)[:, None]
    d = np.where(bad, 3.0, d)
    sim = np.clip(np.exp(-(d ** 2) / (2 * p.sim_bw ** 2)), 1e-4, 1)
    w = sim ** p.a_opp * ua["fixed"][None, :]                              # T x games
    adj_w = (w * ua["plays"] * ua["x_adj"]).sum(1) / (w * ua["plays"]).sum(1)
    n_eff = w.sum(1) ** 2 / (w ** 2).sum(1)
    lam = n_eff / (n_eff + p.k)
    epa = lam * adj_w + (1 - lam) * b if not pd.isna(b) else adj_w
    return epa, n_eff


def power_ratings(season: int, asof: pd.Timestamp, p: Params, pm: PointsModel) -> tuple[pd.DataFrame, np.ndarray, list]:
    teams = sorted(fbs_teams(season))
    prof = profiles_asof(season, asof.isoformat())
    lg = prof.attrs.get("lg_epa", 0.0) if not prof.empty else 0.0
    scales = feature_scales()
    vecs = {(r.team, r.side): np.array([getattr(r, c) / scales[c] for c, _ in SIM_FEATURES], dtype=float)
            for r in prof.itertuples(index=False)}
    nf = len(SIM_FEATURES)
    U = {}
    for t in teams:
        qb, key = last_qb(t, season, asof), key_players(t, season, asof)
        km = key_player_models(t, season, asof)
        for side in ("offense", "defense"):
            off = side == "offense"
            rows = unit_ingredients(t, side, season, asof, t, "neutral", GOOD_WEATHER,
                                    qb if off else None, key if off else None)
            if off and p.lineup and km and not rows.empty:   # same lineup handling as matchup_features
                kk = {pos: {**mm, "d_epa": mm["d_epa_raw"] * mm["touches"] / (mm["touches"] + (p.k_rb if pos == "rb" else p.k_wr)),
                            "d_sr": mm["d_sr_raw"] * mm["touches"] / (mm["touches"] + (p.k_rb if pos == "rb" else p.k_wr))}
                      for pos, mm in km.items()}
                rows = lineup_adjust(rows, kk, {pos: mm["expected"] for pos, mm in km.items()}, p.lineup)
            U[(t, side)] = (_unit_arrays(rows, p, vecs, "defense" if off else "offense"),
                            baseline_for(t, side, season, prof))
    # Targets: each team's defense (what an offense faces) and offense (what a defense faces)
    tgt = {s: np.array([vecs.get((t, s), np.full(nf, np.nan)) for t in teams]) for s in ("offense", "defense")}
    n = len(teams)
    E_off = np.zeros((n, n)); N_off = np.zeros((n, n))    # [i, j]: i's offense vs j's defense
    E_def = np.zeros((n, n)); N_def = np.zeros((n, n))    # [j, i]: j's defense vs i's offense
    P_off = np.zeros(n); P_def = np.zeros(n)
    for i, t in enumerate(teams):
        ua, base = U[(t, "offense")]
        E_off[i], N_off[i] = _epa_vs_all(ua, base, tgt["defense"], p)
        P_off[i] = ua["mean_plays"] if ua else 65.0
        ud, based = U[(t, "defense")]
        E_def[i], N_def[i] = _epa_vs_all(ud, based, tgt["offense"], p)
        P_def[i] = ud["mean_plays"] if ud else 65.0
    # points[i, j] = i's projected points against j (neutral site)
    e = E_off + E_def.T - lg
    plays = (P_off[:, None] + P_def[None, :]) / 2
    pts = pm.a + pm.b * e + pm.c * (plays - 65)
    neff = np.minimum(N_off, N_def.T)
    margin = pts - pts.T
    n_game = np.minimum(neff, neff.T)
    sd = pm.sd0 * np.sqrt(1 + pm.sd_k / np.maximum(n_game, 0.5))
    wp = 0.5 * (1 + np.vectorize(math.erf)(margin / sd / math.sqrt(2)))
    off_mask = ~np.eye(n, dtype=bool)
    k = off_mask.sum(1)
    df = pd.DataFrame({
        "team": teams,
        "rating": (margin * off_mask).sum(1) / k,
        "pts_for": (pts * off_mask).sum(1) / k,
        "pts_against": (pts.T * off_mask).sum(1) / k,
        "win_pct": (wp * off_mask).sum(1) / k,
        "favored": ((margin > 0) & off_mask).sum(1),
        "opponents": k,
    })
    df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df, pts, teams


def check(season: int = 2026, pairs=(("BYU", "Utah"), ("Ohio State", "Indiana"), ("Georgia", "Texas"),
                                     ("Air Force", "UTEP"), ("Hawai'i", "Kent State"))) -> float:
    """Largest difference (points) between the fast round robin and model.project."""
    from engine.export_site import load_tuned
    p, pm, _ = load_tuned()
    asof = pd.Timestamp.now("UTC").floor("h")
    _, pts, teams = power_ratings(season, asof, p, pm)
    ix = {t: i for i, t in enumerate(teams)}
    worst = 0.0
    for h, a in pairs:
        m = Matchup(season, asof, h, a, True, GOOD_WEATHER)
        r = project(m, matchup_features(m, matchup_units(m), p), pm)
        fast_h, fast_a = pts[ix[h], ix[a]], pts[ix[a], ix[h]]
        print(f"{a} vs {h}: model {r['away_pts']:.3f}-{r['home_pts']:.3f}  fast {fast_a:.3f}-{fast_h:.3f}")
        worst = max(worst, abs(fast_h - r["home_pts"]), abs(fast_a - r["away_pts"]))
    print(f"max difference: {worst:.2e} pts")
    return worst


if __name__ == "__main__":
    sys.exit(0 if check() < 1e-6 else 1)
