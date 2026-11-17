"""Walk-forward backtest of the weighted model vs the equal-weight model.

Step 1 (slow, cached): for every completed FBS-vs-FBS game in the test seasons, collect the
pre-game weighting ingredients for all four units, using only data available before kickoff.
Step 2 (fast): for any parameter set, compute expected EPA, fit the points model on the
training seasons, and score the held-out season.

Usage:
  python -m engine.backtest collect      # builds data/backtest/ingredients.pkl
  python -m engine.backtest tune         # coordinate search, writes data/backtest/results.json
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.model import (Matchup, Params, PointsModel, baseline_for, fbs_teams, key_players, last_qb, load,
                          profiles_asof, unit_ingredients)

OUT = Path(__file__).resolve().parents[1] / "data" / "backtest"
SEASONS = (2024, 2025, 2026)
TRAIN, TEST = (2024, 2025), (2026,)
MIN_GAMES = 3
SITE_DEFAULT_A = {"a_opp": 1.0, "a_rec": 1.0, "a_set": 1.0, "a_ros": 1.0}   # the site's starting sliders


def collect() -> list[dict]:
    u, g = load()
    recs = []
    for season in SEASONS:
        fbs = fbs_teams(season)
        gs = g[(g.season == season) & (g.completed == True) & g.homeTeam.isin(fbs) & g.awayTeam.isin(fbs)]  # noqa: E712
        gs = gs.sort_values("date")
        for i, row in enumerate(gs.itertuples(index=False)):
            date = row.date
            n_h = ((u.team == row.homeTeam) & (u.side == "offense") & (u.season == season) & (u.date < date)).sum()
            n_a = ((u.team == row.awayTeam) & (u.side == "offense") & (u.season == season) & (u.date < date)).sum()
            if min(n_h, n_a) < MIN_GAMES:
                continue
            m = Matchup(season, date, row.homeTeam, row.awayTeam, bool(row.neutralSite),
                        {"indoors": row.indoors, "temp": row.temp, "wind": row.wind, "precip": row.precip})
            prof = profiles_asof(season, date.isoformat())
            rec = {"gameId": row.gameId, "season": season, "date": str(date), "home": row.homeTeam,
                   "away": row.awayTeam, "neutral": bool(row.neutralSite), "home_pts": row.homePoints,
                   "away_pts": row.awayPoints, "spread_home": row.spread_home, "lg": prof.attrs["lg_epa"],
                   "units": {}}
            for team, opp in ((row.homeTeam, row.awayTeam), (row.awayTeam, row.homeTeam)):
                qb = last_qb(team, season, date)
                key = key_players(team, season, date)
                for side in ("offense", "defense"):
                    off = side == "offense"
                    r = unit_ingredients(team, side, season, date, opp, m.site(team), m.weather,
                                         qb if off else None, key if off else None)
                    base = baseline_for(team, side, season, prof)
                    rec["units"][f"{team}|{side}"] = {
                        "dist": r.dist.to_numpy(), "weeks_ago": r.weeks_ago.to_numpy(),
                        "site_base": r.site_base.to_numpy(),
                        "wx_d2": np.array([np.nan if v is None else v for v in r.wx_d2], dtype=float),
                        "qb_match": r.qb_match.to_numpy(dtype=float), "epa": r.epa_play.to_numpy(dtype=float),
                        "rb_match": r.rb_match.to_numpy(dtype=float), "wr_match": r.wr_match.to_numpy(dtype=float),
                        "plays": r.plays.to_numpy(dtype=float), "base_epa": float(base["epa_play"]),
                        "opp_level": r.opp_level.to_numpy(dtype=float), "lg": float(prof.attrs["lg_epa"]),
                    }
            recs.append(rec)
            if i % 100 == 0:
                print(f"  {season}: {i}/{len(gs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ingredients.pkl", "wb") as f:
        pickle.dump(recs, f)
    print(f"collected {len(recs)} games")
    return recs


def unit_epa(d: dict, p: Params) -> tuple[float, float, float]:
    sim = np.clip(np.exp(-(d["dist"] ** 2) / (2 * p.sim_bw ** 2)), 1e-4, 1)
    rec = np.clip(np.power(0.5, d["weeks_ago"] / p.half_life), 1e-4, 1)
    wx = np.where(np.isnan(d["wx_d2"]), 1.0, np.exp(-np.nan_to_num(d["wx_d2"]) / 2))
    st = np.clip(d["site_base"] * wx, 1e-4, 1)
    ros = (np.where(d["qb_match"] >= 1, 1.0, p.qb_miss)
           * np.where(d.get("rb_match", 1) >= 1, 1.0, p.skill_miss)
           * np.where(d.get("wr_match", 1) >= 1, 1.0, p.skill_miss))
    w = sim ** p.a_opp * rec ** p.a_rec * st ** p.a_set * ros ** p.a_ros
    n = d["plays"]
    x = d["epa"] - p.adj * (d["opp_level"] - d["lg"])
    weighted = float(np.sum(w * n * x) / np.sum(w * n))
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    lam = n_eff / (n_eff + p.k)
    base = d["base_epa"] if not math.isnan(d["base_epa"]) else weighted
    return lam * weighted + (1 - lam) * base, float(np.mean(n)), n_eff


def features(recs: list[dict], p: Params) -> pd.DataFrame:
    rows = []
    for r in recs:
        out = {"gameId": r["gameId"], "season": r["season"], "margin": r["home_pts"] - r["away_pts"],
               "home_pts": r["home_pts"], "away_pts": r["away_pts"], "neutral": r["neutral"],
               "spread_home": r["spread_home"]}
        neffs = []
        for tag, team, opp in (("h", r["home"], r["away"]), ("a", r["away"], r["home"])):
            eo, po, no = unit_epa(r["units"][f"{team}|offense"], p)
            ed, pd_, nd = unit_epa(r["units"][f"{opp}|defense"], p)
            out[f"e_{tag}"] = eo + ed - r["lg"]
            out[f"plays_{tag}"] = (po + pd_) / 2
            neffs += [no, nd]
        out["n_eff"] = min(neffs)
        rows.append(out)
    return pd.DataFrame(rows)


def fit_points(df: pd.DataFrame) -> PointsModel:
    """OLS on team-game rows: pts = a + b*e + c*(plays-65) + hfa*loc."""
    loc = np.where(df.neutral, 0, 1)
    X = np.vstack([
        np.column_stack([np.ones(len(df)), df.e_h, df.plays_h - 65, loc]),
        np.column_stack([np.ones(len(df)), df.e_a, df.plays_a - 65, -loc]),
    ])
    y = np.concatenate([df.home_pts, df.away_pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pm = PointsModel(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), hfa=float(coef[3]))
    pred = predict(df, pm)
    resid = df.margin - pred
    # SD model: resid^2 ~ sd0^2 * (1 + sd_k / n_eff); fit by least squares on squared residuals
    A = np.column_stack([np.ones(len(df)), 1 / df.n_eff.clip(lower=0.5)])
    s, *_ = np.linalg.lstsq(A, resid ** 2, rcond=None)
    sd0sq = max(s[0], 1.0)
    pm.sd0 = float(math.sqrt(sd0sq))
    pm.sd_k = float(max(s[1] / sd0sq, 0.0))
    return pm


def predict(df: pd.DataFrame, pm: PointsModel) -> np.ndarray:
    loc = np.where(df.neutral, 0, 1)
    ph = pm.a + pm.b * df.e_h + pm.c * (df.plays_h - 65) + pm.hfa * loc
    pa = pm.a + pm.b * df.e_a + pm.c * (df.plays_a - 65) - pm.hfa * loc
    return (ph - pa).to_numpy()


def score(df: pd.DataFrame, pm: PointsModel) -> dict:
    pred = predict(df, pm)
    sd = pm.sd0 * np.sqrt(1 + pm.sd_k / df.n_eff.clip(lower=0.5))
    from math import erf
    wp = np.array([0.5 * (1 + erf(z / math.sqrt(2))) for z in pred / sd])
    win = (df.margin > 0).astype(float).to_numpy()
    tie = df.margin.to_numpy() == 0
    return {
        "games": int(len(df)),
        "mae": float(np.mean(np.abs(df.margin - pred))),
        "rmse": float(np.sqrt(np.mean((df.margin - pred) ** 2))),
        "winner_acc": float(np.mean((np.sign(pred) == np.sign(df.margin))[~tie])),
        "brier": float(np.mean((wp - win)[~tie] ** 2)),
    }


def evaluate(recs: list[dict], p: Params) -> dict:
    df = features(recs, p)
    tr, te = df[df.season.isin(TRAIN)], df[df.season.isin(TEST)]
    pm = fit_points(tr)
    return {"train": score(tr, pm), "test": score(te, pm), "pm": asdict(pm), "df_test": te}


GRID = {
    "a_opp": [0, 0.5, 1, 1.5, 2, 3],
    "a_rec": [0, 0.5, 1, 1.5, 2],
    "a_set": [0, 0.25, 0.5, 1, 1.5],
    "a_ros": [0, 0.5, 1, 2],
    "half_life": [3, 4, 6, 8, 12],
    "sim_bw": [0.5, 0.75, 1.0, 1.5, 2.0],
    "k": [1, 2, 3, 5, 8],
    "qb_miss": [0.2, 0.35, 0.5, 0.75],
    "adj": [0, 0.5, 0.75, 1.0, 1.25],
    "skill_miss": [0.4, 0.55, 0.7, 0.85, 1.0],
}


def tune(recs: list[dict]) -> dict:
    """Coordinate descent on TRAIN-season RMSE (test season untouched)."""
    def train_rmse(p):
        df = features(recs, p)
        tr = df[df.season.isin(TRAIN)]
        return score(tr, fit_points(tr))["rmse"]

    best = Params()
    best_s = train_rmse(best)
    for _ in range(3):
        improved = False
        for name, vals in GRID.items():
            for v in vals:
                cand = Params(**{**asdict(best), name: v})
                s = train_rmse(cand)
                if s < best_s - 1e-4:
                    best, best_s, improved = cand, s, True
            print(f"  {name} -> {getattr(best, name)} (train RMSE {best_s:.3f})", flush=True)
        if not improved:
            break
    return asdict(best)


def main(argv):
    cmd = argv[0] if argv else "tune"
    if cmd == "collect":
        collect()
        return 0
    with open(OUT / "ingredients.pkl", "rb") as f:
        recs = pickle.load(f)
    tuned = Params(**tune(recs))
    res_w = evaluate(recs, tuned)
    res_e = evaluate(recs, tuned.equal())
    site_default = Params(**{**asdict(tuned), **SITE_DEFAULT_A})
    res_d = evaluate(recs, site_default)
    te = res_w["df_test"]
    mk = te.dropna(subset=["spread_home"])
    market = {"games": int(len(mk)), "mae": float(np.mean(np.abs(mk.margin + mk.spread_home))),
              "rmse": float(np.sqrt(np.mean((mk.margin + mk.spread_home) ** 2)))}
    pm = PointsModel(**res_w["pm"])
    ours_on_mk = score(mk, pm)
    out = {
        "train_seasons": list(TRAIN), "test_seasons": list(TEST),
        "params": asdict(tuned), "points_model": res_w["pm"],
        "weighted": {"train": res_w["train"], "test": res_w["test"]},
        "equal_weight": {"train": res_e["train"], "test": res_e["test"], "points_model": res_e["pm"]},
        "site_default": {"a": SITE_DEFAULT_A, "train": res_d["train"], "test": res_d["test"]},
        "market": {"test": market, "ours_same_games": ours_on_mk},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
