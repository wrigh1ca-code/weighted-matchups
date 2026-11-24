# Weighted Matchups

College football matchup pages built on advanced stats (EPA, success rate, explosiveness, havoc…),
where each team's profile is **weighted toward the games that look most like the upcoming one**:

- **Similar opponents**: unit-on-unit (a defense is judged mostly on games vs offenses like this week's opponent)
- **Recency**: exponential decay by weeks ago
- **Setting**: home/away/neutral and forecast weather
- **Roster**: games played with the lineup expected this week (starting QB first)

Weighted profiles are blended back toward the season average by confidence
(effective sample size), then turned into a score projection and win probability.

## Strength of schedule

Each game's EPA/play is adjusted for the opponent's strength (`model.opp_ratings`): a weighted ridge
regression over every game (FBS and FCS) solves all offenses, defenses and conferences at once
(EPA = league + offense + defense + conference terms + home edge). Conference terms pool every
nonconference game, so Group of 5 schedules are valued on dozens of games. Last season's baseline is
the same schedule-adjusted rating. Backtest (2026 test games): margin MAE 13.16 → 12.92 at the site's
settings, 12.93 → 12.66 at the tuned settings (market 11.95).

## Results blend

Each team also has a results rating (`model.results_ratings`), built on **game control** (`RES_METRIC`):
for every game, `pipeline/game_flow.py` averages each team's in-game win probability over regulation (WP
model: score, time left, possession, field position; no pregame information). Control is put on a points
scale and solved for all teams and conferences at once with a home edge (games without play-by-play, e.g.
Division II/III, fall back to final margin). Average lead, final margin and a three-way average are also
available; backtest them with `python -m engine.backtest results`. Game control was chosen for what it
measures (who controlled the game); final margin was about 0.05 pts more accurate, within noise. Projections blend it with the play-by-play model (`model.Blend`): final margin = 0.70 x
play-by-play margin + b x results gap + h x home, with b and h fit on 2024-25 (`backtest.fit_blend`); the
0.70 gives the two about equal say and was as accurate as the free fit. Total points stay play-by-play.
Backtest (2026 test games, site settings): MAE 12.92 -> 12.74, winners 65% -> 68%.

## Power rankings

`engine/rankings.py` projects every FBS team against every other (neutral site, good weather) with the
site's model and ranks by average projected margin; `site/rankings.html` shows it.

## Build a matchup

`site/build.html` lets visitors pick any two FBS teams, a venue (either team's stadium or a neutral site) and a
weather scenario (sunny and warm, cold and dry, warm with rain, cold with rain or snow, dome, or not modeled).
The daily export writes `site/data/teams/<team>.json` and `site/data/league.json`; `site/custom.js` rebuilds the
matchup-specific inputs (opponent similarity, site match, weather match) in the browser, mirroring
`engine/model.py`, and the regular matchup page renders it (`game.html?home=…&away=…[&neutral=1][&wx=…]`).

## Layout

- `pipeline/`: data download and processing (Python)
- `.github/workflows/`: scheduled and manual jobs

## Data

Data from [CollegeFootballData.com](https://collegefootballdata.com). The API key lives only in the
`CFBD_API_KEY` repository secret. Raw API responses are cached under `data/raw/`, which is git-ignored and
never published; the site publishes only derived numbers.
