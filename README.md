# Weighted Matchups

College football matchup pages built on advanced stats (EPA, success rate, explosiveness, havoc…),
where each team's profile is **weighted toward the games that look most like the upcoming one**:

- **Similar opponents**: unit-on-unit (a defense is judged mostly on games vs offenses like this week's opponent)
- **Recency**: exponential decay by weeks ago
- **Setting**: home/away/neutral and forecast weather
- **Roster**: games played with the lineup expected this week (starting QB first)

Weighted profiles are blended back toward the season average by confidence
(effective sample size), then turned into a score projection and win probability.

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
