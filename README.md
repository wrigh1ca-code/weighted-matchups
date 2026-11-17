# Weighted Matchups

College football matchup pages built on advanced stats (EPA, success rate, explosiveness, havoc…),
where each team's profile is **weighted toward the games that look most like the upcoming one**:

- **Similar opponents**: unit-on-unit (a defense is judged mostly on games vs offenses like this week's opponent)
- **Recency**: exponential decay by weeks ago
- **Setting**: home/away/neutral and forecast weather
- **Roster**: games played with the lineup expected this week (starting QB first)

Weighted profiles are blended back toward the season average by confidence
(effective sample size), then turned into a score projection and win probability.

## Layout

- `pipeline/`: data download and processing (Python)
- `.github/workflows/`: scheduled and manual jobs

## Data

Data from [CollegeFootballData.com](https://collegefootballdata.com). The API key lives only in the
`CFBD_API_KEY` repository secret. Raw API responses are cached under `data/raw/`, which is git-ignored and
never published; the site publishes only derived numbers.
