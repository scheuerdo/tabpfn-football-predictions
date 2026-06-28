# TabPFN Football Predictions

This repository is a template to participate in Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/worldcup). It has a basic script that outputs predictions with a standard prediction template. Use this template to generate predictions. The `predict.py` script should only be a source of inspiration, feel free to fork the repo and add your own ideas.

The script predicts international football match outcomes using [TabPFN](https://github.com/PriorLabs/TabPFN) using the [client repository](https://github.com/PriorLabs/tabpfn-client).

The model is trained on engineered features: ELO ratings, recent form, head-to-head record, rest days, tournament importance, online attack/defense goal ratings, and draw propensity. Data comes from [martj42/international_results](https://github.com/martj42/international_results).

## Setup

```bash
git clone https://github.com/eliott-kalfon/tabpfn-football-predictions.git
cd tabpfn-football-predictions
pip install -r requirements.txt
```

## Run

```bash
python predict.py
```

This will:

1. Download the full international results dataset (~49 000 matches) on first run
2. Build features with a single chronological pass (no leakage)
3. Run a quick backtest on the previous calendar month and print accuracy + log-loss
4. Train on recent matches (since 2014, capped at `MAX_TRAIN` = 20 000 rows) and predict all upcoming fixtures
5. Save predictions to `predictions_YYYYMMDD.csv` and print them to the console

To refresh the dataset from source before predicting:

```bash
python predict.py --refresh
```

## Output

```
Latest game in dataset: 2026-06-14
Data freshness: 0 days 18:32:11

Backtest 2026-05 (87 matches): accuracy 59%, log-loss 0.861

142 fixture predictions -> predictions_20260616.csv

  2026-06-18           Argentina vs Australia             -> home_win   H  72% | D  17% | A  11%
  2026-06-18              France vs Morocco              -> home_win   H  61% | D  23% | A  16%
  ...
```

## Features

Feature engineering lives in `elo.py` (a World-Football-style Elo engine) and
`features.py` (a single, leakage-safe chronological pass). The 33 numeric columns
fed to TabPFN are `features.FEATURE_COLUMNS` = the 25 base columns plus 8 added
goal-rating / draw columns.

Base columns:

| Feature | Description |
|---|---|
| `elo1`, `elo2` | Current ELO rating of the home / away team |
| `elo_diff` | ELO gap (home − away) |
| `elo_expected` | Home win-expectancy from ELO (home advantage applied on non-neutral games) |
| `neutral` | 1 if played at a neutral venue |
| `home_adv` | 1 if the home team has home advantage (i.e. not neutral) |
| `importance` | Competition importance weight in [0, 1] (0.2 = friendly … 1.0 = World Cup) |
| `is_world_cup` | 1 if the match is a FIFA World Cup finals game |
| `form1_ppg`, `form2_ppg` | Points per game over the last 10 matches (home / away) |
| `form1_gf`, `form2_gf` | Goals scored per game over the last 10 matches |
| `form1_ga`, `form2_ga` | Goals conceded per game over the last 10 matches |
| `form1_gd`, `form2_gd` | Average goal difference over the last 10 matches |
| `form1_n`, `form2_n` | Number of matches available in the form window |
| `form_ppg_diff`, `form_gd_diff` | Home−away differences in form PPG and goal difference |
| `h2h_n` | Number of recent head-to-head meetings (last 8) |
| `h2h_team1_winrate` | Home team win rate in head-to-head |
| `h2h_team1_gd` | Home team average goal difference in head-to-head |
| `rest1`, `rest2` | Days since each team's last match (capped at 45) |

Added goal-rating & draw columns:

| Feature | Description |
|---|---|
| `att1`, `att2` | Online attack (goal-scoring) rating of home / away team (0 = league average) |
| `def1`, `def2` | Online defensive leakiness rating (higher = concedes more) |
| `xg1`, `xg2` | Pre-match expected goals for home / away from the attack/defense model |
| `draw_rate1`, `draw_rate2` | Shrunk historical draw rate of the home / away team |
