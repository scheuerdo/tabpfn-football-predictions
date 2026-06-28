"""Feature engineering.

A single chronological pass over every match builds:

  * `train` : one row per *played* match, with pre-match features + an integer
              label for the result from team1's (= the home_team field) point of
              view -> 0 team1 loss, 1 draw, 2 team1 win.
  * `upcoming` : one row per *unplayed* match (the upcoming games), with the
              same features computed from the final post-history state.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd

from elo import EloEngine, match_importance

FORM_WINDOW = 10  # matches used for rolling form
H2H_WINDOW = 8  # most recent meetings used for head-to-head
REST_CAP_DAYS = 45  # clip "days since last match"

# Online attack/defense (goal-rating) settings.
AD_LR = 0.06  # learning rate for the attack/defense update
AD_CLIP = 3.0  # clip attack/defense ratings to a sane range
BASE_HOME_GOALS = 1.5  # prior home goals before enough history accrues
BASE_AWAY_GOALS = 1.1  # prior away goals
# Draw-propensity shrinkage toward the global base draw rate (~0.23 internationally).
DRAW_BASE = 0.23
DRAW_ALPHA = 12

FEATURE_COLUMNS: List[str] = [
    "elo1",
    "elo2",
    "elo_diff",
    "elo_expected",
    "neutral",
    "home_adv",
    "importance",
    "is_world_cup",
    "form1_ppg",
    "form1_gf",
    "form1_ga",
    "form1_gd",
    "form1_n",
    "form2_ppg",
    "form2_gf",
    "form2_ga",
    "form2_gd",
    "form2_n",
    "form_ppg_diff",
    "form_gd_diff",
    "h2h_n",
    "h2h_team1_winrate",
    "h2h_team1_gd",
    "rest1",
    "rest2",
    "att1",  # online attack rating for home team
    "att2",  # online attack rating for away team
    "def1",  # online defense rating for home team
    "def2",  # online defense rating for away team
    "xg1",  # pre-match expected goals for home team
    "xg2",  # pre-match expected goals for away team
    "draw_rate1",  # shrunk historical draw rate for home team
    "draw_rate2",  # shrunk historical draw rate for away team
]

LABEL_COLUMN = "label"
META_COLUMNS = ["date", "home_team", "away_team", "tournament", "city", "neutral"]


def _form_stats(hist: Deque[dict]) -> Tuple[float, float, float, float, int]:
    """(points-per-game, goals-for, goals-against, goal-diff, n) over the window."""
    n = len(hist)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    pts = np.mean([h["pts"] for h in hist])
    gf = np.mean([h["gf"] for h in hist])
    ga = np.mean([h["ga"] for h in hist])
    return float(pts), float(gf), float(ga), float(gf - ga), n


def _h2h_stats(
    meetings: Deque[Tuple[str, int, int, str]], team1: str
) -> Tuple[int, float, float]:
    """(#meetings, team1 win-rate, team1 avg goal-diff) over recent meetings."""
    n = len(meetings)
    if n == 0:
        return 0, 0.5, 0.0
    wins = gd = 0.0
    for home, hs, as_, _away in meetings:
        # express the historical meeting from team1's perspective
        t1_gf, t1_ga = (hs, as_) if home == team1 else (as_, hs)
        gd += t1_gf - t1_ga
        if t1_gf > t1_ga:
            wins += 1
        elif t1_gf == t1_ga:
            wins += 0.5
    return n, wins / n, gd / n


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, upcoming_df). `df` must be chronologically sorted."""
    elo = EloEngine()
    form: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h: Dict[frozenset, Deque[Tuple[str, int, int, str]]] = defaultdict(
        lambda: deque(maxlen=H2H_WINDOW)
    )
    last_played: Dict[str, pd.Timestamp] = {}
    # Online attack (goals scored) and defense (goals conceded) ratings per team,
    # both 0 = league-average. xg = baseline + own attack + opponent's leakiness.
    att: Dict[str, float] = defaultdict(float)
    leak: Dict[str, float] = defaultdict(float)  # higher = concedes more
    draws: Dict[str, int] = defaultdict(int)
    games: Dict[str, int] = defaultdict(int)
    n_g = sum_h = sum_a = 0  # running totals for the global goal baselines

    train_rows: List[dict] = []
    upcoming_rows: List[dict] = []

    def draw_rate(team: str) -> float:
        """Shrunk historical draw rate (Bayesian toward the global base rate)."""
        return (draws[team] + DRAW_ALPHA * DRAW_BASE) / (games[team] + DRAW_ALPHA)

    for r in df.itertuples(index=False):
        home, away, neutral = r.home_team, r.away_team, bool(r.neutral)
        imp = match_importance(r.tournament)

        f1 = _form_stats(form[home])
        f2 = _form_stats(form[away])
        h2h_n, h2h_wr, h2h_gd = _h2h_stats(h2h[frozenset((home, away))], home)
        rest1 = (
            min((r.date - last_played[home]).days, REST_CAP_DAYS)
            if home in last_played
            else REST_CAP_DAYS
        )
        rest2 = (
            min((r.date - last_played[away]).days, REST_CAP_DAYS)
            if away in last_played
            else REST_CAP_DAYS
        )

        # Pre-match expected goals from the attack/defense ratings (leakage-safe).
        base_h = sum_h / n_g if n_g > 50 else BASE_HOME_GOALS
        base_a = sum_a / n_g if n_g > 50 else BASE_AWAY_GOALS
        xg1 = max(base_h + att[home] + leak[away], 0.05)
        xg2 = max(base_a + att[away] + leak[home], 0.05)

        row = {
            "date": r.date,
            "home_team": home,
            "away_team": away,
            "tournament": r.tournament,
            "city": getattr(r, "city", ""),
            "neutral": int(neutral),
            "elo1": elo.rating(home),
            "elo2": elo.rating(away),
            "elo_diff": elo.rating(home) - elo.rating(away),
            "elo_expected": elo.expected_home(home, away, neutral),
            "home_adv": 0 if neutral else 1,
            "importance": imp,
            "is_world_cup": int(r.tournament == "FIFA World Cup"),
            "form1_ppg": f1[0],
            "form1_gf": f1[1],
            "form1_ga": f1[2],
            "form1_gd": f1[3],
            "form1_n": f1[4],
            "form2_ppg": f2[0],
            "form2_gf": f2[1],
            "form2_ga": f2[2],
            "form2_gd": f2[3],
            "form2_n": f2[4],
            "form_ppg_diff": f1[0] - f2[0],
            "form_gd_diff": f1[3] - f2[3],
            "h2h_n": h2h_n,
            "h2h_team1_winrate": h2h_wr,
            "h2h_team1_gd": h2h_gd,
            "rest1": rest1,
            "rest2": rest2,
            "att1": att[home],
            "att2": att[away],
            "def1": leak[home],
            "def2": leak[away],
            "xg1": xg1,
            "xg2": xg2,
            "draw_rate1": draw_rate(home),
            "draw_rate2": draw_rate(away),
        }

        if bool(r.played):
            hs, as_ = int(r.home_score), int(r.away_score)
            row[LABEL_COLUMN] = 2 if hs > as_ else (0 if hs < as_ else 1)
            row["goals_home"] = hs
            row["goals_away"] = as_
            train_rows.append(row)
            # update running state AFTER recording the row
            elo.update(home, away, hs, as_, r.tournament, neutral)
            pts_h = 3 if hs > as_ else (1 if hs == as_ else 0)
            pts_a = 3 if as_ > hs else (1 if hs == as_ else 0)
            form[home].append({"gf": hs, "ga": as_, "pts": pts_h})
            form[away].append({"gf": as_, "ga": hs, "pts": pts_a})
            h2h[frozenset((home, away))].append((home, hs, as_, away))
            last_played[home] = r.date
            last_played[away] = r.date
            # Attack/defense online update: nudge ratings toward the goal residuals.
            res_h, res_a = hs - xg1, as_ - xg2
            att[home] = float(np.clip(att[home] + AD_LR * res_h, -AD_CLIP, AD_CLIP))
            leak[away] = float(np.clip(leak[away] + AD_LR * res_h, -AD_CLIP, AD_CLIP))
            att[away] = float(np.clip(att[away] + AD_LR * res_a, -AD_CLIP, AD_CLIP))
            leak[home] = float(np.clip(leak[home] + AD_LR * res_a, -AD_CLIP, AD_CLIP))
            n_g += 1
            sum_h += hs
            sum_a += as_
            # Draw propensity counts.
            games[home] += 1
            games[away] += 1
            if hs == as_:
                draws[home] += 1
                draws[away] += 1
        else:
            upcoming_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    upcoming_df = pd.DataFrame(upcoming_rows)
    return train_df, upcoming_df
