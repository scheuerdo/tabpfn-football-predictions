"""A World-Football-style Elo rating engine.

Ratings are updated chronologically, one match at a time:

  * expected score uses a 400-point logistic curve with a home-field bonus,
  * the update step size K scales with the importance of the competition,
  * a goal-difference multiplier G rewards bigger wins.
"""

from __future__ import annotations

from typing import Dict


def match_importance(tournament: str) -> float:
    """Map a competition name to an importance weight in [0, 1].

    1.0 = World Cup finals, ~0.85 = major continental finals, ~0.6 = qualifiers,
    0.2 = friendlies. Used both for the Elo K-factor and as a model feature.
    """
    t = (tournament or "").lower()
    if t == "fifa world cup":
        return 1.0
    if "world cup" in t and "qualif" in t:
        return 0.6
    if "qualif" in t:
        return 0.55
    major = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "confederations",
        "nations league",
    )
    if any(k in t for k in major):
        return 0.85
    if t == "friendly":
        return 0.2
    return 0.45  # other minor tournaments


def k_factor(tournament: str) -> float:
    """Elo step size: 20 (friendly) ... 60 (World Cup finals)."""
    return 10.0 + 50.0 * match_importance(tournament)


def goal_diff_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference index G."""
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


class EloEngine:
    def __init__(self, base: float = 1500.0, home_advantage: float = 65.0):
        self.base = base
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        """Pre-match win expectancy for the home team (incl. draws as 0.5)."""
        ha = 0.0 if neutral else self.home_advantage
        diff = (self.rating(home) + ha) - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> None:
        """Apply one finished match to the ratings."""
        exp_home = self.expected_home(home, away, neutral)
        if home_score > away_score:
            score_home = 1.0
        elif home_score < away_score:
            score_home = 0.0
        else:
            score_home = 0.5
        k = k_factor(tournament)
        g = goal_diff_multiplier(home_score - away_score)
        delta = k * g * (score_home - exp_home)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
