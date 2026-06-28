"""Predict international football matches"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import features as feat
from features import FEATURE_COLUMNS, LABEL_COLUMN

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 50000
DATA = "results.csv"
RAW_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Map the integer label (home-team perspective) back to the competition's strings.
OUTCOME = {2: "home_win", 1: "draw", 0: "away_win"}


def load_data(refresh=False):
    """Load the martj42 results CSV (downloading if missing/refresh) into the
    schema features.build_features() expects: sorted by date, bool `neutral` and `played`."""
    import os

    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    if "city" not in df.columns:
        df["city"] = ""
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def train(pool):
    """Fit the TabPFN client classifier on the engineered feature matrix."""
    from tabpfn_client import TabPFNClassifier  # lazy import: only needed to train

    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    X = pool[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = pool[LABEL_COLUMN].to_numpy()
    clf.fit(X, y)
    return clf


def predict_outcomes(clf, upcoming):
    """Build the competition submission frame for a set of upcoming matches.

    Columns: date, home_team, away_team, predicted, p_home_win, p_draw, p_away_win.
    """
    proba = clf.predict_proba(upcoming[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    col = {int(c): i for i, c in enumerate(clf.classes_)}
    out = upcoming[["date", "home_team", "away_team"]].copy()
    out["predicted"] = [OUTCOME[int(clf.classes_[i])] for i in proba.argmax(1)]
    out["p_home_win"] = proba[:, col[2]]
    out["p_draw"] = proba[:, col[1]]
    out["p_away_win"] = proba[:, col[0]]
    return out.reset_index(drop=True)


def main():
    """Backtest on the previous calendar month, then predict all upcoming matches."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download dataset from source"
    )
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    latest_date = df[df["played"]]["date"].max()
    print(f"Latest game in dataset: {latest_date.date()}")
    print(f"Data freshness: {pd.Timestamp.now() - latest_date}")

    train_df, upcoming_df = feat.build_features(df)
    played = train_df[train_df["date"] >= TRAIN_START]
    future = upcoming_df[upcoming_df["date"] >= TODAY].sort_values("date")

    # Backtest on the previous calendar month.
    month = TODAY.to_period("M") - 1
    test = played[
        (played["date"] >= month.start_time) & (played["date"] < (month + 1).start_time)
    ]
    if len(test):
        clf = train(played[played["date"] < month.start_time].tail(MAX_TRAIN))
        proba = clf.predict_proba(test[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        pred = clf.classes_[proba.argmax(1)]
        print(
            f"\nBacktest {month} ({len(test)} matches): "
            f"accuracy {accuracy_score(test[LABEL_COLUMN], pred):.0%}, "
            f"log-loss {log_loss(test[LABEL_COLUMN], proba, labels=clf.classes_):.3f}"
        )

    if not len(future):
        print("\nNo upcoming matches to predict.")
        return

    # Train on the most recent matches and predict every upcoming match.
    clf = train(played.tail(MAX_TRAIN))
    out = predict_outcomes(clf, future)
    filename = f"predictions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    out.to_csv(filename, index=False)

    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r in out.itertuples():
        print(
            f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
            f"-> {r.predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}"
        )


if __name__ == "__main__":
    main()
