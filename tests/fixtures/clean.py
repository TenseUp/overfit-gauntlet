"""Fixture: causal, leak-free backtest code. The linter must find nothing."""
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler


def build(df, X, y):
    df["target"] = df["price"].shift(1)             # past -> present, fine
    df["ret"] = df["price"].pct_change(1)           # positive periods, fine
    df["ma"] = df["price"].rolling(20).mean()       # trailing window, fine
    df["filled"] = df["raw"].ffill()                # forward fill, fine

    # split first, THEN fit the scaler on the training split only
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False)  # explicit
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)             # after split, fine
    cv = TimeSeriesSplit(n_splits=5)                # correct CV for time series
    return X_tr_s, cv
