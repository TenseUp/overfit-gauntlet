"""Fixture: exercises every look-ahead rule LA01-LA08. Not real code."""
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def build(df, X, y):
    df["target"] = df["price"].shift(-1)            # LA01
    df["fwd_ret"] = df["price"].pct_change(-5)      # LA08
    df["ma"] = df["price"].rolling(20, center=True).mean()  # LA02
    df["filled"] = df["raw"].bfill()                # LA03
    df["filled2"] = df["raw"].fillna(method="bfill")  # LA03
    df["z"] = (df["price"] - df["price"].mean()) / df["price"].std()  # LA07 warn

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)                    # LA06 (before split)

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y)  # LA04
    cv = KFold(n_splits=5, shuffle=True)            # LA05 error
    cv2 = KFold(n_splits=5)                         # LA05 warn
    return X_tr, cv, cv2
