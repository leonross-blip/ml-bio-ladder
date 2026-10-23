"""K-means from scratch on the Palmer penguins (MLB L02).

Built step by step: load + standardise, pick starting centres,
assign each point to its nearest centre. Update, WSS and the
full loop come next.
"""

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def load_penguins():
    """Return Z (n, 4) standardised features and the matching species labels."""
    df = pd.read_csv(URL)
    clean = df.dropna(subset=FEATURES)          # 344 -> 342: drop birds with no measurements
    species = clean["species"].to_numpy()       # labels, only for checking at the end
    X = clean[FEATURES].to_numpy()              # (342, 4) raw numbers
    Z = (X - X.mean(axis=0)) / X.std(axis=0)    # each column: mean 0, sd 1
    return Z, species


def init_centres(Z, k, rng):
    """Pick k different data points as the starting centres -> (k, 4)."""
    rows = rng.choice(len(Z), size=k, replace=False)
    return Z[rows]


def assign(Z, centres):
    """Return labels (n,): the index of each point's nearest centre."""
    D = []
    for p in Z:                                    # one penguin at a time
        row = []
        for c in centres:                          # each centre in turn
            row.append(((p - c) ** 2).sum())       # squared distance p -> c
        D.append(row)
    D = np.array(D)                                # (n, k) distance table
    return D.argmin(axis=1)                        # position of smallest in each row


if __name__ == "__main__":
    Z, species = load_penguins()
    print("Z:", Z.shape)                                   # (342, 4)
    print("means:", Z.mean(axis=0).round(3))               # ~0
    print("sds:  ", Z.std(axis=0).round(3))                # ~1

    rng = np.random.default_rng(0)                         # seed 0: same picks every run
    centres = init_centres(Z, 3, rng)
    print("centres:", centres.shape)                       # (3, 4)

    labels = assign(Z, centres)
    print("labels:", labels.shape, labels[:10])            # (342,), values 0-2
    print("cluster sizes:", np.bincount(labels))           # points per cluster