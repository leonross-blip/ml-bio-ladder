"""K-means from scratch on the Palmer penguins (MLB L02).

Steps: load + standardise -> pick starting centres -> repeat
(assign each point to its nearest centre, move each centre to the
mean of its points) until no point changes cluster.

Done: K-means, WSS, best-of-N random restarts.
Still to add: choosing K (WSS curve + silhouette).
"""

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


# ---------------------------------------------------------------- data

def load_penguins():
    """Return Z (n, 4) standardised features and the matching species labels."""
    df = pd.read_csv(URL)
    clean = df.dropna(subset=FEATURES)          # 344 -> 342: drop birds with no measurements
    species = clean["species"].to_numpy()       # labels, only for checking at the end
    X = clean[FEATURES].to_numpy()              # (342, 4) raw numbers
    Z = (X - X.mean(axis=0)) / X.std(axis=0)    # each column: mean 0, sd 1
    return Z, species


# ---------------------------------------------------------------- K-means pieces

def init_centres(Z, k, rng):
    """Pick k different data points as the starting centres -> (k, 4).

    Starting on real points guarantees every centre owns at least one
    point at the start, so no cluster begins empty.
    """
    rows = rng.choice(len(Z), size=k, replace=False)   # k different row numbers
    return Z[rows]


def assign(Z, centres):
    """ASSIGN step: return labels (n,), the index of each point's nearest centre."""
    D = []
    for p in Z:                                    # one penguin at a time
        row = []
        for c in centres:                          # each centre in turn
            row.append(((p - c) ** 2).sum())       # squared distance p -> c
        D.append(row)
    D = np.array(D)                                # (n, k) distance table
    return D.argmin(axis=1)                        # column of the smallest in each row


def update(Z, labels, k):
    """UPDATE step: move each centre to the mean of its points -> (k, 4).

    The mean is the point that minimises the sum of squared distances
    to a group, so this step can only lower WSS.
    """
    centres = []
    for j in range(k):                             # clusters 0 .. k-1
        members = Z[labels == j]                   # only cluster j's penguins
        centres.append(members.mean(axis=0))       # average down each column
    return np.array(centres)


def wss(Z, labels, centres):
    """Within-cluster sum of squares: each point's squared distance to its own centre, summed.

    centres[labels] repeats each point's own centre -> (n, 4), lined up with Z.
    """
    return ((Z - centres[labels]) ** 2).sum()


# ---------------------------------------------------------------- the algorithm

def kmeans(Z, k, rng, max_iterations=100):
    """Run K-means once from random starting centres.

    Returns (centres, labels, n_rounds). Stops when a round of ASSIGN
    moves no point (labels unchanged), or after max_iterations.
    """
    centres = init_centres(Z, k, rng)
    previous_labels = np.array([])                 # empty: can never match round 1
    for i in range(max_iterations):
        labels = assign(Z, centres)
        if np.array_equal(previous_labels, labels):
            break                                  # nobody moved: converged
        centres = update(Z, labels, k)
        previous_labels = labels
    return centres, labels, i + 1                  # i counts from 0


def best_of_restarts(Z, k, n_starts=10, species=None):
    """Run K-means from n_starts different seeds; keep the run with the lowest WSS.

    K-means only finds the nearest dip in WSS from where it starts, so we
    try several starts. The choice uses WSS alone (no labels), so it is fair.
    If species is given, each run's agreement is printed as a check.
    """
    best = None
    for seed in range(n_starts):
        rng = np.random.default_rng(seed)          # one generator per seed: each run reproducible
        centres, labels, n_rounds = kmeans(Z, k, rng)
        score = wss(Z, labels, centres)

        line = f"seed {seed}: WSS {score:.1f}, {n_rounds} rounds"
        if species is not None:
            line += f", agreement {agreement(species, labels):.1%}"
        print(line)

        if best is None or score < best["wss"]:
            best = {"seed": seed, "wss": score, "labels": labels, "centres": centres}
    return best


# ---------------------------------------------------------------- checking

def agreement(species, labels):
    """Share of points whose cluster's majority species is their own species."""
    table = pd.crosstab(species, labels)           # species x cluster counts
    return table.max(axis=0).sum() / len(labels)   # majority per cluster, summed


# ---------------------------------------------------------------- run

if __name__ == "__main__":
    Z, species = load_penguins()
    print("Z:", Z.shape)                                       # (342, 4)

    # 1. One run, seed 0.
    rng = np.random.default_rng(0)
    centres, labels, n_rounds = kmeans(Z, 3, rng)
    print(f"\nOne run: {n_rounds} rounds, WSS {wss(Z, labels, centres):.1f}, "
          f"agreement {agreement(species, labels):.1%}")
    print(pd.crosstab(species, labels))

    # 2. Ten restarts, keep the lowest WSS.
    print("\nRestarts:")
    best = best_of_restarts(Z, 3, n_starts=10, species=species)
    print(f"\nBest: seed {best['seed']}, WSS {best['wss']:.1f}, "
          f"agreement {agreement(species, best['labels']):.1%}")
    print(pd.crosstab(species, best["labels"]))