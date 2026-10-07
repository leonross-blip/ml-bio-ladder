"""
rung1/net.py — one-hidden-layer network from scratch in NumPy: forward pass and loss.

Notation: x = 30 standardised inputs, h = 16 hidden units, z = pre-sigmoid, p = sigmoid(z).

For one tumour x (30 numbers):
    z1 = W1ᵀx + b1        16 weighted sums, one per hidden unit        W1 (30×16), b1 (16)
    h  = ReLU(z1)          = max(z1, 0): negatives become 0            h  (16)
    z2 = W2ᵀh + b2         logistic regression on the 16 h's          W2 (16×1), b2 (1)
    p  = sigmoid(z2)       probability benign
For all n rows at once, stack the x's as rows of X (n×30) and the same
lines read  Z1 = X @ W1 + b1,  H = max(Z1, 0),  Z2 = H @ W2 + b2,  P = σ(Z2).
Parameters: 30·16 + 16 + 16·1 + 1 = 513.
Loss: the same binary cross-entropy as the baseline,
    L = -mean( y log p + (1-y) log(1-p) ).
Check: with tiny random weights every z2 ≈ 0, so every p ≈ 0.5, so every row's
loss is -log(0.5) = ln 2 ≈ 0.693 whatever its label. If I see 0.693, the
forward pass and the loss are both right.
"""

import numpy as np
from data import load_data


def init_params(seed=0, n_in=30, n_hidden=16):
    """Create the four parameter arrays, small random weights and zero biases."""
    rng = np.random.default_rng(seed)                 # seeded generator -> same init every run
    return {
        "W1": rng.normal(0, 0.01, (n_in, n_hidden)),  # (30,16): column j = weights into hidden unit j
        "b1": np.zeros(n_hidden),                     # (16,): one bias per hidden unit
        "W2": rng.normal(0, 0.01, (n_hidden, 1)),     # (16,1): one weight per hidden unit into z2
        "b2": np.zeros(1),                            # (1,): the final bias
    }                                                 # small weights -> z ≈ 0 -> p ≈ 0.5 at the start


def forward(params, X):
    """X (n,30) -> p (n,1). Also return the intermediates the backward pass will need."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1              # (n,30)@(30,16) -> (n,16); b1 (16,) broadcasts across rows
    h = np.maximum(z1, 0)         # ReLU, elementwise: the only non-linearity before the sigmoid
    z2 = h @ W2 + b2              # (n,16)@(16,1) -> (n,1): the logistic-regression step on h
    p = 1 / (1 + np.exp(-z2))     # sigmoid, elementwise -> (n,1) probabilities
    cache = {"X": X, "z1": z1, "h": h}   # kept for backprop: each gradient needs the layer's input
    return p, cache


def loss(p, y):
    """Binary cross-entropy, averaged over rows. p (n,1) probabilities, y (n,) labels in {0,1}."""
    p = np.clip(p, 1e-7, 1 - 1e-7)       # log(0) is -inf; keep p strictly inside (0,1)
    y = y.reshape(-1, 1)                 # (n,) -> (n,1) so it lines up with p elementwise
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))   # per row only one term is alive


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    params = init_params()
    n_params = sum(v.size for v in params.values())
    print("parameters:", n_params)                       # expect 513

    p, cache = forward(params, X_train)
    print("p shape", p.shape, "min", p.min().round(4), "max", p.max().round(4))  # (397,1), both ≈ 0.5
    for k, v in cache.items():
        print(f"  {k}: {v.shape}")                       # X (397,30), z1 (397,16), h (397,16)

    L = loss(p, y_train)
    print("loss at init", L.round(4), "  ln 2 =", np.log(2).round(4))
    assert abs(L - np.log(2)) < 0.01, "forward pass or loss is wrong — check shapes first"