# ml-bio-ladder
Three small ML-for-biology builds, each against a baseline

## Rung 1 — feed-forward network from scratch vs logistic regression

Data: breast cancer Wisconsin (diagnostic), 569 samples × 30 features, shipped with scikit-learn.
Split: 70/15/15 train/val/test, stratified, seed 0. Features standardised on the training set only.

### Baseline
Logistic regression (scikit-learn, L-BFGS, max_iter=1000).
val 0.977 · **test 0.965** (86 samples each; one sample = 1.2 %)

### Network
*(pending)*
