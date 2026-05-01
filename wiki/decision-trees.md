# Decision Trees

Non-parametric supervised learning for classification and regression. The model partitions feature space into axis-aligned regions using a sequence of if-then-else rules.

---

## Overview

**Pros:**
- Interpretable and visualizable.
- No feature scaling needed.
- Handles numerical and multi-output problems.
- Inference is $O(\log n_\text{samples})$.

**Cons:**
- High variance; prone to overfitting. Mitigate with pruning or ensembles.
- Unstable — small data changes can produce completely different trees.
- Piecewise-constant predictions; poor extrapolation.
- Greedy construction (CART) finds locally optimal splits, not globally optimal.
- Biased toward dominant classes in imbalanced datasets.

scikit-learn uses an optimized **CART** implementation.

---

## Classification: `DecisionTreeClassifier`

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=None)
clf.fit(X, y)
clf.predict(X_new)
clf.predict_proba(X_new)  # fraction of training samples per class in the leaf
```

**Key parameters:**

| Parameter | Effect |
|-----------|--------|
| `criterion` | Split quality: `'gini'` (default), `'entropy'` / `'log_loss'` |
| `max_depth` | Cap tree depth to limit overfitting |
| `min_samples_split` | Min samples to split a node (int or fraction) |
| `min_samples_leaf` | Min samples in a leaf — smooths regression |
| `max_features` | Features considered per split: `None` (all), `'sqrt'`, `'log2'`, int, or float |
| `class_weight` | Handle imbalance: dict or `'balanced'` |
| `ccp_alpha` | Cost-complexity pruning parameter |

**Multiclass:** ties broken by lowest class index.

---

## Regression: `DecisionTreeRegressor`

Same API; `y` is float. Leaf prediction is the mean (MSE/Poisson criteria) or median (MAE criterion).

**Regression criteria:**

| Criterion | Leaf predicts | Notes |
|-----------|--------------|-------|
| `'squared_error'` (default) | mean | Fast |
| `'poisson'` | mean | Good for counts (requires $y \geq 0$) |
| `'absolute_error'` (MAE) | median | Robust to outliers; 3–6× slower |

---

## Multi-output

Pass a 2D `y` of shape `(n_samples, n_outputs)`. The tree stores all output values in each leaf and minimizes average impurity across outputs.

---

## Mathematical Formulation

At each node $m$, find the best split $\theta^* = (j, t_m)$ (feature $j$, threshold $t_m$) to minimize the weighted impurity:

$$G(Q_m, \theta) = \frac{n_m^\text{left}}{n_m} H(Q_m^\text{left}(\theta)) + \frac{n_m^\text{right}}{n_m} H(Q_m^\text{right}(\theta))$$

**Classification impurity functions:**
- Gini: $H = \sum_k p_{mk}(1 - p_{mk})$
- Entropy: $H = -\sum_k p_{mk} \log(p_{mk})$

**Splitter strategies:**
- `splitter='best'` — exhaustive greedy search over all features and thresholds.
- `splitter='random'` — random threshold per feature; faster, slightly higher bias.

---

## Missing Values

`DecisionTreeClassifier` / `DecisionTreeRegressor` support missing values natively with `splitter='best'`:
- During training, missing values are assigned to whichever child (left/right) gives the best impurity reduction.
- At prediction, missing values follow the direction learned during training. If the feature was never missing in training, they go to the child with the most samples.

`ExtraTreeClassifier`/`ExtraTreeRegressor` handle missing values similarly but with random splits.

---

## Pruning

### Pre-pruning (stopping criteria)
Control tree size at training time:
- `max_depth` — most effective first control.
- `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `min_impurity_decrease`, `max_leaf_nodes`.

### Post-pruning: Minimal Cost-Complexity Pruning

For a subtree $T_t$ rooted at $t$, define:

$$\alpha_\text{eff}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

The node with the smallest $\alpha_\text{eff}$ (the "weakest link") is pruned first. Set `ccp_alpha` to control how aggressively to prune.

```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
# path.ccp_alphas, path.impurities
# Pick ccp_alpha via cross-validation, then refit
```

---

## Complexity

| Phase | Cost |
|-------|------|
| Training (best splitter) | $O(n_\text{features} \cdot n_\text{samples}^2 \log n_\text{samples})$ |
| Training (random splitter) | $O(n_\text{features} \cdot n_\text{samples}^2)$ |
| Inference | $O(\text{depth})$ ≈ $O(\log n_\text{samples})$ for balanced trees |

---

## Practical Tips

- Start with `max_depth=3` to understand the tree, then increase.
- Prefer `min_samples_leaf=5` as a starting point; use a float for relative sizes.
- Balance imbalanced datasets before fitting (or use `class_weight='balanced'`).
- For high-dimensional sparse input: convert to `csc_matrix` before `fit`, `csr_matrix` before `predict`.
- Decision trees overfit with many features and few samples — consider PCA or feature selection first.
- For quantile targets use `criterion='poisson'` (counts) or `criterion='absolute_error'` (robust regression).

---

## Visualization

```python
from sklearn.tree import plot_tree, export_text
plot_tree(clf, feature_names=..., class_names=..., filled=True)
print(export_text(clf, feature_names=...))
```
