# Ensemble Methods

Ensembles combine multiple base estimators to improve generalization and reduce variance or bias over any single model.

Two main families:
- **Averaging methods** (Bagging, Random Forests): parallel estimators, average predictions.
- **Boosting methods** (Gradient Boosting, AdaBoost): sequential estimators, each correcting the previous.

---

## Gradient-Boosted Trees (GBDT)

Fits trees sequentially; each tree models the **negative gradient** of the loss of the current ensemble. Excellent for tabular data, both regression and classification.

### HistGradientBoosting (recommended)

`HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` — inspired by LightGBM. Orders-of-magnitude faster than classic GBDT for n_samples > ~10k.

**How it's faster:** bins continuous features into integer-valued bins (up to 255 by default) once before training. Node-splitting cost drops from $O(n_\text{features} \cdot n \log n)$ to $O(n_\text{features} \cdot n)$.

**Built-in support for:**
- Missing values (learns per-split direction).
- Categorical features (`categorical_features` parameter — better than one-hot encoding).
- Sample weights.
- Monotonic constraints (`monotonic_cst`).
- Interaction constraints (`interaction_cst`).
- Early stopping (default when n_samples > 10,000).

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)
clf.fit(X_train, y_train)
```

**Regression losses:** `'squared_error'`, `'absolute_error'`, `'gamma'`, `'poisson'`, `'quantile'`.  
**Classification loss:** `'log_loss'` (only option; auto-selects binary/multi-class based on y).

**Key regularization params:** `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`, `max_bins` (fewer bins = more regularization).

### Classic GradientBoosting

`GradientBoostingClassifier` / `GradientBoostingRegressor` — same concept, slower but more mature feature set.

**Key params:** `n_estimators` (number of trees), `learning_rate`, `max_depth`/`max_leaf_nodes`, `subsample` (stochastic gradient boosting, e.g. 0.5).

**Learning rate & n_estimators tradeoff:** smaller `learning_rate` needs more trees. Use `learning_rate ≤ 0.1` and tune `n_estimators` with early stopping.

```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_train)
clf.feature_importances_  # impurity-based
```

**Warm start:** `warm_start=True` lets you add more estimators to an already-fitted model.

---

## Random Forests

Each tree is trained on a bootstrap sample of the data, with a random subset of `max_features` features considered at each split. Averages predictions across all trees.

- Reduces variance via two sources of randomness: sample bootstrapping + feature subsampling.
- Trees are built to full depth (high individual variance, low individual bias).

**Good defaults:**
- `max_features='sqrt'` for classification; `1.0` (all features) for regression.
- `max_depth=None` + `min_samples_split=2` (fully grown trees).
- `n_estimators`: more is better up to a point; start with 100.
- `oob_score=True`: free estimate of generalization error using out-of-bag samples.

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True)
clf.fit(X, y)
clf.oob_score_
clf.feature_importances_  # mean decrease in impurity (MDI)
```

**Warning on feature importances:** MDI favors high-cardinality features and reflects training-set statistics. Prefer `permutation_importance` for held-out data.

**Warm start** is supported; add trees incrementally with `set_params(n_estimators=200, warm_start=True)`.

### Extremely Randomized Trees (Extra-Trees)

Like Random Forests, but thresholds are randomly drawn for each feature rather than optimized. Even more randomization → slightly higher bias but often lower variance.

```python
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100, max_features='sqrt')
```

### RandomForests vs HistGradientBoosting

| | Random Forests | HistGBT |
|-|---------------|---------|
| Trees | Deep (overfit individually) | Shallow (underfit individually) |
| Training | Parallel | Sequential |
| Speed on large data | Slower | Much faster (binning) |
| Missing values | Not built-in | Built-in |
| Categorical features | Not built-in | Native support |

---

## Bagging

`BaggingClassifier` / `BaggingRegressor` — wraps any estimator and trains it on random subsets.

Variants by what is subsampled:
- **Pasting:** samples without replacement.
- **Bagging:** samples with replacement (default, `bootstrap=True`).
- **Random Subspaces:** features without replacement (`bootstrap_features=False`).
- **Random Patches:** both samples and features subsampled.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
```

Works best with strong, complex base estimators. Use `oob_score=True` for free test-error estimate.

---

## Voting

Combines diverse estimators by majority vote (hard) or averaged probabilities (soft).

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

clf = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB())],
    voting='soft',
    weights=[2, 1, 1]  # optional
)
```

**Hard voting:** majority class label.  
**Soft voting:** argmax of weighted average predicted probabilities. Requires `predict_proba` on all estimators.

`VotingRegressor` averages predicted values.

---

## Stacking

Uses predictions of base estimators as features for a meta-estimator. Base estimators are trained on the full training set; the meta-estimator is trained on out-of-sample predictions via cross-validation (to avoid overfitting).

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import GradientBoostingClassifier

estimators = [('lr', LogisticRegression()), ('gb', GradientBoostingClassifier())]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
clf.fit(X_train, y_train)
clf.transform(X_test)  # get base estimator predictions
```

Can achieve better accuracy than any individual base estimator by combining their strengths. Computationally expensive. Multi-layer stacking is supported by nesting `StackingClassifier`/`StackingRegressor` as the `final_estimator`.

---

## AdaBoost

Fits a sequence of weak learners (typically shallow trees) on weighted data. Misclassified samples get higher weights each round, forcing subsequent learners to focus on hard examples. Final prediction is a weighted majority vote.

- `AdaBoostClassifier`: uses AdaBoost.SAMME.
- `AdaBoostRegressor`: uses AdaBoost.R2.

```python
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
clf.fit(X, y)
```

**Key params:** `n_estimators`, `learning_rate` (shrinks each estimator's contribution), `estimator` (default: `DecisionTreeClassifier(max_depth=1)`).

Works best with weak base estimators (shallow trees). Contrast with Bagging, which works best with strong/complex estimators.
