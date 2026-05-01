# Linear Models

Linear models predict a target as a linear combination of features:

$$\hat{y}(w, x) = w_0 + w_1 x_1 + \ldots + w_p x_p$$

Coefficients are stored in `coef_` ($w_1,\ldots,w_p$) and `intercept_` ($w_0$).

---

## Ordinary Least Squares (OLS)

`LinearRegression` minimizes the residual sum of squares:

$$\min_w \|Xw - y\|_2^2$$

Solved via SVD; complexity $O(n_\text{samples} \cdot n_\text{features}^2)$.

**Pitfall:** Sensitive to multicollinearity — correlated features inflate coefficient variance.

**Non-negative variant:** Pass `positive=True` to constrain all coefficients ≥ 0.

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
reg.coef_, reg.intercept_
```

---

## Ridge Regression

Adds an ℓ₂ penalty to shrink coefficients and reduce variance from multicollinearity:

$$\min_w \|Xw - y\|_2^2 + \alpha \|w\|_2^2$$

Larger `alpha` → stronger shrinkage. Use `alpha=0` is numerically discouraged; use `LinearRegression` instead.

**Auto solver selection** (`solver='auto'`):

| Condition | Solver |
|-----------|--------|
| `positive=True` | lbfgs |
| Dense X | cholesky |
| Sparse X | sparse_cg |

**Cross-validated alpha:** `RidgeCV` uses efficient leave-one-out CV. For k-fold, set `cv=10`.

**Classification variant:** `RidgeClassifier` converts labels to {-1, 1} and solves regression. Can be faster than `LogisticRegression` for many classes because it computes $(X^TX)^{-1}X^T$ once.

```python
from sklearn.linear_model import Ridge, RidgeCV
reg = Ridge(alpha=0.5).fit(X, y)
cv_reg = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
cv_reg.alpha_  # best alpha found
```

---

## Lasso

ℓ₁ penalty drives some coefficients to exactly zero → **sparse models** / **feature selection**:

$$\min_w \frac{1}{2 n} \|Xw - y\|_2^2 + \alpha \|w\|_1$$

Uses coordinate descent. `LassoCV` or `LassoLarsCV` select `alpha` by CV.

- Use `LassoCV` for many collinear features (high-dimensional data).
- Use `LassoLarsCV` when `n_samples << n_features` (explores more alpha values efficiently).
- Use `LassoLarsIC` with AIC/BIC for cheap model selection (one path, no refit).

```python
from sklearn.linear_model import Lasso, LassoCV
reg = Lasso(alpha=0.1).fit(X, y)
cv_reg = LassoCV(cv=5).fit(X, y)
```

**Note:** `alpha` in Lasso corresponds to `1/C` in SVM (`alpha = 1/(n_samples * C)` depending on estimator).

---

## Elastic-Net

Combines ℓ₁ and ℓ₂ penalties via `l1_ratio` ($\rho$):

$$\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \alpha\rho\|w\|_1 + \frac{\alpha(1-\rho)}{2}\|w\|_2^2$$

- `l1_ratio=1` → Lasso; `l1_ratio=0` → Ridge.
- Useful when multiple correlated features exist: Lasso picks one at random; Elastic-Net tends to keep both.
- `ElasticNetCV` tunes both `alpha` and `l1_ratio`.

---

## Multi-task Lasso / Elastic-Net

For multiple related regression problems (`y` is 2D). Forces the same features to be selected across all tasks.

- `MultiTaskLasso`: mixed ℓ₁/ℓ₂ (ℓ₂,₁ norm) on the coefficient matrix.
- `MultiTaskElasticNet`: adds ℓ₂ (Frobenius) regularization as well.

---

## LARS (Least Angle Regression)

Efficient for `n_features >> n_samples`. Produces the full piecewise-linear regularization path at near-zero extra cost. Sensitive to noise.

`LassoLars` implements Lasso via LARS: yields the exact piecewise-linear solution as a function of the ℓ₁ norm.

---

## Orthogonal Matching Pursuit (OMP)

Greedy forward-selection with a constraint on the number of non-zero coefficients ($\|w\|_0 \leq k$) or on the error ($\|y - Xw\|_2^2 \leq \text{tol}$). Re-projects the residual onto the selected atoms at each step.

---

## Bayesian Regression

Treats regularization parameters as random variables estimated from data rather than set by hand.

**Bayesian Ridge:** Places a Gaussian prior on $w$ and Gamma priors on the precision parameters $\alpha, \lambda$; estimates all jointly via log marginal likelihood maximization. More robust to ill-posed problems.

**ARD (Automatic Relevance Determination):** Each coefficient $w_i$ has its own precision $\lambda_i$, leading to sparser solutions. Also called *Sparse Bayesian Learning* or *Relevance Vector Machine*.

---

## Logistic Regression

A linear classifier (despite the name). Models class probabilities via the logistic function:

$$\hat{p}(X_i) = \frac{1}{1 + \exp(-X_i w - w_0)}$$

Regularization is applied by default (controlled by `C = 1/alpha`; higher `C` → less regularization).

**Penalty options:** `none`, `l1`, `l2` (default), `elasticnet`.

**Solver guide:**

| Goal | Recommended solver |
|------|--------------------|
| Default / robust | `lbfgs` |
| Large n_samples >> n_features | `newton-cholesky` |
| Large datasets, low precision | `sag` / `saga` |
| L1 or Elastic-Net penalty | `saga` |
| Multinomial | any except `liblinear` |

`LogisticRegressionCV` does built-in CV over `C` and `l1_ratio`.

---

## Generalized Linear Models (GLM)

Extend linear regression via a link function $h$ and a non-Gaussian loss:

$$\hat{y}(w, X) = h(Xw)$$

`TweedieRegressor(power=p)` covers Normal (p=0), Poisson (p=1), Gamma (p=2), Inverse-Gaussian (p=3). Convenience aliases: `PoissonRegressor`, `GammaRegressor`.

**Choosing distribution:**
- Count/frequency targets → Poisson with log-link.
- Positive, skewed targets → Gamma.
- Heavier tailed than Gamma → Inverse-Gaussian.
- Probabilities → Bernoulli (binary) or Categorical (multiclass).

---

## SGD-based Linear Models

`SGDClassifier` and `SGDRegressor` fit linear models via stochastic gradient descent. Very fast for large datasets. Requires feature scaling.

- `loss='log_loss'` → logistic regression.
- `loss='hinge'` → linear SVM.
- `partial_fit` supports online / out-of-core learning.

**Perceptron:** SGD with perceptron loss, no regularization, updates only on mistakes.

**Passive-Aggressive:** Online learning without a learning rate; includes a regularization parameter `eta0`.

---

## Quantile Regression

`QuantileRegressor` estimates conditional quantiles (e.g., median) by minimizing the pinball loss. Useful for prediction intervals with heteroscedastic or non-normal errors.

---

## Polynomial Regression

Use `PolynomialFeatures` to generate polynomial/interaction features, then apply any linear model:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
model.fit(x[:, None], y)
```

Set `interaction_only=True` to include only cross-terms (useful for boolean features).
