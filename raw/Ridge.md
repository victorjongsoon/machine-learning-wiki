---
title: "Ridge"
source: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?utm_source=chatgpt.com"
author:
published:
created: 2026-05-01
description: "Gallery examples: Prediction Latency Compressive sensing: tomography reconstruction with L1 prior (Lasso) Comparison of kernel ridge and Gaussian process regression Imputing missing values with var..."
tags:
  - "raw"
---
*class* sklearn.linear\_model.Ridge(*alpha=1.0*, *\**, *fit\_intercept=True*, *copy\_X=True*, *max\_iter=None*, *tol=0.0001*, *solver='auto'*, *positive=False*, *random\_state=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_ridge.py#L1028) [#](#sklearn.linear_model.Ridge "Link to this definition")

Linear least squares with l2 regularization.

Minimizes the objective function:

```
||y - Xw||^2_2 + alpha * ||w||^2_2
```

This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape (n\_samples, n\_targets)).

Read more in the [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).

Parameters:

**alpha** {float, ndarray of shape (n\_targets,)}, default=1.0

Constant that multiplies the L2 term, controlling regularization strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

When `alpha = 0`, the objective is equivalent to ordinary least squares, solved by the [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression") object. For numerical reasons, using `alpha = 0` with the `Ridge` object is not advised. Instead, you should use the [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression") object.

If an array is passed, penalties are assumed to be specific to the targets. Hence they must correspond in number.

**fit\_intercept** bool, default=True

Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations (i.e. `X` and `y` are expected to be centered).

**copy\_X** bool, default=True

If True, X will be copied; else, it may be overwritten.

**max\_iter** int, default=None

Maximum number of iterations for conjugate gradient solver. For ‘sparse\_cg’ and ‘lsqr’ solvers, the default value is determined by scipy.sparse.linalg. For ‘sag’ solver, the default value is 1000. For ‘lbfgs’ solver, the default value is 15000.

**tol** float, default=1e-4

The precision of the solution (`coef_`) is determined by `tol` which specifies a different convergence criterion for each solver:

- ‘svd’: `tol` has no impact.
- ‘cholesky’: `tol` has no impact.
- ‘sparse\_cg’: norm of residuals smaller than `tol`.
- ‘lsqr’: `tol` is set as atol and btol of scipy.sparse.linalg.lsqr, which control the norm of the residual vector in terms of the norms of matrix and coefficients.
- ‘sag’ and ‘saga’: relative change of coef smaller than `tol`.
- ‘lbfgs’: maximum of the absolute (projected) gradient=max|residuals| smaller than `tol`.

Changed in version 1.2: Default value changed from 1e-3 to 1e-4 for consistency with other linear models.

**solver** {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse\_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’

Solver to use in the computational routines:

- ‘auto’ chooses the solver automatically based on the type of data.
- ‘svd’ uses a Singular Value Decomposition of X to compute the Ridge coefficients. It is the most stable solver, in particular more stable for singular matrices than ‘cholesky’ at the cost of being slower.
- ‘cholesky’ uses the standard [`scipy.linalg.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve "(in SciPy v1.17.0)") function to obtain a closed-form solution.
- ‘sparse\_cg’ uses the conjugate gradient solver as found in [`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.cg "(in SciPy v1.17.0)"). As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set `tol` and `max_iter`).
- ‘lsqr’ uses the dedicated regularized least-squares routine [`scipy.sparse.linalg.lsqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr "(in SciPy v1.17.0)"). It is the fastest and uses an iterative procedure.
- ‘sag’ uses a Stochastic Average Gradient descent, and ‘saga’ uses its improved, unbiased version named SAGA. Both methods also use an iterative procedure, and are often faster than other solvers when both n\_samples and n\_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from [`sklearn.preprocessing`](https://scikit-learn.org/stable/api/sklearn.preprocessing.html#module-sklearn.preprocessing "sklearn.preprocessing").
- ‘lbfgs’ uses L-BFGS-B algorithm implemented in [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize "(in SciPy v1.17.0)"). It can be used only when `positive` is True.

All solvers except ‘svd’ support both dense and sparse data. However, only ‘lsqr’, ‘sag’, ‘sparse\_cg’, and ‘lbfgs’ support sparse input when `fit_intercept` is True.

Added in version 0.17: Stochastic Average Gradient descent solver.

Added in version 0.19: SAGA solver.

**positive** bool, default=False

When set to `True`, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.

**random\_state** int, RandomState instance, default=None

Used when `solver` == ‘sag’ or ‘saga’ to shuffle the data. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state) for details.

Added in version 0.17: `random_state` to support Stochastic Average Gradient.

Attributes:

**coef\_** ndarray of shape (n\_features,) or (n\_targets, n\_features)

Weight vector(s).

**intercept\_** float or ndarray of shape (n\_targets,)

Independent term in decision function. Set to 0.0 if `fit_intercept = False`.

**n\_iter\_** None or ndarray of shape (n\_targets,)

Actual number of iterations for each target. Available only for ‘sag’ and ‘lsqr’ solvers. Other solvers will return None.

Added in version 0.17.

**n\_features\_in\_** int

Number of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

Added in version 0.24.

**feature\_names\_in\_** ndarray of shape (`n_features_in_`,)

Names of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when `X` has feature names that are all strings.

Added in version 1.0.

**solver\_** str

The solver that was used at fit time by the computational routines.

Added in version 1.5.

> [!note] See also
> [`RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier "sklearn.linear_model.RidgeClassifier")
> 
> Ridge classifier.
> 
> [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV "sklearn.linear_model.RidgeCV")
> 
> Ridge regression with built-in cross validation.
> 
> [`KernelRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge "sklearn.kernel_ridge.KernelRidge")
> 
> Kernel ridge regression combines ridge regression with the kernel trick.

Notes

Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. Alpha corresponds to `1 / (2C)` in other linear models such as [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") or [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC").

Examples

```
>>> from sklearn.linear_model import Ridge
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = Ridge(alpha=1.0)
>>> clf.fit(X, y)
Ridge()
```

fit(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_ridge.py#L1231) [#](#sklearn.linear_model.Ridge.fit "Link to this definition")

Fit Ridge regression model.

Parameters:

**X** {ndarray, sparse matrix} of shape (n\_samples, n\_features)

Training data.

**y** ndarray of shape (n\_samples,) or (n\_samples, n\_targets)

Target values.

**sample\_weight** float or ndarray of shape (n\_samples,), default=None

Individual weights for each sample. If given a float, every sample will have the same weight.

Returns:

**self** object

Fitted estimator.

Get metadata routing of this object.

Please check [User Guide](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing) on how the routing mechanism works.

Returns:

**routing** MetadataRequest

A [`MetadataRequest`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.metadata_routing.MetadataRequest.html#sklearn.utils.metadata_routing.MetadataRequest "sklearn.utils.metadata_routing.MetadataRequest") encapsulating routing information.

get\_params(*deep=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L240) [#](#sklearn.linear_model.Ridge.get_params "Link to this definition")

Get parameters for this estimator.

Parameters:

**deep** bool, default=True

If True, will return the parameters for this estimator and contained subobjects that are estimators.

Returns:

**params** dict

Parameter names mapped to their values.

predict(*X*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_base.py#L297) [#](#sklearn.linear_model.Ridge.predict "Link to this definition")

Predict using the linear model.

Parameters:

**X** array-like or sparse matrix, shape (n\_samples, n\_features)

Samples.

Returns:

**C** array, shape (n\_samples,)

Returns predicted values.

score(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L610) [#](#sklearn.linear_model.Ridge.score "Link to this definition")

Return [coefficient of determination](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score) on test data.

The coefficient of determination, $R^{2}$, is defined as $\left(\right. 1 - \frac{u}{v} \left.\right)$, where $u$ is the residual sum of squares `((y_true - y_pred)** 2).sum()` and $v$ is the total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of `y`, disregarding the input features, would get a $R^{2}$ score of 0.0.

Parameters:

**X** array-like of shape (n\_samples, n\_features)

Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape `(n_samples, n_samples_fitted)`, where `n_samples_fitted` is the number of samples used in the fitting for the estimator.

**y** array-like of shape (n\_samples,) or (n\_samples, n\_outputs)

True values for `X`.

**sample\_weight** array-like of shape (n\_samples,), default=None

Sample weights.

Returns:

**score** float

$R^{2}$ of `self.predict(X)` w.r.t. `y`.

Notes

The $R^{2}$ score used when calling `score` on a regressor uses `multioutput='uniform_average'` from version 0.23 to keep consistent with default value of [`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score "sklearn.metrics.r2_score"). This influences the `score` method of all the multioutput regressors (except for [`MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor "sklearn.multioutput.MultiOutputRegressor")).

set\_fit\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.linear_model.Ridge.set_fit_request "Link to this definition")

Configure whether metadata should be requested to be passed to the `fit` method.

Note that this method is only relevant when this estimator is used as a sub-estimator within a [meta-estimator](https://scikit-learn.org/stable/glossary.html#term-meta-estimator) and metadata routing is enabled with `enable_metadata_routing=True` (see [`sklearn.set_config`](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config "sklearn.set_config")). Please check the [User Guide](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing) on how the routing mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `fit` if provided. The request is ignored if metadata is not provided.
- `False`: metadata is not requested and the meta-estimator will not pass it to `fit`.
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the existing request. This allows you to change the request for some parameters and not others.

Added in version 1.3.

Parameters:

**sample\_weight** str, True, False, or None, default=sklearn.utils.metadata\_routing.UNCHANGED

Metadata routing for `sample_weight` parameter in `fit`.

Returns:

**self** object

The updated object.

set\_params(*\*\*params*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L338) [#](#sklearn.linear_model.Ridge.set_params "Link to this definition")

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects (such as [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline")). The latter have parameters of the form `<component>__<parameter>` so that it’s possible to update each component of a nested object.

Parameters:

**\*\*params** dict

Estimator parameters.

Returns:

**self** estimator instance

Estimator instance.

set\_score\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.linear_model.Ridge.set_score_request "Link to this definition")

Configure whether metadata should be requested to be passed to the `score` method.

Note that this method is only relevant when this estimator is used as a sub-estimator within a [meta-estimator](https://scikit-learn.org/stable/glossary.html#term-meta-estimator) and metadata routing is enabled with `enable_metadata_routing=True` (see [`sklearn.set_config`](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config "sklearn.set_config")). Please check the [User Guide](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing) on how the routing mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `score` if provided. The request is ignored if metadata is not provided.
- `False`: metadata is not requested and the meta-estimator will not pass it to `score`.
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the existing request. This allows you to change the request for some parameters and not others.

Added in version 1.3.

Parameters:

**sample\_weight** str, True, False, or None, default=sklearn.utils.metadata\_routing.UNCHANGED

Metadata routing for `sample_weight` parameter in `score`.

Returns:

**self** object

The updated object.