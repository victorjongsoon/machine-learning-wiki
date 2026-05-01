---
title: "LinearRegression"
source: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?utm_source=chatgpt.com"
author:
published:
created: 2026-05-01
description: "Gallery examples: Principal Component Regression vs Partial Least Squares Regression Plot individual and voting regression predictions Failure of Machine Learning to infer causal effects Comparing ..."
tags:
  - "clippings"
---
*class* sklearn.linear\_model.LinearRegression(*\**, *fit\_intercept=True*, *copy\_X=True*, *tol=1e-06*, *n\_jobs=None*, *positive=False*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_base.py#L470) [#](#sklearn.linear_model.LinearRegression "Link to this definition")

Ordinary least squares Linear Regression.

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

Parameters:

**fit\_intercept** bool, default=True

Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

**copy\_X** bool, default=True

If True, X will be copied; else, it may be overwritten.

**tol** float, default=1e-6

The precision of the solution (`coef_`) is determined by `tol` which specifies a different convergence criterion for the `lsqr` solver. `tol` is set as `atol` and `btol` of [`scipy.sparse.linalg.lsqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr "(in SciPy v1.17.0)") when fitting on sparse training data. This parameter has no effect when fitting on dense data.

Added in version 1.7.

**n\_jobs** int, default=None

The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly `n_targets > 1` and secondly `X` is sparse or if `positive` is set to `True`. `None` means 1 unless in a [`joblib.parallel_backend`](https://joblib.readthedocs.io/en/latest/generated/joblib.parallel_backend.html#joblib.parallel_backend "(in joblib v1.6.dev0)") context. `-1` means using all processors. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-n_jobs) for more details.

**positive** bool, default=False

When set to `True`, forces the coefficients to be positive. This option is only supported for dense arrays.

For a comparison between a linear regression model with positive constraints on the regression coefficients and a linear regression without such constraints, see [Non-negative least squares](https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py).

Added in version 0.24.

Attributes:

**coef\_** array of shape (n\_features, ) or (n\_targets, n\_features)

Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n\_targets, n\_features), while if only one target is passed, this is a 1D array of length n\_features.

**rank\_** int

Rank of matrix `X`. Only available when `X` is dense.

**singular\_** array of shape (min(X, y),)

Singular values of `X`. Only available when `X` is dense.

**intercept\_** float or array of shape (n\_targets,)

Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.

**n\_features\_in\_** int

Number of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

Added in version 0.24.

**feature\_names\_in\_** ndarray of shape (`n_features_in_`,)

Names of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when `X` has feature names that are all strings.

Added in version 1.0.

> [!note] See also
> [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge")
> 
> Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients with l2 regularization.
> 
> [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso "sklearn.linear_model.Lasso")
> 
> The Lasso is a linear model that estimates sparse coefficients with l1 regularization.
> 
> [`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet "sklearn.linear_model.ElasticNet")
> 
> Elastic-Net is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients.

Notes

From the implementation point of view, this is just plain Ordinary Least Squares ([`scipy.linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq "(in SciPy v1.17.0)")) or Non Negative Least Squares ([`scipy.optimize.nnls`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls "(in SciPy v1.17.0)")) wrapped as a predictor object.

Examples

```
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.coef_
array([1., 2.])
>>> reg.intercept_
np.float64(3.0)
>>> reg.predict(np.array([[3, 5]]))
array([16.])
```

fit(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_base.py#L602) [#](#sklearn.linear_model.LinearRegression.fit "Link to this definition")

Fit linear model.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

Training data.

**y** array-like of shape (n\_samples,) or (n\_samples, n\_targets)

Target values. Will be cast to X’s dtype if necessary.

**sample\_weight** array-like of shape (n\_samples,), default=None

Individual weights for each sample.

Added in version 0.17: parameter *sample\_weight* support to LinearRegression.

Returns:

**self** object

Fitted Estimator.

Get metadata routing of this object.

Please check [User Guide](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing) on how the routing mechanism works.

Returns:

**routing** MetadataRequest

A [`MetadataRequest`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.metadata_routing.MetadataRequest.html#sklearn.utils.metadata_routing.MetadataRequest "sklearn.utils.metadata_routing.MetadataRequest") encapsulating routing information.

get\_params(*deep=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L240) [#](#sklearn.linear_model.LinearRegression.get_params "Link to this definition")

Get parameters for this estimator.

Parameters:

**deep** bool, default=True

If True, will return the parameters for this estimator and contained subobjects that are estimators.

Returns:

**params** dict

Parameter names mapped to their values.

predict(*X*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/linear_model/_base.py#L297) [#](#sklearn.linear_model.LinearRegression.predict "Link to this definition")

Predict using the linear model.

Parameters:

**X** array-like or sparse matrix, shape (n\_samples, n\_features)

Samples.

Returns:

**C** array, shape (n\_samples,)

Returns predicted values.

score(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L610) [#](#sklearn.linear_model.LinearRegression.score "Link to this definition")

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

set\_fit\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.linear_model.LinearRegression.set_fit_request "Link to this definition")

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

set\_params(*\*\*params*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L338) [#](#sklearn.linear_model.LinearRegression.set_params "Link to this definition")

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects (such as [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline")). The latter have parameters of the form `<component>__<parameter>` so that it’s possible to update each component of a nested object.

Parameters:

**\*\*params** dict

Estimator parameters.

Returns:

**self** estimator instance

Estimator instance.

set\_score\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.linear_model.LinearRegression.set_score_request "Link to this definition")

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