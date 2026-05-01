---
title: "1.1. Linear Models"
source: "https://scikit-learn.org/stable/modules/linear_model.html?utm_source=chatgpt.com"
author:
published:
created: 2026-05-01
description:
tags:
  - "raw/"
---
The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the features. In mathematical notation, if $\hat{y}$ is the predicted value.

$$
\hat{y} \left(\right. w , x \left.\right) = w_{0} + w_{1} x_{1} + . . . + w_{p} x_{p}
$$

Across the module, we designate the vector $w = \left(\right. w_{1} , . . . , w_{p} \left.\right)$ as `coef_` and $w_{0}$ as `intercept_`.

To perform classification with generalized linear models, see.

## 1.1.1. Ordinary Least Squares

[`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression") fits a linear model with coefficients $w = \left(\right. w_{1} , . . . , w_{p} \left.\right)$ to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. Mathematically it solves a problem of the form:

$$
\underset{w}{min} \left|\right. \left|\right. X w - y \left|\right. \left|\right._{2}^{2}
$$
![../_images/sphx_glr_plot_ols_ridge_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_ridge_001.png)

[`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression") takes in its `fit` method arguments `X`, `y`, `sample_weight` and stores the coefficients $w$ of the linear model in its `coef_` and `intercept_` attributes:

```
>>> from sklearn import linear_model
>>> reg = linear_model.LinearRegression()
>>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression()
>>> reg.coef_
array([0.5, 0.5])
>>> reg.intercept_
0.0
```

The coefficient estimates for Ordinary Least Squares rely on the independence of the features. When features are correlated and some columns of the design matrix $X$ have an approximately linear dependence, the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed target, producing a large variance. This situation of *multicollinearity* can arise, for example, when data are collected without an experimental design.

Examples

- [Ordinary Least Squares and Ridge Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-py)

### 1.1.1.1. Non-Negative Least Squares

It is possible to constrain all the coefficients to be non-negative, which may be useful when they represent some physical or naturally non-negative quantities (e.g., frequency counts or prices of goods). [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression") accepts a boolean `positive` parameter: when set to `True` [Non-Negative Least Squares](https://en.wikipedia.org/wiki/Non-negative_least_squares) are then applied.

Examples

- [Non-negative least squares](https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py)

### 1.1.1.2. Ordinary Least Squares Complexity

The least squares solution is computed using the singular value decomposition of $X$. If $X$ is a matrix of shape `(n_samples, n_features)` this method has a cost of $O \left(\right. n_{\text{samples}} n_{\text{features}}^{2} \left.\right)$, assuming that $n_{\text{samples}} \geq n_{\text{features}}$.

## 1.1.2. Ridge regression and classification

### 1.1.2.1. Regression

[`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") regression addresses some of the problems of by imposing a penalty on the size of the coefficients. The ridge coefficients minimize a penalized residual sum of squares:

$$
\underset{w}{min} \left|\right. \left|\right. X w - y \left|\right. \left|\right._{2}^{2} + \alpha \left|\right. \left|\right. w \left|\right. \left|\right._{2}^{2}
$$

The complexity parameter $\alpha \geq 0$ controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

![../_images/sphx_glr_plot_ridge_path_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_ridge_path_001.png)

As with other linear models, [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") will take in its `fit` method arrays `X`, `y` and will store the coefficients $w$ of the linear model in its `coef_` member:

```
>>> from sklearn import linear_model
>>> reg = linear_model.Ridge(alpha=.5)
>>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
Ridge(alpha=0.5)
>>> reg.coef_
array([0.34545455, 0.34545455])
>>> reg.intercept_
np.float64(0.13636)
```

Note that the class [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") allows for the user to specify that the solver be automatically chosen by setting `solver="auto"`. When this option is specified, [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") will choose between the `"lbfgs"`, `"cholesky"`, and `"sparse_cg"` solvers. [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") will begin checking the conditions shown in the following table from top to bottom. If the condition is true, the corresponding solver is chosen.

| **Solver** | **Condition** |
| --- | --- |
| ‘lbfgs’ | The `positive=True` option is specified. |
| ‘cholesky’ | The input array X is not sparse. |
| ‘sparse\_cg’ | None of the above conditions are fulfilled. |

Examples

- [Ordinary Least Squares and Ridge Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-py)
- [Plot Ridge coefficients as a function of the regularization](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py)
- [Common pitfalls in the interpretation of coefficients of linear models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)
- [Ridge coefficients as a function of the L2 Regularization](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html#sphx-glr-auto-examples-linear-model-plot-ridge-coeffs-py)

### 1.1.2.2. Classification

The [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge") regressor has a classifier variant: [`RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier "sklearn.linear_model.RidgeClassifier"). This classifier first converts binary targets to `{-1, 1}` and then treats the problem as a regression task, optimizing the same objective as above. The predicted class corresponds to the sign of the regressor’s prediction. For multiclass classification, the problem is treated as multi-output regression, and the predicted class corresponds to the output with the highest value.

It might seem questionable to use a (penalized) Least Squares loss to fit a classification model instead of the more traditional logistic or hinge losses. However, in practice, all those models can lead to similar cross-validation scores in terms of accuracy or precision/recall, while the penalized least squares loss used by the [`RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier "sklearn.linear_model.RidgeClassifier") allows for a very different choice of the numerical solvers with distinct computational performance profiles.

The [`RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier "sklearn.linear_model.RidgeClassifier") can be significantly faster than e.g. [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") with a high number of classes because it can compute the projection matrix $\left(\right. X^{T} X \left.\right)^{- 1} X^{T}$ only once.

This classifier is sometimes referred to as a [Least Squares Support Vector Machine](https://en.wikipedia.org/wiki/Least-squares_support-vector_machine) with a linear kernel.

Examples

- [Classification of text documents using sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)

### 1.1.2.3. Ridge Complexity

This method has the same order of complexity as.

### 1.1.2.4. Setting the regularization parameter: leave-one-out Cross-Validation

[`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV "sklearn.linear_model.RidgeCV") and [`RidgeClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV "sklearn.linear_model.RidgeClassifierCV") implement ridge regression/classification with built-in cross-validation of the alpha parameter. They work in the same way as [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV") except that it defaults to efficient Leave-One-Out [cross-validation](https://scikit-learn.org/stable/glossary.html#term-cross-validation). When using the default [cross-validation](https://scikit-learn.org/stable/glossary.html#term-cross-validation), alpha cannot be 0 due to the formulation used to calculate Leave-One-Out error. See for details.

Usage example:

```
>>> import numpy as np
>>> from sklearn import linear_model
>>> reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
>>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
RidgeCV(alphas=array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
      1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]))
>>> reg.alpha_
np.float64(0.01)
```

Specifying the value of the [cv](https://scikit-learn.org/stable/glossary.html#term-cv) attribute will trigger the use of cross-validation with [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV"), for example `cv=10` for 10-fold cross-validation, rather than Leave-One-Out Cross-Validation.

## 1.1.3. Lasso

The [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso "sklearn.linear_model.Lasso") is a linear model that estimates sparse coefficients, i.e., it is able to set coefficients exactly to zero. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. For this reason, Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero coefficients (see [Compressive sensing: tomography reconstruction with L1 prior (Lasso)](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)).

Mathematically, it consists of a linear model with an added regularization term. The objective function to minimize is:

$$
\underset{w}{min} P \left(\right. w \left.\right) = \frac{1}{2 n_{\text{samples}}} \left|\right. \left|\right. X w - y \left|\right. \left|\right._{2}^{2} + \alpha \left|\right. \left|\right. w \left|\right. \left|\right._{1}
$$

The lasso estimate thus solves the least-squares with added penalty $\alpha \left|\right. \left|\right. w \left|\right. \left|\right._{1}$, where $\alpha$ is a constant and $\left|\right. \left|\right. w \left|\right. \left|\right._{1}$ is the $ℓ_{1}$ -norm of the coefficient vector.

The implementation in the class [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso "sklearn.linear_model.Lasso") uses coordinate descent as the algorithm to fit the coefficients. See for another implementation:

```
>>> from sklearn import linear_model
>>> reg = linear_model.Lasso(alpha=0.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])
Lasso(alpha=0.1)
>>> reg.predict([[1, 1]])
array([0.8])
```

The function [`lasso_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path "sklearn.linear_model.lasso_path") is useful for lower-level tasks, as it computes the coefficients along the full path of possible values.

Examples

- [L1-based models for Sparse Signals](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
- [Compressive sensing: tomography reconstruction with L1 prior (Lasso)](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)
- [Common pitfalls in the interpretation of coefficients of linear models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)
- [Lasso model selection: AIC-BIC / cross-validation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py)

> [!note] Note
> **Feature selection with Lasso**
> 
> As the Lasso regression yields sparse models, it can thus be used to perform feature selection, as detailed in [L1-based feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection).

### 1.1.3.1. Coordinate Descent with Gap Safe Screening Rules

Coordinate descent (CD) is a strategy to solve a minimization problem that considers a single feature $j$ at a time. This way, the optimization problem is reduced to a 1-dimensional problem which is easier to solve:

$$
\underset{w_{j}}{min} \frac{1}{2 n_{\text{samples}}} \left|\right. \left|\right. x_{j} w_{j} + X_{- j} w_{- j} - y \left|\right. \left|\right._{2}^{2} + \alpha \left|\right. w_{j} \left|\right.
$$

with index $- j$ meaning all features but $j$. The solution is

$$
w_{j} = \frac{S \left(\right. x_{j}^{T} \left(\right. y - X_{- j} w_{- j} \left.\right) , \alpha \left.\right)}{\left|\right. \left|\right. x_{j} \left|\right. \left|\right._{2}^{2}}
$$

with the soft-thresholding function $S \left(\right. z , \alpha \left.\right) = sign \left(\right. z \left.\right) max \left(\right. 0 , \left|\right. z \left|\right. - \alpha \left.\right)$. Note that the soft-thresholding function is exactly zero whenever $\alpha \geq \left|\right. z \left|\right.$. The CD solver then loops over the features either in a cycle, picking one feature after the other in the order given by `X` (`selection="cyclic"`), or by randomly picking features (`selection="random"`). It stops if the duality gap is smaller than the provided tolerance `tol`.

A clever method to speedup the coordinate descent algorithm is to screen features such that at optimum $w_{j} = 0$. Gap safe screening rules are such a tool. Anywhere during the optimization algorithm, they can tell which feature we can safely exclude, i.e., set to zero with certainty.

### 1.1.3.2. Setting regularization parameter

The `alpha` parameter controls the degree of sparsity of the estimated coefficients.

#### 1.1.3.2.1. Using cross-validation

scikit-learn exposes objects that set the Lasso `alpha` parameter by cross-validation: [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV "sklearn.linear_model.LassoCV") and [`LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV "sklearn.linear_model.LassoLarsCV"). [`LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV "sklearn.linear_model.LassoLarsCV") is based on the algorithm explained below.

For high-dimensional datasets with many collinear features, [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV "sklearn.linear_model.LassoCV") is most often preferable. However, [`LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV "sklearn.linear_model.LassoLarsCV") has the advantage of exploring more relevant values of `alpha` parameter, and if the number of samples is very small compared to the number of features, it is often faster than [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV "sklearn.linear_model.LassoCV").

 **[![lasso_cv_1](https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_model_selection_002.png)](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html) [![lasso_cv_2](https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_model_selection_003.png)](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html)**

#### 1.1.3.2.2. Information-criteria based model selection

Alternatively, the estimator [`LassoLarsIC`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC "sklearn.linear_model.LassoLarsIC") proposes to use the Akaike information criterion (AIC) and the Bayes Information criterion (BIC). It is a computationally cheaper alternative to find the optimal value of alpha as the regularization path is computed only once instead of k+1 times when using k-fold cross-validation.

Indeed, these criteria are computed on the in-sample training set. In short, they penalize the over-optimistic scores of the different Lasso models by their flexibility (cf. to “Mathematical details” section below).

However, such criteria need a proper estimation of the degrees of freedom of the solution, are derived for large samples (asymptotic results) and assume the correct model is candidates under investigation. They also tend to break when the problem is badly conditioned (e.g. more features than samples).

![../_images/sphx_glr_plot_lasso_lars_ic_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_lars_ic_001.png)

Examples

- [Lasso model selection: AIC-BIC / cross-validation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py)
- [Lasso model selection via information criteria](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars_ic.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-ic-py)

#### 1.1.3.2.3. AIC and BIC criteria

The definition of AIC (and thus BIC) might differ in the literature. In this section, we give more information regarding the criterion computed in scikit-learn.

#### 1.1.3.2.4. Comparison with the regularization parameter of SVM

The equivalence between `alpha` and the regularization parameter of SVM, `C` is given by `alpha = 1 / C` or `alpha = 1 / (n_samples * C)`, depending on the estimator and the exact objective function optimized by the model.

## 1.1.4. Multi-task Lasso

The [`MultiTaskLasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso "sklearn.linear_model.MultiTaskLasso") is a linear model that estimates sparse coefficients for multiple regression problems jointly: `y` is a 2D array, of shape `(n_samples, n_tasks)`. The constraint is that the selected features are the same for all the regression problems, also called tasks.

The following figure compares the location of the non-zero entries in the coefficient matrix W obtained with a simple Lasso or a MultiTaskLasso. The Lasso estimates yield scattered non-zeros while the non-zeros of the MultiTaskLasso are full columns.

 **[![multi_task_lasso_1](https://scikit-learn.org/stable/_images/sphx_glr_plot_multi_task_lasso_support_001.png)](https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html) [![multi_task_lasso_2](https://scikit-learn.org/stable/_images/sphx_glr_plot_multi_task_lasso_support_002.png)](https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html)**

**Fitting a time-series model, imposing that any active feature be active at all times.**

Examples

- [Joint feature selection with multi-task Lasso](https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html#sphx-glr-auto-examples-linear-model-plot-multi-task-lasso-support-py)

## 1.1.5. Elastic-Net

[`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet "sklearn.linear_model.ElasticNet") is a linear regression model trained with both $ℓ_{1}$ and $ℓ_{2}$ -norm regularization of the coefficients. This combination allows for learning a sparse model where few of the weights are non-zero like [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso "sklearn.linear_model.Lasso"), while still maintaining the regularization properties of [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge"). We control the convex combination of $ℓ_{1}$ and $ℓ_{2}$ using the `l1_ratio` parameter.

Elastic-net is useful when there are multiple features that are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

A practical advantage of trading-off between Lasso and Ridge is that it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

The objective function to minimize is in this case

$$
\underset{w}{min} \frac{1}{2 n_{\text{samples}}} \left|\right. \left|\right. X w - y \left|\right. \left|\right._{2}^{2} + \alpha \rho \left|\right. \left|\right. w \left|\right. \left|\right._{1} + \frac{\alpha \left(\right. 1 - \rho \left.\right)}{2} \left|\right. \left|\right. w \left|\right. \left|\right._{2}^{2}
$$
![../_images/sphx_glr_plot_lasso_lasso_lars_elasticnet_path_002.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_lasso_lars_elasticnet_path_002.png)

The class [`ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV "sklearn.linear_model.ElasticNetCV") can be used to set the parameters `alpha` ($\alpha$) and `l1_ratio` ($\rho$) by cross-validation.

Examples

- [L1-based models for Sparse Signals](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
- [Lasso, Lasso-LARS, and Elastic Net paths](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lasso_lars_elasticnet_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-lasso-lars-elasticnet-path-py)
- [Fitting an Elastic Net with a precomputed Gram Matrix and Weighted Samples](https://scikit-learn.org/stable/auto_examples/linear_model/plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.html#sphx-glr-auto-examples-linear-model-plot-elastic-net-precomputed-gram-matrix-with-weighted-samples-py)

## 1.1.6. Multi-task Elastic-Net

The [`MultiTaskElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet "sklearn.linear_model.MultiTaskElasticNet") is an elastic-net model that estimates sparse coefficients for multiple regression problems jointly: `Y` is a 2D array of shape `(n_samples, n_tasks)`. The constraint is that the selected features are the same for all the regression problems, also called tasks.

Mathematically, it consists of a linear model trained with a mixed $ℓ_{1}$ $ℓ_{2}$ -norm and $ℓ_{2}$ -norm for regularization. The objective function to minimize is:

$$
\underset{W}{min} \frac{1}{2 n_{\text{samples}}} \left|\right. \left|\right. X W - Y \left|\right. \left|\right._{\text{Fro}}^{2} + \alpha \rho \left|\right. \left|\right. W \left|\right. \left|\right._{2 1} + \frac{\alpha \left(\right. 1 - \rho \left.\right)}{2} \left|\right. \left|\right. W \left|\right. \left|\right._{\text{Fro}}^{2}
$$

The implementation in the class [`MultiTaskElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet "sklearn.linear_model.MultiTaskElasticNet") uses coordinate descent as the algorithm to fit the coefficients.

The class [`MultiTaskElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV "sklearn.linear_model.MultiTaskElasticNetCV") can be used to set the parameters `alpha` ($\alpha$) and `l1_ratio` ($\rho$) by cross-validation.

## 1.1.7. Least Angle Regression

Least-angle regression (LARS) is a regression algorithm for high-dimensional data, developed by Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani. LARS is similar to forward stepwise regression. At each step, it finds the feature most correlated with the target. When there are multiple features having equal correlation, instead of continuing along the same feature, it proceeds in a direction equiangular between the features.

The advantages of LARS are:

- It is numerically efficient in contexts where the number of features is significantly greater than the number of samples.
- It is computationally just as fast as forward selection and has the same order of complexity as ordinary least squares.
- It produces a full piecewise linear solution path, which is useful in cross-validation or similar attempts to tune the model.
- If two features are almost equally correlated with the target, then their coefficients should increase at approximately the same rate. The algorithm thus behaves as intuition would expect, and also is more stable.
- It is easily modified to produce solutions for other estimators, like the Lasso.

The disadvantages of the LARS method include:

- Because LARS is based upon an iterative refitting of the residuals, it would appear to be especially sensitive to the effects of noise. This problem is discussed in detail by Weisberg in the discussion section of the Efron et al. (2004) Annals of Statistics article.

The LARS model can be used via the estimator [`Lars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars "sklearn.linear_model.Lars"), or its low-level implementation [`lars_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path "sklearn.linear_model.lars_path") or [`lars_path_gram`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html#sklearn.linear_model.lars_path_gram "sklearn.linear_model.lars_path_gram").

## 1.1.8. LARS Lasso

[`LassoLars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars "sklearn.linear_model.LassoLars") is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate descent, this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients.

![../_images/sphx_glr_plot_lasso_lasso_lars_elasticnet_path_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_lasso_lars_elasticnet_path_001.png)

```
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLars(alpha=.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])
LassoLars(alpha=0.1)
>>> reg.coef_
array([0.6, 0.        ])
```

Examples

- [Lasso, Lasso-LARS, and Elastic Net paths](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lasso_lars_elasticnet_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-lasso-lars-elasticnet-path-py)

The LARS algorithm provides the full path of the coefficients along the regularization parameter almost for free, thus a common operation is to retrieve the path with one of the functions [`lars_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path "sklearn.linear_model.lars_path") or [`lars_path_gram`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html#sklearn.linear_model.lars_path_gram "sklearn.linear_model.lars_path_gram").

## 1.1.9. Orthogonal Matching Pursuit (OMP)

[`OrthogonalMatchingPursuit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit "sklearn.linear_model.OrthogonalMatchingPursuit") and [`orthogonal_mp`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp "sklearn.linear_model.orthogonal_mp") implement the OMP algorithm for approximating the fit of a linear model with constraints imposed on the number of non-zero coefficients (i.e. the $ℓ_{0}$ pseudo-norm).

Being a forward feature selection method like, orthogonal matching pursuit can approximate the optimum solution vector with a fixed number of non-zero elements:

$$
\underset{w}{arg min} \left|\right. \left|\right. y - X w \left|\right. \left|\right._{2}^{2} \text{subject to } \left|\right. \left|\right. w \left|\right. \left|\right._{0} \leq n_{\text{nonzero}_\text{coefs}}
$$

Alternatively, orthogonal matching pursuit can target a specific error instead of a specific number of non-zero coefficients. This can be expressed as:

$$
\underset{w}{arg min} \left|\right. \left|\right. w \left|\right. \left|\right._{0} \text{subject to } \left|\right. \left|\right. y - X w \left|\right. \left|\right._{2}^{2} \leq \text{tol}
$$

OMP is based on a greedy algorithm that includes at each step the atom most highly correlated with the current residual. It is similar to the simpler matching pursuit (MP) method, but better in that at each iteration, the residual is recomputed using an orthogonal projection on the space of the previously chosen dictionary elements.

Examples

- [Orthogonal Matching Pursuit](https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py)

## 1.1.10. Bayesian Regression

Bayesian regression techniques can be used to include regularization parameters in the estimation procedure: the regularization parameter is not set in a hard sense but tuned to the data at hand.

This can be done by introducing [uninformative priors](https://en.wikipedia.org/wiki/Non-informative_prior#Uninformative_priors) over the hyper parameters of the model. The $ℓ_{2}$ regularization used in is equivalent to finding a maximum a posteriori estimation under a Gaussian prior over the coefficients $w$ with precision $\lambda^{- 1}$. Instead of setting `lambda` manually, it is possible to treat it as a random variable to be estimated from the data.

To obtain a fully probabilistic model, the output $y$ is assumed to be Gaussian distributed around $X w$:

$$
p \left(\right. y \left|\right. X , w , \alpha \left.\right) = \mathcal{N} \left(\right. y \left|\right. X w , \alpha^{- 1} \left.\right)
$$

where $\alpha$ is again treated as a random variable that is to be estimated from the data.

The advantages of Bayesian Regression are:

- It adapts to the data at hand.
- It can be used to include regularization parameters in the estimation procedure.

The disadvantages of Bayesian regression include:

- Inference of the model can be time consuming.

### 1.1.10.1. Bayesian Ridge Regression

[`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge "sklearn.linear_model.BayesianRidge") estimates a probabilistic model of the regression problem as described above. The prior for the coefficient $w$ is given by a spherical Gaussian:

$$
p \left(\right. w \left|\right. \lambda \left.\right) = \mathcal{N} \left(\right. w \left|\right. 0 , \lambda^{- 1} \mathbf{I}_{p} \left.\right)
$$

The priors over $\alpha$ and $\lambda$ are chosen to be [gamma distributions](https://en.wikipedia.org/wiki/Gamma_distribution), the conjugate prior for the precision of the Gaussian. The resulting model is called *Bayesian Ridge Regression*, and is similar to the classical [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge").

The parameters $w$, $\alpha$ and $\lambda$ are estimated jointly during the fit of the model, the regularization parameters $\alpha$ and $\lambda$ being estimated by maximizing the *log marginal likelihood*. The scikit-learn implementation is based on the algorithm described in Appendix A of (Tipping, 2001) where the update of the parameters $\alpha$ and $\lambda$ is done as suggested in (MacKay, 1992). The initial value of the maximization procedure can be set with the hyperparameters `alpha_init` and `lambda_init`.

There are four more hyperparameters, $\alpha_{1}$, $\alpha_{2}$, $\lambda_{1}$ and $\lambda_{2}$ of the gamma prior distributions over $\alpha$ and $\lambda$. These are usually chosen to be *non-informative*. By default $\alpha_{1} = \alpha_{2} = \lambda_{1} = \lambda_{2} = 10^{- 6}$.

Bayesian Ridge Regression is used for regression:

```
>>> from sklearn import linear_model
>>> X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
>>> Y = [0., 1., 2., 3.]
>>> reg = linear_model.BayesianRidge()
>>> reg.fit(X, Y)
BayesianRidge()
```

After being fitted, the model can then be used to predict new values:

```
>>> reg.predict([[1, 0.]])
array([0.50000013])
```

The coefficients $w$ of the model can be accessed:

```
>>> reg.coef_
array([0.49999993, 0.49999993])
```

Due to the Bayesian framework, the weights found are slightly different from the ones found by. However, Bayesian Ridge Regression is more robust to ill-posed problems.

Examples

- [Curve Fitting with Bayesian Ridge Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-curvefit-py)

### 1.1.10.2. Automatic Relevance Determination - ARD

The Automatic Relevance Determination (as being implemented in [`ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression "sklearn.linear_model.ARDRegression")) is a kind of linear model which is very similar to the, but that leads to sparser coefficients $w$.

[`ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression "sklearn.linear_model.ARDRegression") poses a different prior over $w$: it drops the spherical Gaussian distribution for a centered elliptic Gaussian distribution. This means each coefficient $w_{i}$ can itself be drawn from a Gaussian distribution, centered on zero and with a precision $\lambda_{i}$:

$$
p \left(\right. w \left|\right. \lambda \left.\right) = \mathcal{N} \left(\right. w \left|\right. 0 , A^{- 1} \left.\right)
$$

with $A$ being a positive definite diagonal matrix and $\text{diag} \left(\right. A \left.\right) = \lambda = \left{\right. \lambda_{1} , . . . , \lambda_{p} \left.\right}$.

In contrast to the, each coordinate of $w_{i}$ has its own standard deviation $\frac{1}{\lambda_{i}}$. The prior over all $\lambda_{i}$ is chosen to be the same gamma distribution given by the hyperparameters $\lambda_{1}$ and $\lambda_{2}$.

ARD is also known in the literature as *Sparse Bayesian Learning* and *Relevance Vector Machine*.

See [Comparing Linear Bayesian Regressors](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py) for a worked-out comparison between ARD and.

See [L1-based models for Sparse Signals](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py) for a comparison between various methods - Lasso, ARD and ElasticNet - on correlated data.

References

## 1.1.11. Logistic regression

The logistic regression is implemented in [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression"). Despite its name, it is implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature. The logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

This implementation can fit binary, One-vs-Rest, or multinomial logistic regression with optional $ℓ_{1}$, $ℓ_{2}$ or Elastic-Net regularization.

> [!note] Note
> **Regularization**
> 
> Regularization is applied by default, which is common in machine learning but not in statistics. Another advantage of regularization is that it improves numerical stability. No regularization amounts to setting C to a very high value.

> [!note] Note
> **Logistic Regression as a special case of the Generalized Linear Models (GLM)**
> 
> Logistic regression is a special case of with a Binomial / Bernoulli conditional distribution and a Logit link. The numerical output of the logistic regression, which is the predicted probability, can be used as a classifier by applying a threshold (by default 0.5) to it. This is how it is implemented in scikit-learn, so it expects a categorical target, making the Logistic Regression a classifier.

Examples

- [L1 Penalty and Sparsity in Logistic Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py)
- [Regularization path of L1- Logistic Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py)
- [Decision Boundaries of Multinomial and One-vs-Rest Logistic Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py)
- [Multiclass sparse logistic regression on 20newgroups](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py)
- [MNIST classification using multinomial logistic + L1](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py)
- [Plot classification probability](https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py)

### 1.1.11.1. Binary Case

For notational ease, we assume that the target $y_{i}$ takes values in the set $\left{\right. 0 , 1 \left.\right}$ for data point $i$. Once fitted, the [`predict_proba`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba "sklearn.linear_model.LogisticRegression.predict_proba") method of [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") predicts the probability of the positive class $P \left(\right. y_{i} = 1 \left|\right. X_{i} \left.\right)$ as

$$
\hat{p} \left(\right. X_{i} \left.\right) = expit \left(\right. X_{i} w + w_{0} \left.\right) = \frac{1}{1 + exp ⁡ \left(\right. - X_{i} w - w_{0} \left.\right)} .
$$

As an optimization problem, binary class logistic regression with regularization term $r \left(\right. w \left.\right)$ minimizes the following cost function:

(1) [#](#regularized-logistic-loss "Link to this equation") 
$$
\underset{w}{min} \frac{1}{S} \sum_{i = 1}^{n} s_{i} \left(\right. - y_{i} log ⁡ \left(\right. \hat{p} \left(\right. X_{i} \left.\right) \left.\right) - \left(\right. 1 - y_{i} \left.\right) log ⁡ \left(\right. 1 - \hat{p} \left(\right. X_{i} \left.\right) \left.\right) \left.\right) + \frac{r \left(\right. w \left.\right)}{S C} ,
$$

where $s_{i}$ corresponds to the weights assigned by the user to a specific training sample (the vector $s$ is formed by element-wise multiplication of the class weights and sample weights), and the sum $S = \sum_{i = 1}^{n} s_{i}$.

We currently provide four choices for the regularization or penalty term $r \left(\right. w \left.\right)$ via the arguments `C` and `l1_ratio`:

| penalty | $r \left(\right. w \left.\right)$ |
| --- | --- |
| none (`C=np.inf`) | $0$ |
| $ℓ_{1}$ (`l1_ratio=1`) | $\parallel w \parallel_{1}$ |
| $ℓ_{2}$ (`l1_ratio=0`) | $\frac{1}{2} \parallel w \parallel_{2}^{2} = \frac{1}{2} w^{T} w$ |
| ElasticNet (`0<l1_ratio<1`) | $\frac{1 - \rho}{2} w^{T} w + \rho \parallel w \parallel_{1}$ |

For ElasticNet, $\rho$ (which corresponds to the `l1_ratio` parameter) controls the strength of $ℓ_{1}$ regularization vs. $ℓ_{2}$ regularization. Elastic-Net is equivalent to $ℓ_{1}$ when $\rho = 1$ and equivalent to $ℓ_{2}$ when $\rho = 0$.

Note that the scale of the class weights and the sample weights will influence the optimization problem. For instance, multiplying the sample weights by a constant $b > 0$ is equivalent to multiplying the (inverse) regularization strength `C` by $b$.

### 1.1.11.2. Multinomial Case

The binary case can be extended to $K$ classes leading to the multinomial logistic regression, see also [log-linear model](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model).

> [!note] Note
> It is possible to parameterize a $K$ -class classification model using only $K - 1$ weight vectors, leaving one class probability fully determined by the other class probabilities by leveraging the fact that all class probabilities must sum to one. We deliberately choose to overparameterize the model using $K$ weight vectors for ease of implementation and to preserve the symmetrical inductive bias regarding ordering of classes, see. This effect becomes especially important when using regularization. The choice of overparameterization can be detrimental for unpenalized models since then the solution may not be unique, as shown in.

### 1.1.11.3. Solvers

The solvers implemented in the class [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") are “lbfgs”, “liblinear”, “newton-cg”, “newton-cholesky”, “sag” and “saga”:

The following table summarizes the penalties and multinomial multiclass supported by each solver:

<table><tbody><tr><td></td><td colspan="7"><p><strong>Solvers</strong></p></td></tr><tr><td><p><strong>Penalties</strong></p></td><td><p><strong>‘lbfgs’</strong></p></td><td colspan="2"><p><strong>‘liblinear’</strong></p></td><td><p><strong>‘newton-cg’</strong></p></td><td><p><strong>‘newton-cholesky’</strong></p></td><td><p><strong>‘sag’</strong></p></td><td><p><strong>‘saga’</strong></p></td></tr><tr><td><p>L2 penalty</p></td><td><p>yes</p></td><td colspan="2"><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td></tr><tr><td><p>L1 penalty</p></td><td><p>no</p></td><td colspan="2"><p>yes</p></td><td><p>no</p></td><td><p>no</p></td><td><p>no</p></td><td><p>yes</p></td></tr><tr><td><p>Elastic-Net (L1 + L2)</p></td><td><p>no</p></td><td colspan="2"><p>no</p></td><td><p>no</p></td><td><p>no</p></td><td><p>no</p></td><td><p>yes</p></td></tr><tr><td><p>No penalty</p></td><td><p>yes</p></td><td colspan="2"><p>no</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td></tr><tr><td><p><strong>Multiclass support</strong></p></td><td colspan="7"></td></tr><tr><td><p>multinomial multiclass</p></td><td><p>yes</p></td><td colspan="2"><p>no</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td></tr><tr><td><p><strong>Behaviors</strong></p></td><td colspan="7"></td></tr><tr><td><p>Penalize the intercept (bad)</p></td><td><p>no</p></td><td colspan="2"><p>yes</p></td><td><p>no</p></td><td><p>no</p></td><td><p>no</p></td><td><p>no</p></td></tr><tr><td><p>Faster for large datasets</p></td><td><p>no</p></td><td colspan="2"><p>no</p></td><td><p>no</p></td><td><p>no</p></td><td><p>yes</p></td><td><p>yes</p></td></tr><tr><td><p>Robust to unscaled datasets</p></td><td><p>yes</p></td><td colspan="2"><p>yes</p></td><td><p>yes</p></td><td><p>yes</p></td><td><p>no</p></td><td><p>no</p></td></tr></tbody></table>

The “lbfgs” solver is used by default for its robustness. For `n_samples >> n_features`, “newton-cholesky” is a good choice and can reach high precision (tiny `tol` values). For large datasets the “saga” solver is usually faster (than “lbfgs”), in particular for low precision (high `tol`). For large dataset, you may also consider using [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier") with `loss="log_loss"`, which might be even faster but requires more tuning.

#### 1.1.11.3.1. Differences between solvers

There might be a difference in the scores obtained between [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") with `solver=liblinear` or [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC") and the external liblinear library directly, when `fit_intercept=False` and the fit `coef_` (or) the data to be predicted are zeroes. This is because for the sample(s) with `decision_function` zero, [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression") and [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC") predict the negative class, while liblinear predicts the positive class. Note that a model with `fit_intercept=False` and having many samples with `decision_function` zero, is likely to be an underfit, bad model and you are advised to set `fit_intercept=True` and increase the `intercept_scaling`.

> [!note] Note
> **Feature selection with sparse logistic regression**
> 
> A logistic regression with $ℓ_{1}$ penalty yields sparse models, and can thus be used to perform feature selection, as detailed in [L1-based feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection).

> [!note] Note
> **P-value estimation**
> 
> It is possible to obtain the p-values and confidence intervals for coefficients in cases of regression without penalization. The [statsmodels package](https://pypi.org/project/statsmodels/) natively supports this. Within sklearn, one could use bootstrapping instead as well.

[`LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV "sklearn.linear_model.LogisticRegressionCV") implements Logistic Regression with built-in cross-validation support, to find the optimal `C` and `l1_ratio` parameters according to the `scoring` attribute. The “newton-cg”, “sag”, “saga” and “lbfgs” solvers are found to be faster for high-dimensional dense data, due to warm-starting (see [Glossary](https://scikit-learn.org/stable/glossary.html#term-warm_start)).

## 1.1.12. Generalized Linear Models

Generalized Linear Models (GLM) extend linear models in two ways. First, the predicted values $\hat{y}$ are linked to a linear combination of the input variables $X$ via an inverse link function $h$ as

$$
\hat{y} \left(\right. w , X \left.\right) = h \left(\right. X w \left.\right) .
$$

Secondly, the squared loss function is replaced by the unit deviance $d$ of a distribution in the exponential family (or more precisely, a reproductive exponential dispersion model (EDM) ).

The minimization problem becomes:

$$
\underset{w}{min} \frac{1}{2 n_{\text{samples}}} \underset{i}{\sum} d \left(\right. y_{i} , \hat{y}_{i} \left.\right) + \frac{\alpha}{2} \left|\right. \left|\right. w \left|\right. \left|\right._{2}^{2} ,
$$

where $\alpha$ is the L2 regularization penalty. When sample weights are provided, the average becomes a weighted average.

The following table lists some specific EDMs and their unit deviance:

| Distribution | Target Domain | Unit Deviance $d \left(\right. y , \hat{y} \left.\right)$ |
| --- | --- | --- |
| Normal | $y \in \left(\right. - \infty , \infty \left.\right)$ | $\left(\right. y - \hat{y} \left.\right)^{2}$ |
| Bernoulli | $y \in \left{\right. 0 , 1 \left.\right}$ | $2 \left(\right. y log ⁡ \frac{y}{\hat{y}} + \left(\right. 1 - y \left.\right) log ⁡ \frac{1 - y}{1 - \hat{y}} \left.\right)$ |
| Categorical | $y \in \left{\right. 0 , 1 , . . . , k \left.\right}$ | $2 \underset{i \in \left{\right. 0 , 1 , . . . , k \left.\right}}{\sum} I \left(\right. y = i \left.\right) y_{\text{i}} log ⁡ \frac{I \left(\right. y = i \left.\right)}{\hat{I \left(\right. y = i \left.\right)}}$ |
| Poisson | $y \in \left[\right. 0 , \infty \left.\right)$ | $2 \left(\right. y log ⁡ \frac{y}{\hat{y}} - y + \hat{y} \left.\right)$ |
| Gamma | $y \in \left(\right. 0 , \infty \left.\right)$ | $2 \left(\right. log ⁡ \frac{\hat{y}}{y} + \frac{y}{\hat{y}} - 1 \left.\right)$ |
| Inverse Gaussian | $y \in \left(\right. 0 , \infty \left.\right)$ | $\frac{\left(\right. y - \hat{y} \left.\right)^{2}}{y \hat{y}^{2}}$ |

The Probability Density Functions (PDF) of these distributions are illustrated in the following figure,

![../_images/poisson_gamma_tweedie_distributions.png](https://scikit-learn.org/stable/_images/poisson_gamma_tweedie_distributions.png)

PDF of a random variable Y following Poisson, Tweedie (power=1.5) and Gamma distributions with different mean values ( μ ). Observe the point mass at Y = 0 for the Poisson distribution and the Tweedie (power=1.5) distribution, but not for the Gamma distribution which has a strictly positive target domain. #

The Bernoulli distribution is a discrete probability distribution modelling a Bernoulli trial - an event that has only two mutually exclusive outcomes. The Categorical distribution is a generalization of the Bernoulli distribution for a categorical random variable. While a random variable in a Bernoulli distribution has two possible outcomes, a Categorical random variable can take on one of K possible categories, with the probability of each category specified separately.

The choice of the distribution depends on the problem at hand:

- If the target values $y$ are counts (non-negative integer valued) or relative frequencies (non-negative), you might use a Poisson distribution with a log-link.
- If the target values are positive valued and skewed, you might try a Gamma distribution with a log-link.
- If the target values seem to be heavier tailed than a Gamma distribution, you might try an Inverse Gaussian distribution (or even higher variance powers of the Tweedie family).
- If the target values $y$ are probabilities, you can use the Bernoulli distribution. The Bernoulli distribution with a logit link can be used for binary classification. The Categorical distribution with a softmax link can be used for multiclass classification.

References

### 1.1.12.1. Usage

[`TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor "sklearn.linear_model.TweedieRegressor") implements a generalized linear model for the Tweedie distribution, that allows to model any of the above mentioned distributions using the appropriate `power` parameter. In particular:

- `power = 0`: Normal distribution. Specific estimators such as [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge"), [`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet "sklearn.linear_model.ElasticNet") are generally more appropriate in this case.
- `power = 1`: Poisson distribution. [`PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor "sklearn.linear_model.PoissonRegressor") is exposed for convenience. However, it is strictly equivalent to `TweedieRegressor(power=1, link='log')`.
- `power = 2`: Gamma distribution. [`GammaRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor "sklearn.linear_model.GammaRegressor") is exposed for convenience. However, it is strictly equivalent to `TweedieRegressor(power=2, link='log')`.
- `power = 3`: Inverse Gaussian distribution.

The link function is determined by the `link` parameter.

Usage example:

```
>>> from sklearn.linear_model import TweedieRegressor
>>> reg = TweedieRegressor(power=1, alpha=0.5, link='log')
>>> reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2])
TweedieRegressor(alpha=0.5, link='log', power=1)
>>> reg.coef_
array([0.2463, 0.4337])
>>> reg.intercept_
np.float64(-0.7638)
```

Examples

- [Poisson regression and non-normal loss](https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-auto-examples-linear-model-plot-poisson-regression-non-normal-loss-py)
- [Tweedie regression on insurance claims](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#sphx-glr-auto-examples-linear-model-plot-tweedie-regression-insurance-claims-py)

## 1.1.13. Stochastic Gradient Descent - SGD

Stochastic gradient descent is a simple yet very efficient approach to fit linear models. It is particularly useful when the number of samples (and the number of features) is very large. The `partial_fit` method allows online/out-of-core learning.

The classes [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier") and [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor "sklearn.linear_model.SGDRegressor") provide functionality to fit linear models for classification and regression using different (convex) loss functions and different penalties. E.g., with `loss="log"`, [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier") fits a logistic regression model, while with `loss="hinge"` it fits a linear support vector machine (SVM).

You can refer to the dedicated [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html#sgd) documentation section for more details.

### 1.1.13.1. Perceptron

The [`Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron "sklearn.linear_model.Perceptron") is another simple classification algorithm suitable for large scale learning and derives from SGD. By default:

- It does not require a learning rate.
- It is not regularized (penalized).
- It updates its model only on mistakes.

The last characteristic implies that the Perceptron is slightly faster to train than SGD with the hinge loss and that the resulting models are sparser.

In fact, the [`Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron "sklearn.linear_model.Perceptron") is a wrapper around the [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier") class using a perceptron loss and a constant learning rate. Refer to [mathematical section](https://scikit-learn.org/stable/modules/sgd.html#sgd-mathematical-formulation) of the SGD procedure for more details.

### 1.1.13.2. Passive Aggressive Algorithms

The passive-aggressive (PA) algorithms are another family of 2 algorithms (PA-I and PA-II) for large-scale online learning that derive from SGD. They are similar to the Perceptron in that they do not require a learning rate. However, contrary to the Perceptron, they include a regularization parameter `eta0` ($C$ in the reference paper).

For classification, `SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0)` can be used for PA-I or with `learning_rate="pa2"` for PA-II. For regression, `SGDRegressor(loss="epsilon_insensitive", penalty=None, learning_rate="pa1", eta0=1.0)` can be used for PA-I or with `learning_rate="pa2"` for PA-II.

## 1.1.15. Quantile Regression

Quantile regression estimates the median or other quantiles of $y$ conditional on $X$, while ordinary least squares (OLS) estimates the conditional mean.

Quantile regression may be useful if one is interested in predicting an interval instead of point prediction. Sometimes, prediction intervals are calculated based on the assumption that prediction error is distributed normally with zero mean and constant variance. Quantile regression provides sensible prediction intervals even for errors with non-constant (but predictable) variance or non-normal distribution.

![../_images/sphx_glr_plot_quantile_regression_002.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_quantile_regression_002.png)

Based on minimizing the pinball loss, conditional quantiles can also be estimated by models other than linear models. For example, [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor "sklearn.ensemble.GradientBoostingRegressor") can predict conditional quantiles if its parameter `loss` is set to `"quantile"` and parameter `alpha` is set to the quantile that should be predicted. See the example in [Prediction Intervals for Gradient Boosting Regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-quantile-py).

Most implementations of quantile regression are based on linear programming problem. The current implementation is based on [`scipy.optimize.linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog "(in SciPy v1.17.0)").

Examples

- [Quantile regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html#sphx-glr-auto-examples-linear-model-plot-quantile-regression-py)

## 1.1.16. Polynomial regression: extending linear models with basis functions

One common pattern within machine learning is to use linear models trained on nonlinear functions of the data. This approach maintains the generally fast performance of linear methods, while allowing them to fit a much wider range of data.

Here is an example of applying this idea to one-dimensional data, using polynomial features of varying degrees:

![../_images/sphx_glr_plot_polynomial_interpolation_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_polynomial_interpolation_001.png)

This figure is created using the [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures "sklearn.preprocessing.PolynomialFeatures") transformer, which transforms an input data matrix into a new data matrix of a given degree. It can be used as follows:

```
>>> from sklearn.preprocessing import PolynomialFeatures
>>> import numpy as np
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(degree=2)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
```

The features of `X` have been transformed from $\left[\right. x_{1} , x_{2} \left]\right.$ to $\left[\right. 1 , x_{1} , x_{2} , x_{1}^{2} , x_{1} x_{2} , x_{2}^{2} \left]\right.$, and can now be used within any linear model.

This sort of preprocessing can be streamlined with the [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) tools. A single object representing a simple polynomial regression can be created and used as follows:

```
>>> from sklearn.preprocessing import PolynomialFeatures
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.pipeline import Pipeline
>>> import numpy as np
>>> model = Pipeline([('poly', PolynomialFeatures(degree=3)),
...                   ('linear', LinearRegression(fit_intercept=False))])
>>> # fit to an order-3 polynomial data
>>> x = np.arange(5)
>>> y = 3 - 2 * x + x ** 2 - x ** 3
>>> model = model.fit(x[:, np.newaxis], y)
>>> model.named_steps['linear'].coef_
array([ 3., -2.,  1., -1.])
```

The linear model trained on polynomial features is able to exactly recover the input polynomial coefficients.

In some cases it’s not necessary to include higher powers of any single feature, but only the so-called *interaction features* that multiply together at most $d$ distinct features. These can be gotten from [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures "sklearn.preprocessing.PolynomialFeatures") with the setting `interaction_only=True`.

For example, when dealing with boolean features, $x_{i}^{n} = x_{i}$ for all $n$ and is therefore useless; but $x_{i} x_{j}$ represents the conjunction of two booleans. This way, we can solve the XOR problem with a linear classifier:

```
>>> from sklearn.linear_model import Perceptron
>>> from sklearn.preprocessing import PolynomialFeatures
>>> import numpy as np
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
>>> y = X[:, 0] ^ X[:, 1]
>>> y
array([0, 1, 1, 0])
>>> X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
>>> X
array([[1, 0, 0, 0],
       [1, 0, 1, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 1]])
>>> clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,
...                  shuffle=False).fit(X, y)
```

And the classifier “predictions” are perfect:

```
>>> clf.predict(X)
array([0, 1, 1, 0])
>>> clf.score(X, y)
1.0
```

[^1]: $$
A I C = - 2 log ⁡ \left(\right. \hat{L} \left.\right) + 2 d
$$
 
$$
B I C = - 2 log ⁡ \left(\right. \hat{L} \left.\right) + log ⁡ \left(\right. N \left.\right) d
$$
 
$$
log ⁡ \left(\right. \hat{L} \left.\right) = - \frac{n}{2} log ⁡ \left(\right. 2 \pi \left.\right) - \frac{n}{2} log ⁡ \left(\right. \sigma^{2} \left.\right) - \frac{\sum_{i = 1}^{n} \left(\right. y_{i} - \hat{y}_{i} \left.\right)^{2}}{2 \sigma^{2}}
$$
 
$$
A I C = n log ⁡ \left(\right. 2 \pi \sigma^{2} \left.\right) + \frac{\sum_{i = 1}^{n} \left(\right. y_{i} - \hat{y}_{i} \left.\right)^{2}}{\sigma^{2}} + 2 d
$$
 
$$
\sigma^{2} = \frac{\sum_{i = 1}^{n} \left(\right. y_{i} - \hat{y}_{i} \left.\right)^{2}}{n - p}
$$

References

[^2]: At last, we mentioned above that $\sigma^{2}$ is an estimate of the noise variance. In [`LassoLarsIC`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC "sklearn.linear_model.LassoLarsIC") when the parameter `noise_variance` is not provided (default), the noise variance is estimated via the unbiased estimator defined as:

$$
\sigma^{2} = \frac{\sum_{i = 1}^{n} \left(\right. y_{i} - \hat{y}_{i} \left.\right)^{2}}{n - p}
$$

where $p$ is the number of features and $\hat{y}_{i}$ is the predicted target using an ordinary least squares regression. Note, that this formula is valid only when `n_samples > n_features`

References

[^3]: The Automatic Relevance Determination (as being implemented in [`ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression "sklearn.linear_model.ARDRegression")) is a kind of linear model which is very similar to the, but that leads to sparser coefficients $w$

[`ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression "sklearn.linear_model.ARDRegression") poses a different prior over $w$: it drops the spherical Gaussian distribution for a centered elliptic Gaussian distribution. This means each coefficient $w_{i}$ can itself be drawn from a Gaussian distribution, centered on zero and with a precision $\lambda_{i}$:

$$
p \left(\right. w \left|\right. \lambda \left.\right) = \mathcal{N} \left(\right. w \left|\right. 0 , A^{- 1} \left.\right)
$$

with $A$ being a positive definite diagonal matrix and $\text{diag} \left(\right. A \left.\right) = \lambda = \left{\right. \lambda_{1} , . . . , \lambda_{p} \left.\right}$

In contrast to the, each coordinate of $w_{i}$ has its own standard deviation $\frac{1}{\lambda_{i}}$. The prior over all $\lambda_{i}$ is chosen to be the same gamma distribution given by the hyperparameters $\lambda_{1}$ and $\lambda_{2}$

[^4]: ARD is also known in the literature as *Sparse Bayesian Learning* and *Relevance Vector Machine*

See [Comparing Linear Bayesian Regressors](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py) for a worked-out comparison between ARD and

See [L1-based models for Sparse Signals](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py) for a comparison between various methods - Lasso, ARD and ElasticNet - on correlated data.

References

[^5]: > [!note] Note
> It is possible to parameterize a $K$ -class classification model using only $K - 1$ weight vectors, leaving one class probability fully determined by the other class probabilities by leveraging the fact that all class probabilities must sum to one. We deliberately choose to overparameterize the model using $K$ weight vectors for ease of implementation and to preserve the symmetrical inductive bias regarding ordering of classes, see. This effect becomes especially important when using regularization. The choice of overparameterization can be detrimental for unpenalized models since then the solution may not be unique, as shown in.

[^6]: References

[^7]: The “sag” solver uses Stochastic Average Gradient descent. It is faster than other solvers for large datasets, when both the number of samples and the number of features are large.

[^8]: The “saga” solver is a variant of “sag” that also supports the non-smooth $ℓ_{1}$ penalty (`l1_ratio=1`). This is therefore the solver of choice for sparse multinomial logistic regression. It is also the only solver that supports Elastic-Net (`0 < l1_ratio < 1`).

[^9]: The “lbfgs” is an optimization algorithm that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm, which belongs to quasi-Newton methods. As such, it can deal with a wide range of different training data and is therefore the default solver. Its performance, however, suffers on poorly scaled datasets and on datasets with one-hot encoded categorical features with rare categories.

[^10]: For a comparison of some of these solvers, see

References

[^11]: Generalized Linear Models (GLM) extend linear models in two ways. First, the predicted values $\hat{y}$ are linked to a linear combination of the input variables $X$ via an inverse link function $h$ as 

$$
\hat{y} \left(\right. w , X \left.\right) = h \left(\right. X w \left.\right) .
$$

[^12]: Secondly, the squared loss function is replaced by the unit deviance $d$ of a distribution in the exponential family (or more precisely, a reproductive exponential dispersion model (EDM) ).

The minimization problem becomes:

$$
\underset{w}{min} \frac{1}{2 n_{\text{samples}}} \underset{i}{\sum} d \left(\right. y_{i} , \hat{y}_{i} \left.\right) + \frac{\alpha}{2} \left|\right. \left|\right. w \left|\right. \left|\right._{2}^{2} ,
$$

where $\alpha$ is the L2 regularization penalty. When sample weights are provided, the average becomes a weighted average.

The following table lists some specific EDMs and their unit deviance:

| Distribution | Target Domain | Unit Deviance $d \left(\right. y , \hat{y} \left.\right)$ |
| --- | --- | --- |
| Normal | $y \in \left(\right. - \infty , \infty \left.\right)$ | $\left(\right. y - \hat{y} \left.\right)^{2}$ |
| Bernoulli | $y \in \left{\right. 0 , 1 \left.\right}$ | $2 \left(\right. y log ⁡ \frac{y}{\hat{y}} + \left(\right. 1 - y \left.\right) log ⁡ \frac{1 - y}{1 - \hat{y}} \left.\right)$ |
| Categorical | $y \in \left{\right. 0 , 1 , . . . , k \left.\right}$ | $2 \underset{i \in \left{\right. 0 , 1 , . . . , k \left.\right}}{\sum} I \left(\right. y = i \left.\right) y_{\text{i}} log ⁡ \frac{I \left(\right. y = i \left.\right)}{\hat{I \left(\right. y = i \left.\right)}}$ |
| Poisson | $y \in \left[\right. 0 , \infty \left.\right)$ | $2 \left(\right. y log ⁡ \frac{y}{\hat{y}} - y + \hat{y} \left.\right)$ |
| Gamma | $y \in \left(\right. 0 , \infty \left.\right)$ | $2 \left(\right. log ⁡ \frac{\hat{y}}{y} + \frac{y}{\hat{y}} - 1 \left.\right)$ |
| Inverse Gaussian | $y \in \left(\right. 0 , \infty \left.\right)$ | $\frac{\left(\right. y - \hat{y} \left.\right)^{2}}{y \hat{y}^{2}}$ |

The Probability Density Functions (PDF) of these distributions are illustrated in the following figure,

[^13]: ![../_images/sphx_glr_plot_theilsen_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_theilsen_001.png)

$$
\left(\right. \frac{n_{\text{samples}}}{n_{\text{subsamples}}} \left.\right)
$$

References

[^20]: Kärkkäinen and S. Äyrämö: [On Computation of Spatial Median for Robust Data Mining.](http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf)