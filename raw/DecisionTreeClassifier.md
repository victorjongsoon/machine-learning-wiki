---
title: "DecisionTreeClassifier"
source: "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
author:
published:
created: 2026-05-01
description: "Gallery examples: Classifier comparison Multi-class AdaBoosted Decision Trees Two-class AdaBoost Plot the decision surfaces of ensembles of trees on the iris dataset Demonstration of multi-metric e..."
tags:
  - "raw"
---
*class* sklearn.tree.DecisionTreeClassifier(*\**, *criterion='gini'*, *splitter='best'*, *max\_depth=None*, *min\_samples\_split=2*, *min\_samples\_leaf=1*, *min\_weight\_fraction\_leaf=0.0*, *max\_features=None*, *random\_state=None*, *max\_leaf\_nodes=None*, *min\_impurity\_decrease=0.0*, *class\_weight=None*, *ccp\_alpha=0.0*, *monotonic\_cst=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L707) [#](#sklearn.tree.DecisionTreeClassifier "Link to this definition")

A decision tree classifier.

Read more in the [User Guide](https://scikit-learn.org/stable/modules/tree.html#tree).

Parameters:

**criterion** {“gini”, “entropy”, “log\_loss”}, default=”gini”

The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log\_loss” and “entropy” both for the Shannon information gain, see [Mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#tree-mathematical-formulation).

**splitter** {“best”, “random”}, default=”best”

The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

**max\_depth** int, default=None

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min\_samples\_split samples.

**min\_samples\_split** int or float, default=2

The minimum number of samples required to split an internal node:

- If int, then consider `min_samples_split` as the minimum number.
- If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.

Changed in version 0.18: Added float values for fractions.

**min\_samples\_leaf** int or float, default=1

The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

- If int, then consider `min_samples_leaf` as the minimum number.
- If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

Changed in version 0.18: Added float values for fractions.

**min\_weight\_fraction\_leaf** float, default=0.0

The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample\_weight is not provided.

**max\_features** int, float or {“sqrt”, “log2”}, default=None

The number of features to consider when looking for the best split:

- If int, then consider `max_features` features at each split.
- If float, then `max_features` is a fraction and `max(1, int(max_features * n_features_in_))` features are considered at each split.
- If “sqrt”, then `max_features=sqrt(n_features)`.
- If “log2”, then `max_features=log2(n_features)`.
- If None, then `max_features=n_features`.

> [!note] Note
> The search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than `max_features` features.

**random\_state** int, RandomState instance or None, default=None

Controls the randomness of the estimator. The features are always randomly permuted at each split, even if `splitter` is set to `"best"`. When `max_features < n_features`, the algorithm will select `max_features` at random at each split before finding the best split among them. But the best found split may vary across different runs, even if `max_features=n_features`. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random. To obtain a deterministic behaviour during fitting, `random_state` has to be fixed to an integer. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state) for details.

**max\_leaf\_nodes** int, default=None

Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

**min\_impurity\_decrease** float, default=0.0

A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

The weighted impurity decrease equation is the following:

```
N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
```

where `N` is the total number of samples, `N_t` is the number of samples at the current node, `N_t_L` is the number of samples in the left child, and `N_t_R` is the number of samples in the right child.

`N`, `N_t`, `N_t_R` and `N_t_L` all refer to the weighted sum, if `sample_weight` is passed.

Added in version 0.19.

**class\_weight** dict, list of dict or “balanced”, default=None

Weights associated with classes in the form `{class_label: weight}`. If None, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be \[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}\] instead of \[{1:1}, {2:5}, {3:1}, {4:1}\].

The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`

For multi-output, the weights of each column of y will be multiplied.

Note that these weights will be multiplied with sample\_weight (passed through the fit method) if sample\_weight is specified.

**ccp\_alpha** non-negative float, default=0.0

Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccp_alpha` will be chosen. By default, no pruning is performed. See [Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) for details. See [Post pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py) for an example of such pruning.

Added in version 0.22.

**monotonic\_cst** array-like of int of shape (n\_features), default=None

Indicates the monotonicity constraint to enforce on each feature.

- 1: monotonic increase
- 0: no constraint
- \-1: monotonic decrease

If monotonic\_cst is None, no constraints are applied.

Monotonicity constraints are not supported for:

- multiclass classifications (i.e. when `n_classes > 2`),
- multioutput classifications (i.e. when `n_outputs_ > 1`),
- classifications trained on data with missing values.

The constraints hold over the probability of the positive class.

Read more in the [User Guide](https://scikit-learn.org/stable/modules/ensemble.html#monotonic-cst-gbdt).

Added in version 1.4.

Attributes:

**classes\_** ndarray of shape (n\_classes,) or list of ndarray

The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).

`feature_importances_` ndarray of shape (n\_features,)

Return the feature importances.

**max\_features\_** int

The inferred value of max\_features.

**n\_classes\_** int or list of int

The number of classes (for single output problems), or a list containing the number of classes for each output (for multi-output problems).

**n\_features\_in\_** int

Number of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

Added in version 0.24.

**feature\_names\_in\_** ndarray of shape (`n_features_in_`,)

Names of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when `X` has feature names that are all strings.

Added in version 1.0.

**n\_outputs\_** int

The number of outputs when `fit` is performed.

**tree\_** Tree instance

The underlying Tree object. Please refer to `help(sklearn.tree._tree.Tree)` for attributes of Tree object and [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py) for basic usage of these attributes.

> [!note] See also
> [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor "sklearn.tree.DecisionTreeRegressor")
> 
> A decision tree regressor.

Notes

The default values for the parameters controlling the size of the trees (e.g. `max_depth`, `min_samples_leaf`, etc.) lead to fully grown and unpruned trees which can potentially be very large on some data sets. To reduce memory consumption, the complexity and size of the trees should be controlled by setting those parameter values.

The method operates using the [`numpy.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax "(in NumPy v2.4)") function on the outputs of. This means that in case the highest predicted probabilities are tied, the classifier will predict the tied class with the lowest index in [classes\_](https://scikit-learn.org/stable/glossary.html#term-classes_).

References

Examples

```
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()
>>> cross_val_score(clf, iris.data, iris.target, cv=10)
...
...
array([ 1.     ,  0.93,  0.86,  0.93,  0.93,
        0.93,  0.93,  1.     ,  0.93,  1.      ])
```

apply(*X*, *check\_input=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L560) [#](#sklearn.tree.DecisionTreeClassifier.apply "Link to this definition")

Return the index of the leaf that each sample is predicted as.

Added in version 0.17.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csr_matrix`.

**check\_input** bool, default=True

Allow to bypass several input checking. Don’t use this parameter unless you know what you’re doing.

Returns:

**X\_leaves** array-like of shape (n\_samples,)

For each datapoint x in X, return the index of the leaf x ends up in. Leaves are numbered within `[0; self.tree_.node_count)`, possibly with gaps in the numbering.

cost\_complexity\_pruning\_path(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L635) [#](#sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path "Link to this definition")

Compute the pruning path during Minimal Cost-Complexity Pruning.

See [Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) for details on the pruning process.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The training input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csc_matrix`.

**y** array-like of shape (n\_samples,) or (n\_samples, n\_outputs)

The target values (class labels) as integers or strings.

**sample\_weight** array-like of shape (n\_samples,), default=None

Sample weights. If None, then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node. Splits are also ignored if they would result in any single class carrying a negative weight in either child node.

Returns:

**ccp\_path** [`Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch "sklearn.utils.Bunch")

Dictionary-like object, with the following attributes.

ccp\_alphasndarray

Effective alphas of subtree during pruning.

impuritiesndarray

Sum of the impurities of the subtree leaves for the corresponding alpha value in `ccp_alphas`.

decision\_path(*X*, *check\_input=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L588) [#](#sklearn.tree.DecisionTreeClassifier.decision_path "Link to this definition")

Return the decision path in the tree.

Added in version 0.18.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csr_matrix`.

**check\_input** bool, default=True

Allow to bypass several input checking. Don’t use this parameter unless you know what you’re doing.

Returns:

**indicator** sparse matrix of shape (n\_samples, n\_nodes)

Return a node indicator CSR matrix where non zero elements indicates that the samples goes through the nodes.

fit(*X*, *y*, *sample\_weight=None*, *check\_input=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L996) [#](#sklearn.tree.DecisionTreeClassifier.fit "Link to this definition")

Build a decision tree classifier from the training set (X, y).

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The training input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csc_matrix`.

**y** array-like of shape (n\_samples,) or (n\_samples, n\_outputs)

The target values (class labels) as integers or strings.

**sample\_weight** array-like of shape (n\_samples,), default=None

Sample weights. If None, then samples are equally weighted. Splits that would create child nodes with net zero or negative weight are ignored while searching for a split in each node. Splits are also ignored if they would result in any single class carrying a negative weight in either child node.

**check\_input** bool, default=True

Allow to bypass several input checking. Don’t use this parameter unless you know what you’re doing.

Returns:

**self** DecisionTreeClassifier

Fitted estimator.

get\_depth() [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L162) [#](#sklearn.tree.DecisionTreeClassifier.get_depth "Link to this definition")

Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root and any leaf.

Returns:

**self.tree\_.max\_depth** int

The maximum depth of the tree.

Get metadata routing of this object.

Please check [User Guide](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing) on how the routing mechanism works.

Returns:

**routing** MetadataRequest

A [`MetadataRequest`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.metadata_routing.MetadataRequest.html#sklearn.utils.metadata_routing.MetadataRequest "sklearn.utils.metadata_routing.MetadataRequest") encapsulating routing information.

get\_n\_leaves() [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L176) [#](#sklearn.tree.DecisionTreeClassifier.get_n_leaves "Link to this definition")

Return the number of leaves of the decision tree.

Returns:

**self.tree\_.n\_leaves** int

Number of leaves.

get\_params(*deep=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L240) [#](#sklearn.tree.DecisionTreeClassifier.get_params "Link to this definition")

Get parameters for this estimator.

Parameters:

**deep** bool, default=True

If True, will return the parameters for this estimator and contained subobjects that are estimators.

Returns:

**params** dict

Parameter names mapped to their values.

predict(*X*, *check\_input=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L509) [#](#sklearn.tree.DecisionTreeClassifier.predict "Link to this definition")

Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is returned. For a regression model, the predicted value based on X is returned.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csr_matrix`.

**check\_input** bool, default=True

Allow to bypass several input checking. Don’t use this parameter unless you know what you’re doing.

Returns:

**y** array-like of shape (n\_samples,) or (n\_samples, n\_outputs)

The predicted classes, or the predict values.

predict\_log\_proba(*X*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L1072) [#](#sklearn.tree.DecisionTreeClassifier.predict_log_proba "Link to this definition")

Predict class log-probabilities of the input samples X.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csr_matrix`.

Returns:

**proba** ndarray of shape (n\_samples, n\_classes) or list of n\_outputs such arrays if n\_outputs > 1

The class log-probabilities of the input samples. The order of the classes corresponds to that in the attribute [classes\_](https://scikit-learn.org/stable/glossary.html#term-classes_).

predict\_proba(*X*, *check\_input=True*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/tree/_classes.py#L1035) [#](#sklearn.tree.DecisionTreeClassifier.predict_proba "Link to this definition")

Predict class probabilities of the input samples X.

The predicted class probability is the fraction of samples of the same class in a leaf.

Parameters:

**X** {array-like, sparse matrix} of shape (n\_samples, n\_features)

The input samples. Internally, it will be converted to `dtype=np.float32` and if a sparse matrix is provided to a sparse `csr_matrix`.

**check\_input** bool, default=True

Allow to bypass several input checking. Don’t use this parameter unless you know what you’re doing.

Returns:

**proba** ndarray of shape (n\_samples, n\_classes) or list of n\_outputs such arrays if n\_outputs > 1

The class probabilities of the input samples. The order of the classes corresponds to that in the attribute [classes\_](https://scikit-learn.org/stable/glossary.html#term-classes_).

score(*X*, *y*, *sample\_weight=None*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L540) [#](#sklearn.tree.DecisionTreeClassifier.score "Link to this definition")

Return [accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) on provided data and labels.

In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.

Parameters:

**X** array-like of shape (n\_samples, n\_features)

Test samples.

**y** array-like of shape (n\_samples,) or (n\_samples, n\_outputs)

True labels for `X`.

**sample\_weight** array-like of shape (n\_samples,), default=None

Sample weights.

Returns:

**score** float

Mean accuracy of `self.predict(X)` w.r.t. `y`.

set\_fit\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.tree.DecisionTreeClassifier.set_fit_request "Link to this definition")

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

set\_params(*\*\*params*) [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/base.py#L338) [#](#sklearn.tree.DecisionTreeClassifier.set_params "Link to this definition")

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects (such as [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline")). The latter have parameters of the form `<component>__<parameter>` so that it’s possible to update each component of a nested object.

Parameters:

**\*\*params** dict

Estimator parameters.

Returns:

**self** estimator instance

Estimator instance.

set\_score\_request(*\**, *sample\_weight: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = '$UNCHANGED$'*) → [\[source\]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/utils/_metadata_requests.py#L1315) [#](#sklearn.tree.DecisionTreeClassifier.set_score_request "Link to this definition")

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