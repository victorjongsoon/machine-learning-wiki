# Clustering

Unsupervised grouping of unlabeled data. `sklearn.cluster` provides both class-based (`fit` + `labels_`) and function-based APIs.

**Inductive** methods can assign labels to new data. **Transductive** methods only label training data.

---

## Algorithm Overview

| Algorithm | Requires k? | Scalability | Good for |
|-----------|------------|-------------|----------|
| K-Means | Yes | Very large n_samples | Even clusters, flat geometry |
| Mini-Batch K-Means | Yes | Very large | Same as K-Means, faster |
| Affinity Propagation | No (auto) | Small/medium | Many clusters, uneven sizes |
| Mean Shift | No (auto) | Poor | Non-flat geometry, inductive |
| Spectral | Yes | Medium | Non-flat geometry, few clusters |
| Agglomerative | Yes or threshold | Large | Hierarchical, connectivity |
| DBSCAN | No | Very large | Arbitrary shapes, outlier removal |
| HDBSCAN | No | Large | Variable density, outlier removal |
| OPTICS | No | Large | Variable density, hierarchical |
| BIRCH | Yes/optional | Very large | Data reduction, outlier removal |

---

## K-Means

Minimizes **inertia** (within-cluster sum of squares):

$$\sum_{i=0}^n \min_{\mu_j \in C} \|x_i - \mu_j\|^2$$

Lloyd's algorithm: (1) assign each point to nearest centroid, (2) recompute centroids, repeat until convergence.

**Pitfalls:**
- Assumes convex, isotropic clusters; struggles with elongated or irregular shapes.
- Converges to local minima — run multiple times (`n_init`).
- Distances inflate in high dimensions — apply PCA first.

**k-means++ initialization** (`init='k-means++'`): seeds centroids far apart for better convergence. Default.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='k-means++', n_init=10)
labels = km.fit_predict(X)
km.inertia_  # lower is better
```

### Mini-Batch K-Means

Processes random mini-batches each iteration; significantly faster with slightly worse quality.

```python
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=3, batch_size=256)
```

---

## Affinity Propagation

Sends "responsibility" and "availability" messages between all pairs of points until convergence. Automatically determines the number of clusters via the `preference` parameter. $O(N^2 T)$ time — only suitable for small/medium datasets.

---

## Mean Shift

Finds cluster centers as modes of the data density. `bandwidth` parameter controls the kernel size; use `estimate_bandwidth` if unknown. Not scalable; no need to specify k.

---

## Spectral Clustering

Embeds data via the affinity matrix eigenvectors, then clusters in the low-dimensional space (typically with K-Means).

- Requires number of clusters in advance.
- Good for non-convex cluster shapes and image segmentation.
- Similarity matrix must be non-negative and well-distributed; apply a heat kernel if needed: `similarity = np.exp(-beta * distance / distance.std())`.

`assign_labels` options: `'kmeans'` (fine-grained, unstable), `'discretize'` (reproducible, geometric), `'cluster_qr'` (deterministic, often visually best).

---

## Hierarchical Clustering (Agglomerative)

Builds a nested cluster tree bottom-up. `AgglomerativeClustering` supports four linkage strategies:

| Linkage | Merges by | Notes |
|---------|-----------|-------|
| `ward` | Min variance increase | Most regular cluster sizes; Euclidean only |
| `complete` | Max pairwise distance | Compact clusters |
| `average` | Average pairwise distance | Works with non-Euclidean metrics |
| `single` | Min pairwise distance | Efficient; fragile to noise; handles non-globular shapes |

**Connectivity constraints:** pass a sparse adjacency matrix to restrict which clusters can merge (e.g., only neighboring pixels in an image). Speeds up computation for large n.

**Bisecting K-Means** (`BisectingKMeans`): divisive hierarchical variant; splits iteratively. More efficient than K-Means for large k; never produces empty clusters.

---

## DBSCAN

Clusters as dense regions separated by low-density areas. Detects outliers (noise points labeled `-1`).

**Key concepts:**
- **Core sample:** a point with at least `min_samples` neighbors within radius `eps`.
- **Cluster:** maximal set of connected core samples + their border points.

**Parameter guide:**
- `eps` is crucial and dataset-specific. Too small → most points are noise. Too large → clusters merge. Use a knee in the nearest-neighbor distance plot.
- `min_samples` controls noise tolerance.

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
db.labels_  # -1 = noise
```

---

## HDBSCAN

Extension of DBSCAN that runs clustering across all density scales. Uses a **mutual reachability graph** and minimum spanning tree to extract a cluster hierarchy, then selects stable clusters.

- Does not require `eps`; only `min_samples` (and optionally `min_cluster_size`).
- More robust to variable-density clusters than DBSCAN.

**Algorithm sketch:**
1. Compute core distances and mutual reachability distances.
2. Extract MST of mutual reachability graph.
3. Greedily cut edges from highest to lowest weight.
4. Select stable clusters from the resulting hierarchy.

---

## OPTICS

Generalizes DBSCAN by computing a **reachability plot** — an ordering of points by density. Cluster extraction can be done at any `eps` threshold after fitting (`cluster_optics_dbscan`), or automatically using steep-slope detection (`xi` parameter). Supports hierarchical cluster representations.

---

## BIRCH

Builds a Clustering Feature Tree (CFT) that compresses data into subclusters. Memory-efficient for very large datasets. Two parameters: `branching_factor` (tree width) and `threshold` (max subcluster radius). An optional global clusterer (`n_clusters`) relabels the leaf subclusters.

---

## Evaluation Metrics

### With ground truth labels

| Metric | Notes |
|--------|-------|
| `rand_score` / `adjusted_rand_score` | Similarity of assignments; ARI corrects for chance |
| `adjusted_mutual_info_score` (AMI) | Normalized against chance |
| `normalized_mutual_info_score` (NMI) | In [0, 1]; not corrected for chance |
| `homogeneity_score` | Each cluster contains one class |
| `completeness_score` | Each class is in one cluster |
| `v_measure_score` | Harmonic mean of H and C (= NMI with arithmetic mean) |
| `fowlkes_mallows_score` | Geometric mean of pairwise precision/recall |

### Without ground truth labels

| Metric | Direction | Notes |
|--------|-----------|-------|
| **Silhouette Coefficient** | Higher = better (max 1) | $s = (b-a)/\max(a,b)$; requires pairwise distances |
| **Calinski-Harabasz Index** | Higher = better | Ratio of between-cluster to within-cluster dispersion |
| **Davies-Bouldin Index** | Lower = better | Average cluster similarity ratio; 0 is best |

```python
from sklearn import metrics
# With labels
metrics.adjusted_rand_score(labels_true, labels_pred)
metrics.v_measure_score(labels_true, labels_pred)

# Without labels
metrics.silhouette_score(X, labels, metric='euclidean')
metrics.calinski_harabasz_score(X, labels)
metrics.davies_bouldin_score(X, labels)
```
