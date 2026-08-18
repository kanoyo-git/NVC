import numpy as np


def retrieve_index_features(index, index_vectors, query, max_neighbors=8, epsilon=1e-6):
    """Return distance-weighted FAISS features without NaNs or invalid neighbors.

    NVC indices are trained with squared L2 distance. Exact matches therefore
    legitimately have distance zero; clamping them to ``epsilon`` preserves their
    dominance without producing ``inf / inf``. Rows for which FAISS returns no
    valid neighbor fall back to the original query feature.
    """
    query = np.asarray(query)
    index_vectors = np.asarray(index_vectors)
    if query.ndim != 2 or index_vectors.ndim != 2:
        raise ValueError("FAISS query and index vectors must be two-dimensional")
    if query.shape[1] != index_vectors.shape[1]:
        raise ValueError(
            "FAISS query dimension %s does not match index dimension %s"
            % (query.shape[1], index_vectors.shape[1])
        )
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    available = min(int(getattr(index, "ntotal", 0)), len(index_vectors))
    if available <= 0 or len(query) == 0:
        return query.copy()

    search_query = np.ascontiguousarray(query, dtype=np.float32)
    distances, neighbors = index.search(
        search_query, k=min(max_neighbors, available)
    )
    distances = np.asarray(distances)
    neighbors = np.asarray(neighbors)
    if distances.shape != neighbors.shape or distances.shape[0] != query.shape[0]:
        raise ValueError("FAISS returned inconsistent distance/index arrays")

    valid = (
        (neighbors >= 0)
        & (neighbors < len(index_vectors))
        & np.isfinite(distances)
        & (distances >= 0)
    )
    safe_neighbors = np.where(valid, neighbors, 0)
    safe_distances = np.maximum(distances.astype(np.float64, copy=False), epsilon)
    weights = np.where(valid, 1.0 / np.square(safe_distances), 0.0)
    weight_sums = weights.sum(axis=1, keepdims=True)
    valid_rows = weight_sums[:, 0] > 0

    result = query.copy()
    if np.any(valid_rows):
        normalized = weights[valid_rows] / weight_sums[valid_rows]
        selected = index_vectors[safe_neighbors[valid_rows]]
        result[valid_rows] = np.sum(selected * normalized[..., None], axis=1)
    return result
