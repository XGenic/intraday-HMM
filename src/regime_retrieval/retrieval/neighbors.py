"""Deterministic exact kNN with stable tie-breaking and session diversity."""

import numpy as np
from scipy.spatial.distance import cdist


def exact_knn(
    query: np.ndarray,
    candidates: np.ndarray,
    sessions: np.ndarray,
    k: int,
    max_per_session: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    if k < 1 or max_per_session < 1:
        raise ValueError("k and max_per_session must be positive")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("unsupported distance")
    if not np.isfinite(query).all() or not np.isfinite(candidates).all():
        raise ValueError("retrieval requires finite representations")
    if len(candidates) != len(sessions):
        raise ValueError("candidate/session lengths differ")
    if not len(candidates):
        return np.empty(0, dtype=int), np.empty(0)
    if metric == "euclidean":
        distances = cdist(query[None, :], candidates, metric="euclidean")[0]
    else:
        # Cosine is otherwise undefined at zero: both zero match, one zero is orthogonal.
        norms = np.linalg.norm(candidates, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = np.divide(
            candidates @ query,
            norms * query_norm,
            out=np.zeros(len(candidates)),
            where=(norms * query_norm) > 0,
        )
        distances = 1 - np.clip(similarities, -1, 1)
        if query_norm == 0:
            distances[norms == 0] = 0
    order = np.argsort(distances, kind="stable")
    counts: dict[int, int] = {}
    selected = []
    for index in order:
        session = int(sessions[index])
        if counts.get(session, 0) >= max_per_session:
            continue
        selected.append(index)
        counts[session] = counts.get(session, 0) + 1
        if len(selected) == k:
            break
    selected_array = np.asarray(selected, dtype=int)
    return selected_array, distances[selected_array]
