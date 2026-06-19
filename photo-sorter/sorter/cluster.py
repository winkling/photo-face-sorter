import numpy as np
from sklearn.cluster import DBSCAN


def cluster(embs: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embs)
    return labels


def assign_groups(labels: np.ndarray) -> dict:
    """Map each index to a group_XXXX id. Noise points each get a unique singleton group."""
    groups = {}
    cluster_to_group = {}
    next_group = [1]

    def new_group_id():
        gid = f"group_{next_group[0]:04d}"
        next_group[0] += 1
        return gid

    for i, label in enumerate(labels):
        if label == -1:
            groups[i] = new_group_id()
        else:
            if label not in cluster_to_group:
                cluster_to_group[label] = new_group_id()
            groups[i] = cluster_to_group[label]

    return groups
