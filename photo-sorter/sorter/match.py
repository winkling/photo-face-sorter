import numpy as np


def best_match(emb: np.ndarray, mat: np.ndarray, person_ids: list, threshold: float):
    if mat.shape[0] == 0:
        return None, -1.0
    sims = mat @ emb
    i = int(np.argmax(sims))
    sim = float(sims[i])
    return (person_ids[i] if sim >= threshold else None), sim
