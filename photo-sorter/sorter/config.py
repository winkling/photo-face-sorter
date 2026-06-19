import os
import yaml

REQUIRED_KEYS = [
    "input_dir", "output_dir", "review_dir", "unsorted_dir", "other_dir", "db_path",
    "device", "det_size", "min_det_score",
    "match_threshold", "cluster_eps", "cluster_min_samples",
    "placement", "enroll_recognized", "max_embeddings_per_person",
]

PATH_KEYS = ["input_dir", "output_dir", "review_dir", "unsorted_dir", "other_dir", "db_path"]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    for key in PATH_KEYS:
        cfg[key] = os.path.expanduser(cfg[key])
    return cfg
