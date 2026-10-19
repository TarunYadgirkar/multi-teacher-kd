"""Experiment config: one YAML holds shared settings plus a `conditions` table whose
entries override `train` settings per condition."""

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

REQUIRED = ("student", "teachers", "default_teacher", "data", "generation", "train", "conditions")


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["default_teacher"] not in cfg["teachers"]:
        raise ValueError(f"default_teacher {cfg['default_teacher']!r} is not in teachers")
    cfg.setdefault("revisions", {})
    cfg.setdefault("paths", {"results": "results", "outputs": "outputs", "figures": "assets/generated"})
    cfg["config_path"] = str(path)
    return cfg


def condition_train_cfg(cfg: dict, condition: str) -> dict:
    spec = cfg["conditions"][condition]
    return deep_merge(cfg["train"], spec.get("train", {}))


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def apply_memory_cap(cfg: dict) -> None:
    """Cap the MPS allocator before torch initialises it (no effect on CUDA/CPU)."""
    fraction = cfg.get("mps_memory_fraction")
    if fraction:
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(fraction))
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(round(fraction * 0.8, 3)))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
