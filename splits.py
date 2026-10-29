from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SplitSet:
    """One train/val/test patient-ID split."""
    train: List[str]
    val:   List[str]
    test:  List[str]

    def all_ids(self) -> List[str]:
        return self.train + self.val + self.test

    def __repr__(self) -> str:
        return (f"SplitSet(train={len(self.train)}, "
                f"val={len(self.val)}, test={len(self.test)})")


def load_splits(
    split_path: str,
    dataset_name: str,
    split_type: str = "patient_level",
) -> Dict[str, SplitSet]:
    """
    Load splits from a JSON file produced by generate_splits().

    Args:
        split_path: path to patient_splits.json or site_splits.json
        dataset_name: key inside the JSON (e.g. "lung_codex")
        split_type: "patient_level" returns fold-keyed dict; "site_level" returns site-keyed dict

    Returns:
        Dict mapping fold/site key → SplitSet
    """
    with open(split_path) as f:
        data = json.load(f)

    if dataset_name not in data:
        raise KeyError(f"Dataset '{dataset_name}' not found in {split_path}. "
                       f"Available: {[k for k in data if not k.startswith('_')]}")

    ds = data[dataset_name]

    if split_type == "patient_level":
        folds = ds.get("folds", {})
        result = {}
        for fold_key, fold in folds.items():
            train_ids = fold["train"]
            test_ids  = fold["test"]
            overlap   = set(train_ids) & set(test_ids)
            if overlap:
                raise ValueError(
                    f"Data leakage in {dataset_name} fold '{fold_key}': "
                    f"{len(overlap)} patient(s) appear in both train and test "
                    f"(e.g. {sorted(overlap)[:3]}). Check split generation."
                )
            result[fold_key] = SplitSet(
                train=train_ids,
                val=fold.get("val", []),
                test=test_ids,
            )
        return result

    elif split_type == "site_level":
        splits = ds.get("splits", {})
        return {
            site: SplitSet(
                train=info.get("train", []),
                val=info.get("val", []),
                test=info.get("test", []),
            )
            for site, info in splits.items()
        }

    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")


def generate_patient_splits(
    patient_ids: List[str],
    labels: Optional[List] = None,
    n_folds: int = 5,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, SplitSet]:
    """
    Generate stratified k-fold patient splits in memory.

    Args:
        patient_ids: unique patient IDs
        labels: optional labels for stratification (e.g. binary outcome)
        n_folds: number of folds
        val_frac: fraction of training patients held out for validation
        seed: random seed

    Returns:
        Dict '0' .. 'n_folds-1' → SplitSet
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    ids = np.array(patient_ids)
    rng = np.random.default_rng(seed)

    if labels is not None:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kf.split(ids, labels)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kf.split(ids)

    result = {}
    for fold_idx, (train_idx, test_idx) in enumerate(splits_iter):
        train_ids = ids[train_idx].tolist()
        test_ids  = ids[test_idx].tolist()

        # val_frac == 0 must mean NO validation split (don't force a patient out
        # of training). Only round up to >=1 when a validation set was requested.
        n_val = int(len(train_ids) * val_frac)
        if val_frac > 0 and n_val == 0 and len(train_ids) > 1:
            n_val = 1
        n_val = min(n_val, max(0, len(train_ids) - 1))
        perm  = rng.permutation(len(train_ids))
        val_ids   = [train_ids[i] for i in perm[:n_val]]
        train_ids = [train_ids[i] for i in perm[n_val:]]

        result[str(fold_idx)] = SplitSet(train=train_ids, val=val_ids, test=test_ids)

    return result


def generate_site_splits(
    patient_ids: List[str],
    site_ids: List[str],
) -> Dict[str, SplitSet]:
    """
    Generate leave-one-site-out splits.

    Args:
        patient_ids: list of patient IDs (parallel to site_ids)
        site_ids: corresponding site identifier for each patient

    Returns:
        Dict site_name → SplitSet (test = all patients from that site)
    """
    sites = sorted(set(site_ids))
    result = {}
    for held_out in sites:
        test  = [p for p, s in zip(patient_ids, site_ids) if s == held_out]
        train = [p for p, s in zip(patient_ids, site_ids) if s != held_out]
        result[held_out] = SplitSet(train=train, val=[], test=test)
    return result


def generate_loocv_splits(patient_ids: List[str]) -> Dict[str, SplitSet]:
    """
    Leave-one-out cross-validation splits.

    Each patient becomes the single test patient of one fold; all other
    patients form that fold's training set. Suited to small cohorts (e.g. MCC,
    GBM) where k-fold would leave too few patients per test fold.

    A single-patient test fold has no defined AUC/C-index on its own, so callers
    must evaluate LOOCV by pooling out-of-fold predictions across all folds and
    computing one metric on the pooled scores — not by averaging per-fold
    metrics.

    Returns:
        Dict patient_id → SplitSet(train=others, val=[], test=[patient_id])
    """
    ids = list(patient_ids)
    if len(ids) < 2:
        raise ValueError("Need at least 2 patients to generate LOOCV splits")
    return {
        pid: SplitSet(train=[p for p in ids if p != pid], val=[], test=[pid])
        for pid in ids
    }


def save_splits(splits: Dict[str, SplitSet], path: str, dataset_name: str) -> None:
    """Persist generated splits to JSON, merging with any existing file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data: Dict = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    data[dataset_name] = {
        "n_folds": len(splits),
        "folds": {
            k: {"train": s.train, "val": s.val, "test": s.test}
            for k, s in splits.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
