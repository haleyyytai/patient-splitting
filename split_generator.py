from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Standalone copy: these import from the sibling splits.py in this folder
# (upstream this is `from sp_bench.data.splits import ...`).
from splits import (
    generate_loocv_splits,
    generate_patient_splits,
    generate_site_splits,
    save_splits,
)


def collect_patient_metadata_from_csv(
    metadata_csv: str | Path,
    *,
    patient_id_column: str = "patient_id",
    site_id_column: str | None = "site_id",
    label_column: str | None = "label",
    survival_time_column: str | None = None,
    survival_event_column: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect one row per patient from the clinical metadata CSV.

    The CSV is the *source* of clinical metadata. This is the supported path and
    reads no imaging files.

    Args:
        metadata_csv: Path to the clinical metadata CSV.
        patient_id_column: Column holding the patient identifier.
        site_id_column: Column holding the site identifier, or ``None``.
        label_column: Column holding a scalar outcome, or ``None``.
        survival_time_column: Column holding survival time, or ``None``.
        survival_event_column: Column holding the survival event indicator, or ``None``.

    Returns:
        ``{patient_id: {"patient_id", "site_id", "label"}}``. ``label`` is a float, a
        ``[time, event]`` pair when both survival columns are given, or ``None``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        KeyError: If ``patient_id_column`` is absent, or a named outcome column is.
        ValueError: If one patient carries two different labels.
    """
    import pandas as pd

    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    if patient_id_column not in df.columns:
        raise KeyError(
            f"patient_id_column {patient_id_column!r} not in {path.name}; "
            f"columns are {sorted(df.columns)}"
        )
    # Name a missing outcome column rather than silently producing label-less rows,
    # which would surface much later as "cannot stratify".
    for role, column in (
        ("label_column", label_column),
        ("survival_time_column", survival_time_column),
        ("survival_event_column", survival_event_column),
    ):
        if column is not None and column not in df.columns:
            raise KeyError(
                f"{role} {column!r} not in {path.name}; columns are {sorted(df.columns)}"
            )

    rows: dict[str, dict[str, Any]] = {}
    for record in df.to_dict("records"):
        patient_id = str(record[patient_id_column])
        label = _label_from_record(
            record,
            label_column=label_column,
            survival_time_column=survival_time_column,
            survival_event_column=survival_event_column,
        )
        site_id = str(record[site_id_column]) if site_id_column in record else ""
        existing = rows.get(patient_id)
        if existing is None:
            rows[patient_id] = {
                "patient_id": patient_id,
                "site_id": site_id,
                "label": label,
            }
            continue
        # One patient, two labels is a data defect that must not be resolved by
        # whichever row happened to come first.
        if existing["label"] != label and not _both_nan(existing["label"], label):
            raise ValueError(
                f"Patient {patient_id!r} carries conflicting labels in {path.name}: "
                f"{existing['label']!r} and {label!r}"
            )
    return rows


def _both_nan(a: Any, b: Any) -> bool:
    try:
        return bool(np.isnan(a).all() and np.isnan(b).all())
    except (TypeError, ValueError):
        return False


def _label_from_record(
    record: dict[str, Any],
    *,
    label_column: str | None,
    survival_time_column: str | None,
    survival_event_column: str | None,
):
    if survival_time_column and survival_event_column:
        time, event = record.get(survival_time_column), record.get(survival_event_column)
        if time is not None and event is not None:
            return np.asarray([float(time), float(event)], dtype=np.float32)
    if label_column:
        value = record.get(label_column)
        if value is not None and value == value and value != "":  # NaN-safe
            return float(value)
    return None


def _collect_h5_files(root_dir: str) -> list[str]:
    """Recursively list .h5 files under ``root_dir`` (legacy H5 path helper)."""
    import os

    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in sorted(filenames):
            if name.endswith(".h5"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def collect_patient_metadata(data_root: str | Path) -> dict[str, dict[str, Any]]:
    """Collect one metadata row per patient from H5 patches.

    **Legacy.** Label-blind corpora carry no ``/label`` and this returns
    ``label=None`` for every patient. Use :func:`collect_patient_metadata_from_csv`
    instead. Requires ``h5py`` (imported lazily so the CSV path does not need it).
    """
    import h5py  # noqa: F401  (lazy: only the legacy H5 path needs it)

    rows: dict[str, dict[str, Any]] = {}
    for h5_path in _collect_h5_files(str(data_root)):
        with h5py.File(h5_path, "r") as f:
            meta = _read_meta(f)
            patient_id = str(meta.get("patient_id") or Path(h5_path).stem.split("_")[0])
            site_id = str(f.attrs.get("site_id", meta.get("site_id", "")))
            label = _read_label(f)
        rows.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "site_id": site_id,
                "label": label,
            },
        )
    return rows


def generate_splits_from_metadata_csv(
    metadata_csv: str | Path,
    *,
    dataset_name: str,
    patient_split_path: str | Path,
    site_split_path: str | Path | None = None,
    patient_id_column: str = "patient_id",
    site_id_column: str | None = "site_id",
    label_column: str | None = "label",
    survival_time_column: str | None = None,
    survival_event_column: str | None = None,
    n_folds: int = 5,
    val_frac: float = 0.1,
    seed: int = 42,
    stratify: bool = False,
    method: str = "stratified_kfold",
) -> dict[str, Any]:
    """Generate patient-level splits from the clinical metadata CSV, reading no H5.

    Args:
        metadata_csv: Path to the clinical metadata CSV.
        dataset_name: Key the splits are written under in the output JSON.
        patient_split_path: Where the patient splits are written.
        site_split_path: Where leave-one-site-out splits are written, or ``None``.
        patient_id_column: Column holding the patient identifier.
        site_id_column: Column holding the site identifier, or ``None``.
        label_column: Column holding a scalar outcome, or ``None``.
        survival_time_column: Column holding survival time, or ``None``.
        survival_event_column: Column holding the survival event indicator, or ``None``.
        n_folds: Number of folds for ``stratified_kfold``.
        val_frac: Fraction of each fold's training patients held out for validation.
        seed: Random seed.
        stratify: Stratify folds on the label. Requires every patient to have one.
        method: ``"stratified_kfold"`` or ``"loocv"``.

    Returns:
        A summary dict including the fold assignments.

    Raises:
        ValueError: If fewer than two patients are present, or stratification is
            requested and a patient has no label.
    """
    rows = collect_patient_metadata_from_csv(
        metadata_csv,
        patient_id_column=patient_id_column,
        site_id_column=site_id_column,
        label_column=label_column,
        survival_time_column=survival_time_column,
        survival_event_column=survival_event_column,
    )
    if not rows:
        raise ValueError(f"No patient rows found in {metadata_csv}")
    return _splits_from_rows(
        rows,
        dataset_name=dataset_name,
        patient_split_path=patient_split_path,
        site_split_path=site_split_path,
        n_folds=n_folds,
        val_frac=val_frac,
        seed=seed,
        stratify=stratify,
        method=method,
    )


def generate_splits_from_h5(
    data_root: str | Path,
    *,
    dataset_name: str,
    patient_split_path: str | Path,
    site_split_path: str | Path | None = None,
    n_folds: int = 5,
    val_frac: float = 0.1,
    seed: int = 42,
    stratify: bool = False,
    method: str = "stratified_kfold",
) -> dict[str, Any]:
    """Generate splits by reading prepared H5 patches.

    **Legacy**, retained for corpora prepared before patch preparation became
    label-blind. Those corpora carry ``/label``; corpora written since do not, so
    ``stratify=True`` here will fail on them. Use
    :func:`generate_splits_from_metadata_csv`.
    """
    rows = collect_patient_metadata(data_root)
    if not rows:
        raise FileNotFoundError(f"No H5 patient metadata found under {data_root}")
    return _splits_from_rows(
        rows,
        dataset_name=dataset_name,
        patient_split_path=patient_split_path,
        site_split_path=site_split_path,
        n_folds=n_folds,
        val_frac=val_frac,
        seed=seed,
        stratify=stratify,
        method=method,
    )


def _splits_from_rows(
    rows: dict[str, dict[str, Any]],
    *,
    dataset_name: str,
    patient_split_path: str | Path,
    site_split_path: str | Path | None,
    n_folds: int,
    val_frac: float,
    seed: int,
    stratify: bool,
    method: str,
) -> dict[str, Any]:
    """Turn one row per patient into fold assignments. Source-agnostic by design."""
    patient_ids = sorted(rows)
    if len(patient_ids) < 2:
        raise ValueError("Need at least 2 patients to generate patient-level splits")

    labels = [rows[p]["label"] for p in patient_ids] if stratify else None
    if labels is not None and any(v is None for v in labels):
        raise ValueError("Cannot stratify because at least one patient has no label")

    n = len(patient_ids)
    if method.lower() in {"loocv", "leave_one_out", "loo"}:
        # Leave-one-out: each patient is a one-patient test fold. Pair with a
        # pooled out-of-fold evaluator (do NOT average per-fold metrics).
        folds = generate_loocv_splits(patient_ids)
    else:
        folds = generate_patient_splits(
            patient_ids,
            labels=labels,
            n_folds=max(2, min(int(n_folds), n)),
            val_frac=val_frac,
            seed=seed,
        )
    save_splits(folds, str(patient_split_path), dataset_name=dataset_name)

    site_result = None
    if site_split_path is not None:
        site_ids = [str(rows[p].get("site_id", "")) for p in patient_ids]
        site_splits = generate_site_splits(patient_ids, site_ids)
        _save_site_splits(site_splits, site_split_path, dataset_name)
        site_result = {k: v.__dict__ for k, v in site_splits.items()}

    return {
        "dataset_name": dataset_name,
        "n_patients": len(patient_ids),
        "patient_split_path": str(patient_split_path),
        "site_split_path": str(site_split_path) if site_split_path else None,
        "patient_ids": patient_ids,
        "patient_splits": {k: v.__dict__ for k, v in folds.items()},
        "site_splits": site_result,
    }


def _read_meta(f) -> dict[str, Any]:
    if "meta" not in f:
        return {}
    raw = f["meta"][()]
    return json.loads(raw if isinstance(raw, str) else raw.decode())


def _read_label(f):
    if "label" not in f:
        return None
    value = f["label"][()]
    try:
        if getattr(value, "shape", ()):
            return float(value[0])
    except TypeError:
        pass
    return float(value)


def _save_site_splits(splits, path: str | Path, dataset_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[dataset_name] = {
        "sites": sorted(splits),
        "splits": {
            k: {"train": v.train, "val": v.val, "test": v.test}
            for k, v in splits.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate patient-level cross-validation splits. Prefer --metadata-csv: "
            "the CSV is the source of clinical metadata."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata-csv",
        help="Clinical metadata CSV. The supported source; reads no H5.",
    )
    source.add_argument(
        "--data-root",
        help=(
            "LEGACY: read labels back out of prepared H5 patches. Only works for corpora "
            "written before patch preparation became label-blind. Requires h5py."
        ),
    )
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument("--site-id-column", default="site_id")
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Scalar-outcome column, e.g. a 0/1 class. Leave unset for survival-only "
            "runs (use --survival-time-column/--survival-event-column). If set, the "
            "column must exist."
        ),
    )
    parser.add_argument("--survival-time-column")
    parser.add_argument("--survival-event-column")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--patient-splits", required=True)
    parser.add_argument("--site-splits")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument(
        "--method",
        default="stratified_kfold",
        choices=["stratified_kfold", "loocv"],
        help="Patient-split scheme: 'stratified_kfold' (default) or 'loocv' for small cohorts.",
    )
    args = parser.parse_args()

    common = {
        "dataset_name": args.dataset_name,
        "patient_split_path": args.patient_splits,
        "site_split_path": args.site_splits,
        "n_folds": args.n_folds,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "stratify": args.stratify,
        "method": args.method,
    }
    if args.metadata_csv:
        result = generate_splits_from_metadata_csv(
            args.metadata_csv,
            patient_id_column=args.patient_id_column,
            site_id_column=args.site_id_column,
            label_column=args.label_column,
            survival_time_column=args.survival_time_column,
            survival_event_column=args.survival_event_column,
            **common,
        )
    else:
        print(
            "WARN: --data-root reads labels back out of prepared H5 patches. Prefer "
            "--metadata-csv; label-blind corpora carry no label and stratification will fail.",
            file=sys.stderr,
        )
        result = generate_splits_from_h5(args.data_root, **common)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
