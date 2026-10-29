# Patient-level cross-validation splitting (standalone)

A self-contained copy of SP-Bench's patient-split generator. It produces
**patient-level** train/val/test folds from a clinical metadata CSV, so no patient
ever appears in more than one fold. Supports classification labels and survival
`[time, event]` labels.

Extracted from `src/sp_bench/data/{splits,split_generator}.py`. The only change from
the originals is that imports were made local/lazy so this folder runs on its own.

## Install

```bash
pip install -r requirements.txt   # numpy, scikit-learn, pandas
```

(`h5py` is only needed for the legacy `--data-root` path and is left commented out.)

## Input

A CSV with one row per patient (or per sample, as long as `patient_id` repeats
consistently). At minimum a patient-ID column; plus whatever outcome you split on:

| column            | meaning                                   |
|-------------------|-------------------------------------------|
| `patient_id`      | patient identifier (required)             |
| `site_id`         | site/batch, for leave-one-site-out (opt.) |
| `label`           | scalar outcome, e.g. 0/1 (opt.)           |
| `os_time`,`os_event` | survival time + event indicator (opt.) |

## Usage — survival, 5-fold

```bash
python split_generator.py \
  --metadata-csv clinical.csv \
  --dataset-name mycohort \
  --patient-id-column patient_id \
  --survival-time-column os_time \
  --survival-event-column os_event \
  --method stratified_kfold --n-folds 5 --val-frac 0.1 --seed 42 \
  --patient-splits splits/mycohort_patient.json
```

### Small cohorts (leave-one-out)

```bash
python split_generator.py --metadata-csv clinical.csv --dataset-name mycohort \
  --survival-time-column os_time --survival-event-column os_event \
  --method loocv --patient-splits splits/mycohort_patient.json
```

Evaluate LOOCV by **pooling out-of-fold predictions** and computing one metric on the
pooled scores — never average per-fold metrics (a one-patient test fold has no defined
C-index/AUC).

## Output

`--patient-splits` is a JSON keyed by `--dataset-name`, then `folds`, then fold index,
each holding `train` / `val` / `test` patient-ID lists. The writer **merges** into an
existing file, so several cohorts can share one JSON.

Load them back with `load_splits(path, dataset_name)` from `splits.py`, which also
raises on any train/test patient overlap.

## Important caveat: `--stratify` with survival

`--stratify` passes the label straight to scikit-learn's `StratifiedKFold`, which
expects a **1-D class vector**. A survival label is the `[time, event]` pair, so
`--stratify` does **not** stratify on the event the way you'd expect. For survival,
leave `--stratify` off (plain k-fold) unless you first reduce the label to a single
discrete stratum (e.g. the event indicator, or an event×time-bin key) yourself.
`--stratify` is appropriate for a single discrete classification `label`.

## Reproducibility

Folds are deterministic given `--seed`. Record the seed, `--n-folds`, `--val-frac`,
and `--method` alongside results.
