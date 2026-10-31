# NeurIPS Open Polymer Prediction 2025

Predict 5 polymer properties (`Tg`, `FFV`, `Tc`, `Density`, `Rg`) from SMILES strings.
Competition page: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025
(submission deadline has passed — this is a learning/portfolio project, scored locally
with the competition's own weighted-MAE metric).

## Setup

```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Data was downloaded via the Kaggle API into `data/` (`train.csv`, `test.csv`,
`sample_submission.csv`, `train_supplement/*.csv`).

## Key challenge: extreme target sparsity

Each training row typically has only 1-2 of the 5 target properties filled in
(e.g. `FFV` is ~99% populated, `Tg`/`Density`/`Rg`/`Tc` are only 7-11% populated).
`train_supplement/` contains extra SMILES with single-property labels for `Tc`,
`Tg`, and `FFV`, which get merged in. The approach: train **one independent
model per target**, each only on the rows where that target is known.

## EDA (`eda.py`)

Run `./.venv/Scripts/python.exe eda.py` to regenerate stats and plots in
`eda_outputs/`. Key findings:

- Target fill rate: FFV has 7,030 labels, the other four only have 511-867.
  7,266 of 7,973 rows have exactly 1 target filled; only 240 rows have 4.
- **Tg and FFV never co-occur in the same row** — there is zero direct overlap
  to learn a Tg/FFV relationship from within `train.csv` itself.
- Notable target correlations (on the rows where both happen to be known):
  Tc vs. Density is strongly negative (-0.49), Tc vs. Rg is strongly positive
  (0.55) — denser polymers tend to conduct heat less well and have smaller Rg.
- SMILES length ranges from 3 to 306 characters (median 52), i.e. molecule
  complexity varies a lot across the dataset.

## Pipeline (`train_baseline.py`)

1. Load `train.csv` + supplement files, merge/dedupe by SMILES.
2. Featurize every SMILES with RDKit: ~20 physicochemical descriptors (MolWt,
   LogP, TPSA, ring counts, Kappa/Chi shape indices, etc.) + a 256-bit Morgan
   fingerprint (`src/features.py`).
3. For each target: 5-fold CV with LightGBM, early stopping, report MAE.
4. Combine per-target OOF predictions and score with the competition's actual
   weighted MAE formula (`src/metric.py`), which upweights rare/narrow-range
   properties.
5. Retrain each target's model on all available labels, predict on `test.csv`,
   write `submission.csv`.

Run it:

```
./.venv/Scripts/python.exe train_baseline.py
```

Current baseline result: **CV wMAE ≈ 0.027**.

### Feature importance

Across all 5 targets, the hand-crafted RDKit descriptors (`MolLogP`, `BalabanJ`,
`Kappa3`, `FractionCSP3`, `HallKierAlpha`) consistently rank as the most-used
features — more than any individual Morgan fingerprint bit. This suggests the
256-bit fingerprint isn't adding much signal here relative to the descriptors,
and is a good candidate to prune, replace, or expand in a future iteration.

## Ideas to improve from here

- Better/more descriptors: 3D conformer-derived features, more fingerprint bits,
  or a pretrained chemistry language model (e.g. ChemBERTa) embedding of the SMILES.
- Try a graph neural network directly on the molecular graph instead of
  hand-engineered descriptors.
- Tune LightGBM per target (current params are a reasonable shared default)
  or try CatBoost/XGBoost ensembles.
- Use `train_supplement/dataset2.csv` (unlabeled SMILES) for pretraining/
  self-supervised embeddings since it's much larger than the labeled sets.
- Blend multiple models per target and average.
