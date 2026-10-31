"""Baseline pipeline for NeurIPS Open Polymer Prediction 2025.

Per-target LightGBM models trained on RDKit descriptor + Morgan fingerprint
features, each fit only on rows where that target is non-null (optionally
augmented with supplementary labeled SMILES). Reports out-of-fold CV wMAE
using the competition's own metric, then retrains on all data and writes
submission.csv.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import featurize
from metric import TARGETS, property_weights, wmae

DATA_DIR = Path(__file__).parent / "data"
N_SPLITS = 5
SEED = 42

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=SEED,
)


def load_supplement_data():
    """Supplementary labeled SMILES for Tc, Tg, FFV; merge into extra train rows."""
    sup1 = pd.read_csv(DATA_DIR / "train_supplement/dataset1.csv").rename(columns={"TC_mean": "Tc"})
    sup3 = pd.read_csv(DATA_DIR / "train_supplement/dataset3.csv")
    sup4 = pd.read_csv(DATA_DIR / "train_supplement/dataset4.csv")
    frames = []
    for df, cols in [(sup1, ["SMILES", "Tc"]), (sup3, ["SMILES", "Tg"]), (sup4, ["SMILES", "FFV"])]:
        for t in TARGETS:
            if t not in df.columns:
                df[t] = np.nan
        frames.append(df[["SMILES"] + TARGETS])
    return pd.concat(frames, ignore_index=True)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    supplement = load_supplement_data()

    full_train = pd.concat([train[["SMILES"] + TARGETS], supplement], ignore_index=True)
    # If the same SMILES appears multiple times with the same target filled, average them.
    full_train = full_train.groupby("SMILES", as_index=False).mean(numeric_only=True)

    print(f"train rows: {len(train)}  +supplement rows: {len(supplement)}  "
          f"-> combined unique SMILES: {len(full_train)}")
    for t in TARGETS:
        print(f"  {t}: {full_train[t].notna().sum()} labeled")

    print("\nFeaturizing combined train set...")
    X_full = featurize(full_train["SMILES"])
    print("Featurizing test set...")
    X_test = featurize(test["SMILES"])

    weights = property_weights(train, TARGETS)
    print("\nProperty weights:", {k: round(v, 4) for k, v in weights.items()})

    oof_preds = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
    test_preds = pd.DataFrame(index=test.index, columns=TARGETS, dtype=float)

    train_smiles_idx = {s: i for i, s in enumerate(full_train["SMILES"])}
    train_rows_in_full = train["SMILES"].map(train_smiles_idx).values

    for t in TARGETS:
        y_full = full_train[t]
        mask = y_full.notna()
        X_t = X_full.loc[mask]
        y_t = y_full.loc[mask]

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_maes = []
        oof_t = pd.Series(index=X_t.index, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(
                X_t.iloc[tr_idx], y_t.iloc[tr_idx],
                eval_set=[(X_t.iloc[va_idx], y_t.iloc[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            preds = model.predict(X_t.iloc[va_idx])
            oof_t.iloc[va_idx] = preds
            mae = np.abs(preds - y_t.iloc[va_idx]).mean()
            fold_maes.append(mae)

        print(f"{t}: CV MAE = {np.mean(fold_maes):.4f} (+/- {np.std(fold_maes):.4f}), n={mask.sum()}")

        # Map OOF predictions back onto original train rows for the global wMAE calc.
        oof_full = pd.Series(index=full_train.index, dtype=float)
        oof_full.loc[X_t.index] = oof_t
        oof_preds[t] = oof_full.reindex(train_rows_in_full).values

        # Final model on all available labels for this target, used for the test set.
        final_model = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": model.best_iteration_ or 500})
        final_model.fit(X_t, y_t)
        test_preds[t] = final_model.predict(X_test)

        top_feat = pd.Series(final_model.feature_importances_, index=X_t.columns)
        top_feat = top_feat.sort_values(ascending=False).head(5)
        print(f"  top features for {t}: {dict(top_feat)}")

    overall_wmae = wmae(train[TARGETS], oof_preds, weights, TARGETS)
    print(f"\nOverall CV wMAE (competition metric): {overall_wmae:.4f}")

    submission = pd.DataFrame({"id": test["id"]})
    for t in TARGETS:
        submission[t] = test_preds[t]
    submission.to_csv("submission.csv", index=False)
    print("\nWrote submission.csv")
    print(submission)


if __name__ == "__main__":
    main()
