"""Quick exploratory data analysis for the polymer dataset.

Prints target stats/co-occurrence/correlations and saves two plots to
eda_outputs/.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    print("=== Basic stats per target ===")
    print(train[TARGETS].describe().T)

    print("\n=== How many of the 5 targets are filled per row ===")
    n_filled = train[TARGETS].notna().sum(axis=1)
    print(n_filled.value_counts().sort_index())

    print("\n=== Correlation between targets (pairwise, only overlapping rows) ===")
    print(train[TARGETS].corr())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, t in zip(axes.flat, TARGETS):
        train[t].dropna().hist(bins=40, ax=ax)
        ax.set_title(t)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_distributions.png", dpi=120)
    print(f"\nSaved {OUT_DIR / 'target_distributions.png'}")

    lengths = train["SMILES"].str.len()
    plt.figure(figsize=(6, 4))
    lengths.hist(bins=40)
    plt.title("SMILES string length distribution")
    plt.savefig(OUT_DIR / "smiles_length.png", dpi=120)
    print(f"Saved {OUT_DIR / 'smiles_length.png'}")
    print("\nSMILES length stats:\n", lengths.describe())


if __name__ == "__main__":
    main()
