"""Competition weighted MAE (wMAE) metric.

wMAE = sum over rows i, properties k present in row i of:
    w_k * |pred - true|  / (number of (i,k) terms)

w_k = (1/r_k) * sqrt(1/n_k) / sum_j(1/r_j * sqrt(1/n_j)) * K

where r_k is the value range of property k (on the full available labels)
and n_k is the number of available labels for property k.
"""
import numpy as np
import pandas as pd

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]


def property_weights(train_df: pd.DataFrame, targets=TARGETS) -> dict:
    K = len(targets)
    n = {t: train_df[t].notna().sum() for t in targets}
    r = {t: train_df[t].max() - train_df[t].min() for t in targets}
    raw = {t: (1.0 / r[t]) * np.sqrt(1.0 / n[t]) for t in targets}
    norm = sum(raw.values())
    return {t: raw[t] / norm * K for t in targets}


def wmae(y_true: pd.DataFrame, y_pred: pd.DataFrame, weights: dict, targets=TARGETS) -> float:
    """y_true/y_pred: DataFrames with target columns, NaN where label missing."""
    total_err = 0.0
    total_n = 0
    for t in targets:
        mask = y_true[t].notna()
        if mask.sum() == 0:
            continue
        err = (y_true.loc[mask, t] - y_pred.loc[mask, t]).abs().sum()
        total_err += weights[t] * err
        total_n += mask.sum()
    return total_err / total_n
