"""Build the side-by-side comparison table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


def build_results(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "accuracy", "macro_f1", "time_s"])
