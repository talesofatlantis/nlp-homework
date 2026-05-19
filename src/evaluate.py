"""Build the side-by-side comparison table."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
    )


def build_results(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "accuracy", "macro_f1", "time_s"])
