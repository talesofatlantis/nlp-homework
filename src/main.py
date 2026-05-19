"""Load data, run both models on the same split, print + save results."""
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.classical import run_classical
from src.evaluate import build_results, score
from src.load_data import load_sms
from src.transformer import run_transformer

SEED = 42
# SMS Spam has 5,572 messages; slide suggests 5k+1k. We use 4k+1k to fit.
N_TRAIN, N_TEST = 4_000, 1_000
RESULTS_PATH = Path("results/results.csv")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_sms()
    print(f"Loaded {len(df):,} SMS messages "
          f"({int(df['y'].sum())} spam / {int((1 - df['y']).sum())} ham)")

    X = np.asarray(df["text"].tolist())
    y = np.asarray(df["y"].tolist(), dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_TRAIN, test_size=N_TEST,
        random_state=SEED, stratify=y,
    )
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}\n")

    print("[1/2] Classical: TF-IDF + Logistic Regression ...")
    y_pred_c, t_c = run_classical(X_train, y_train, X_test, seed=SEED)
    acc_c, f1_c = score(y_test, y_pred_c)
    print(f"      acc={acc_c:.3f}  macro_f1={f1_c:.3f}  time={t_c:.2f}s\n")

    print("[2/2] Pretrained transformer (DistilBERT SST-2, zero-shot) ...")
    y_pred_t, t_t = run_transformer(X_test)
    acc_t, f1_t = score(y_test, y_pred_t)
    print(f"      acc={acc_t:.3f}  macro_f1={f1_t:.3f}  time={t_t:.2f}s\n")

    results = build_results([
        {"model": "TF-IDF + LogReg",
         "accuracy": round(acc_c, 4), "macro_f1": round(f1_c, 4),
         "time_s": round(t_c, 2)},
        {"model": "DistilBERT SST-2 (zero-shot)",
         "accuracy": round(acc_t, 4), "macro_f1": round(f1_t, 4),
         "time_s": round(t_t, 2)},
    ])
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
