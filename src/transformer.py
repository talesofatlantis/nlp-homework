"""Zero-shot pretrained transformer via Hugging Face pipeline.

Sentiment model misapplied to spam: POSITIVE -> ham (0), NEGATIVE -> spam (1).
The domain mismatch is intentional; the reflection discusses it.
"""
import time

import numpy as np


def run_transformer(texts_test: np.ndarray) -> tuple[np.ndarray, float]:
    from transformers import pipeline

    t0 = time.time()
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    outputs = clf(list(texts_test), truncation=True, max_length=128, batch_size=32)
    y_pred = np.fromiter(
        (0 if o["label"] == "POSITIVE" else 1 for o in outputs),
        dtype=int,
        count=len(outputs),
    )
    return y_pred, time.time() - t0
