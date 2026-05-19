"""Zero-shot pretrained transformer via Hugging Face pipeline.

Uses distilbert-base-uncased-finetuned-sst-2-english (movie review sentiment).
Domain mismatch is intentional — we remap POSITIVE -> ham (0), NEGATIVE -> spam (1).
"""
from __future__ import annotations

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
    y_pred = np.array(
        [0 if o["label"] == "POSITIVE" else 1 for o in outputs], dtype=int
    )
    return y_pred, time.time() - t0
