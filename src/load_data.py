"""Download and load the SMS Spam Collection dataset."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
SMS_PATH = DATA_DIR / "sms.tsv"

MIRRORS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]


def download_sms(target: Path = SMS_PATH) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 50_000:
        return target
    last_err: Exception | None = None
    for url in MIRRORS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nlp-hw/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                target.write_bytes(r.read())
            return target
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All mirrors failed: {last_err}")


def load_sms(path: Path = SMS_PATH) -> pd.DataFrame:
    download_sms(path)
    df = pd.read_csv(
        path, sep="\t", header=None, names=["label", "text"], encoding="latin-1"
    )
    df = df.dropna(subset=["text", "label"]).copy()
    df["y"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return df[["text", "y"]].reset_index(drop=True)
