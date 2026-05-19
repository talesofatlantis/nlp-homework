# Reproducible NLP Pipeline in a Container — Slim Version

Compares a classical TF-IDF + Logistic Regression baseline against a
pretrained, zero-shot Hugging Face transformer (`distilbert-base-uncased-finetuned-sst-2-english`)
on the **SMS Spam Collection** dataset. Same train/test split, same metrics,
side-by-side numbers — packaged so a stranger can reproduce it with two
commands.

## Reproduce

```bash
docker build -t nlp-hw .
docker run --rm nlp-hw
```

The container downloads the dataset and the transformer weights on first
run, trains the classical model, runs zero-shot inference with the
transformer, prints the comparison table, and writes `results/results.csv`.

## Results

| Approach                          | Accuracy | Macro F1 | Time (s) |
| --------------------------------- | -------- | -------- | -------- |
| TF-IDF + Logistic Regression      | 0.967    | 0.920    | 0.12     |
| DistilBERT SST-2 (zero-shot)      | 0.450    | 0.426    | 137.36   |

Numbers from a CPU run on macOS (Apple Silicon). The transformer loses
badly here — see [`reflection.md`](reflection.md) for why this result is
informative rather than embarrassing.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
~5,572 labelled SMS messages (`ham` / `spam`), public domain, Almeida et al.
The repo downloads it from a GitHub mirror at runtime — it is not committed.

## Layout

```
nlp-homework/
  README.md
  reflection.md
  Dockerfile
  requirements.txt
  .gitignore
  src/
    load_data.py     download + load SMS Spam Collection
    classical.py     TF-IDF + LogisticRegression
    transformer.py   HF pipeline, zero-shot sentiment, label remap
    evaluate.py      accuracy + macro F1
    main.py          orchestrates everything
  results/
    results.csv      written by main.py
```

## Local run without Docker

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```
