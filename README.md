# Reproducible NLP Pipeline in a Container

HSLU FS26 - Machine Learning, NLP and Toolchain (Prof. Dr. Marcel Blattner)
Homework SW05, slim track.

Compares a classical **TF-IDF + Logistic Regression** baseline against a
pretrained, zero-shot **Hugging Face transformer**
(`distilbert-base-uncased-finetuned-sst-2-english`) on the
**SMS Spam Collection**. Same train/test split, same metrics,
side-by-side numbers - packaged so a stranger reproduces it in two
commands.

## Reproduce (the two commands)

```bash
docker build -t nlp-hw .
docker run --rm nlp-hw
```

The container downloads the dataset and transformer weights on first run,
trains the classical model, runs zero-shot inference, prints the
comparison table to stdout, and writes `results/results.csv`.

## Setup from scratch

```bash
# 1. Clone
git clone https://github.com/talesofatlantis/nlp-homework.git
cd nlp-homework

# 2. Build the image (~2-3 min on first run; cached afterwards)
docker build -t nlp-hw .

# 3. Run the comparison (~2-3 min on first run for the HF model download)
docker run --rm nlp-hw
```

Expected stdout ends with:

```
                        model  accuracy  macro_f1  time_s
              TF-IDF + LogReg     0.967    0.9204    0.12
 DistilBERT SST-2 (zero-shot)     0.450    0.4256  137.36
```

Numbers will match exactly because `SEED = 42` is fixed in
[`src/main.py`](src/main.py). Wall-clock times will differ.

**Requirements on the host:** Docker (tested with 29.x). No Python,
no GPU, no network access beyond the dataset mirror and the Hugging Face
hub on first run.

## Run locally without Docker (optional)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

Python 3.11 is required - the pinned `torch==2.4.1` wheel does not exist
for 3.13.

## Results

| Approach                          | Accuracy | Macro F1 | Time (s) |
| --------------------------------- | -------- | -------- | -------- |
| TF-IDF + Logistic Regression      | 0.967    | 0.920    | 0.12     |
| DistilBERT SST-2 (zero-shot)      | 0.450    | 0.426    | 137.36   |

The classical baseline wins by 50 percentage points. The transformer
loses because it is a sentiment model misapplied to spam - see
[`reflection.md`](reflection.md) for why this is the most interesting
result the slide deck could have asked for.

## Dataset

- **Name:** SMS Spam Collection (v1)
- **Source:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011),
  *Contributions to the Study of SMS Spam Filtering: New Collection and
  Results*, ACM Symposium on Document Engineering.
- **Original host:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Mirror used at runtime:** `raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv`
- **License:** Public, for research and educational use (UCI terms)
- **Date downloaded:** 2026-05-19
- **Size:** 5,572 labelled SMS messages (4,825 ham / 747 spam)

Not committed to this repository - fetched at runtime into `data/raw/sms.tsv`.

## Layout

```
nlp-homework/
  README.md
  reflection.md
  Dockerfile
  requirements.txt
  .gitignore
  src/
    __init__.py
    load_data.py        # download + load SMS Spam Collection
    classical.py        # TF-IDF + LogisticRegression
    transformer.py      # HF pipeline, zero-shot, label remap
    evaluate.py         # accuracy + macro F1
    main.py             # entry point, fixed seed, prints + saves CSV
  results/
    results.csv         # written by main.py
```

## Reproducibility notes

- Seed: `SEED = 42`, applied to `random`, `numpy`, the stratified split,
  and Logistic Regression.
- Train/test split: stratified by class, 4,000 train / 1,000 test held
  out. (The slide suggests 5,000 + 1,000; the dataset is too small for
  that, so the budget was reduced symmetrically.)
- Test set is used **once**, after both models are frozen.
- Library versions are pinned in [`requirements.txt`](requirements.txt).
- Python interpreter pinned via `FROM python:3.11-slim` in
  [`Dockerfile`](Dockerfile).

## Reflection

Same content as [`reflection.md`](reflection.md) / [`reflection.pdf`](reflection.pdf).

### 1. Which approach won, and when is a classical method still worth using?

Classical won by a mile: **0.967 / 0.920** (acc / macro F1) in 0.12 s, vs. the transformer's **0.450 / 0.426** in 137 s. Always predicting "ham" would have scored 0.866 - so the transformer was worse than nothing.

DistilBERT isn't a bad model, I just used it wrong. It's a sentiment classifier, and I mapped `POSITIVE → ham`, `NEGATIVE → spam`. That only works if spam sounds negative, which it doesn't - spam is usually cheerful ("You won!"), and short legit messages ("ok") read as negative.

A classical method is still worth it when the task is clear, the labels are clean, and a few thousand examples exist. TF-IDF + LogReg learns the obvious signal words ("free", "txt", "claim") in 100 ms. A transformer only earns its cost if the pretraining domain matches, or you fine-tune.

### 2. What's in your Docker image that wouldn't be in a venv?

- **The OS.** `python:3.11-slim` pins Debian bookworm and its glibc. PyTorch wheels are compiled against specific glibc versions - the same `torch==2.4.1` can `ImportError` on an older Linux.
- **Python itself.** `requirements.txt` doesn't pin the interpreter. My first run died on Python 3.13 because `torch==2.4.1` has no wheel for it. `FROM python:3.11-slim` fixes that.
- **The layout.** `python -m src.main` always finds the package because `COPY src/ ./src/` puts it in a known place.

The Python pin was the one that actually bit me, and it's invisible from `requirements.txt` alone.

### 3. One thing that broke, one thing that surprised you.

**Broke:** `train_test_split(train_size=5000, test_size=1000)` crashed because SMS Spam only has 5,572 messages, not 6,000. I copied the split sizes from the slides without checking the dataset. Dropped to 4,000 + 1,000.

**Surprised:** how badly the transformer lost. I expected ±5 points; got -50. The lesson: a leaderboard number tells you nothing about whether a model fits *your* data.
