# BOWpipe — Bag-of-Words NLP Pipeline

A lightweight text classification pipeline built with Python that supports unigram/n-gram encoding, optional TF-IDF transformation, and Gaussian Naive Bayes classification. Demonstrated on the IMDb sentiment analysis dataset.

---

## Overview

`BOWpipe` tokenises and encodes a corpus of text documents into numerical feature vectors using a bag-of-n-grams approach. These vectors can then be fed into a `Gaussian_Classifier` for binary sentiment classification (positive / negative).

---

## Files

| File | Description |
|---|---|
| `bow_pipeline.py` | Core pipeline classes: `BOWpipe` and `Gaussian_Classifier` |
| `BOWpipe.ipynb` | Jupyter notebook demonstrating the pipeline on the IMDb dataset |

---

## Requirements

- Python 3.x
- `spacy` (with the blank English model)
- `scikit-learn`
- `numpy`
- `pandas`

Install dependencies:
```bash
pip install spacy scikit-learn numpy pandas
python -m spacy download en_core_web_sm
```

---

## Usage

### 1. Encoding a Corpus

```python
from bow_pipeline import BOWpipe

# Unigram (default)
pipe = BOWpipe(texts)

# Trigram with TF-IDF
pipe = BOWpipe(texts, n=3, tfidf=True)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | list/Series of strings | — | The text corpus to encode |
| `n` | int | `1` | N-gram size (1 = unigram, 2 = bigram, etc.) |
| `tfidf` | bool | `False` | Apply TF-IDF transformation after encoding |

**Useful attributes on the resulting object:**

- `.encoding` — NumPy array of shape `(n_docs, vocab_size)`
- `.vocab` — Dictionary mapping n-gram tuples to column indices
- `.shape` — Shape of the encoding matrix
- `.nbytes` — Memory footprint in bytes

### 2. Training and Evaluating a Classifier

```python
from bow_pipeline import Gaussian_Classifier

clf = Gaussian_Classifier("Unigram", pipe.encoding, labels)
clf.evaluate_model()
# → Unigram: Acc. - 57.00%, Prec. - 58.00%, Rec. - 62.00%
```

The classifier uses a 67/33 train/test split (`random_state=42`).

---

## Pipeline Steps

1. **Tokenisation** — spaCy blank English model splits text into tokens.
2. **Stop-word removal** — Tokens flagged as stop words are dropped.
3. **N-gram extraction** — Sliding window builds the vocabulary and count vectors.
4. **TF-IDF (optional)** — Term Frequency × Inverse Document Frequency weighting, followed by L2 normalisation.
5. **Classification** — Gaussian Naive Bayes trained on the encoded vectors.

---

## Benchmark Results (IMDb, 500 samples)

| Configuration | Accuracy | Precision | Recall |
|---|---|---|---|
| Unigram | 57% | 58% | 62% |
| Unigram + TF-IDF | 58% | 59% | 64% |
| Trigram | 59% | 69% | 39% |
| Trigram + TF-IDF | 58% | 63% | 42% |

**Memory footprint (500 documents):**

| Configuration | Matrix Size | Memory |
|---|---|---|
| Unigram | 500 × 14,472 | ~55 MB |
| Trigram | 500 × 67,437 | ~257 MB |

> Note: TF-IDF does not change matrix dimensions, only the values.

---

## Notes

- Trigrams capture more context but create significantly larger (and sparser) feature matrices.
- TF-IDF tends to improve precision slightly by down-weighting very common terms.
- Gaussian Naive Bayes assumes feature independence and normally distributed values, which is a rough fit for sparse count vectors — more sophisticated classifiers (e.g. logistic regression, SVM) would likely yield better results.
- The sample size of 500 is small; performance will improve with the full IMDb training set.
