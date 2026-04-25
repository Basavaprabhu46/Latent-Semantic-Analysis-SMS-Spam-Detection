# LSA-SpamGuard: Latent Semantic Analysis for SMS Spam Detection

> SVD + TF-IDF based spam classifier using orthogonal projection and centroid classification — built from scratch with NumPy.

---

## Overview

This project applies **Latent Semantic Analysis (LSA)** to detect spam in SMS messages. It decomposes a TF-IDF document-term matrix using **Singular Value Decomposition (SVD)**, projects messages into a low-dimensional latent space, and classifies them via **cosine similarity to class centroids**.

Built as a mini-project for **UE24MA241B – Linear Algebra and Its Applications**.

---

## Technique

| Component | Detail |
|---|---|
| Vectorization | TF-IDF with bigrams, min/max document frequency filtering |
| Dimensionality Reduction | Truncated SVD (k=100 components) |
| Classification | Centroid-based cosine similarity |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Threshold Tuning | Adjustable decision boundary for precision/recall tradeoff |

---

## Key Fixes Over Baseline

- **No train/test split** → Added proper 80/20 stratified split
- **k=3 SVD components** → Increased to k=100 for meaningful variance capture
- **1-Nearest-Neighbor classifier** → Replaced with robust centroid-based classifier
- **No evaluation** → Added full classification metrics on held-out test set

---

## Project Structure

```
├── spam_filter_claudev2.ipynb   # Main notebook
├── spam.csv                     # Dataset (SMS Spam Collection)
└── README.md
```

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. Place `spam.csv` in the same directory as the notebook
2. Open `spam_filter.ipynb`
3. Run all cells top to bottom

Dataset format expected:

```
v1 (label): ham / spam
v2 (text):  message content
```

---

## How It Works

```
Raw SMS Text
    ↓
TF-IDF Vectorization  →  Document-Term Matrix A (train only)
    ↓
SVD: A ≈ U Σ Vᵀ  →  Latent Space (k=100)
    ↓
Project train + test into latent space
    ↓
Compute spam/ham centroids from training projections
    ↓
Classify test messages via cosine similarity + threshold
    ↓
Evaluate: Accuracy, F1, Confusion Matrix
```

---

## Results

Threshold tuning allows control over the spam/ham precision-recall tradeoff. Default threshold (`0.0`) gives balanced performance. Increase threshold to reduce false spam alerts; decrease to catch more spam.

---

## Course

**UE24MA241B – Linear Algebra and Its Applications**  
Concepts demonstrated: SVD, TF-IDF weighting, rank reduction, orthogonal projection, cosine similarity
