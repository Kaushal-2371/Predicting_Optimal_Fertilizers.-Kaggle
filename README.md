# 🌱 Fertilizer Prediction — Kaggle Playground Series S5E6

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e6)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kaushalsahu123/fertilizer-prediction)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predict the right fertilizer for a given crop, soil type, and environmental conditions — and rank your top 3 guesses.**

Getting fertilizer recommendations wrong doesn't just hurt yields — it wastes resources and damages soil health. This notebook tackles the Kaggle Playground S5E6 challenge: a multi-class classification problem where submissions are scored on **MAP@3** (Mean Average Precision at 3), meaning the model outputs the top 3 most likely fertilizer names per row.

---

## 📋 Table of contents

- [Competition overview](#competition-overview)
- [Approach](#approach)
- [Pipeline](#pipeline)
- [Models](#models)
- [Submission format](#submission-format)
- [Run locally](#run-locally)
- [Repo structure](#repo-structure)
- [Future work](#future-work)
- [License](#license)

---

## Competition overview

| Detail | Value |
|---|---|
| Competition | Kaggle Playground Series — Season 5, Episode 6 |
| Task | Multi-class classification |
| Metric | MAP@3 (Mean Average Precision at 3) |
| Features | Temperature, Humidity, Moisture, Soil Type, Crop Type, NPK values |
| Target | Fertilizer Name (7 classes) |

The key twist: instead of predicting a single label, each submission contains **space-separated top-3 fertilizer predictions** per row, ordered by confidence. Getting the right answer anywhere in those 3 slots — especially in position 1 — is what the MAP@3 metric rewards.

---

## Approach

The pipeline is deliberately kept clean and readable:

1. **Label encoding** for `Soil Type`, `Crop Type`, and `Fertilizer Name` using sklearn's `LabelEncoder`
2. **Train two models** — Random Forest and XGBoost — to compare probability distributions
3. **Extract top-3 predictions** per row by ranking class probabilities with `np.argsort`
4. **Decode** encoded class indices back to human-readable fertilizer names
5. **Generate submission CSVs** for both models separately

---

## Pipeline

```
Raw CSV (train + test)
  └─ Label encoding (Soil Type, Crop Type, Fertilizer Name)
       └─ Feature / target split
            ├─ Random Forest (n_estimators=100)
            │    └─ predict_proba → argsort → top-3 → submission.csv
            └─ XGBoost (multi:softprob, 100 rounds)
                 └─ predict → argsort → top-3 → xgboost_submission.csv
```

---

## Models

### Random Forest
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
Trained on the full training set. Probabilities extracted via `predict_proba`, then top-3 class indices decoded back to fertilizer names.

### XGBoost
```python
params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42
}
xgb.train(params, dtrain, num_boost_round=100)
```
Uses `multi:softprob` which directly outputs per-class probabilities — a natural fit for MAP@3 scoring.

---

## Submission format

Each row in the submission contains the top-3 predicted fertilizer names, space-separated:

```
id,Fertilizer Name
1,Urea DAP 28-28
2,DAP Urea 28-28
3,14-35-14 28-28 Urea
...
```

---

## Run locally

```bash
git clone https://github.com/your-username/fertilizer-prediction.git
cd fertilizer-prediction
pip install -r requirements.txt
jupyter notebook fertilizer-prediction.ipynb
```

**requirements.txt**
```
pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

> To run on Kaggle, add the [Playground Series S5E6](https://www.kaggle.com/competitions/playground-series-s5e6) dataset and click *Run All*.

---

## Repo structure

```
fertilizer-prediction/
├── fertilizer-prediction.ipynb    # main notebook
├── submission.csv                 # Random Forest submission
├── xgboost_submission.csv         # XGBoost submission
├── requirements.txt
└── README.md
```

---

## Future work

- [ ] EDA — visualise feature distributions, class balance, NPK profiles per fertilizer
- [ ] Feature engineering — NPK ratios, soil-crop interaction terms
- [ ] Hyperparameter tuning — RandomizedSearchCV / Optuna for both models
- [ ] LightGBM — typically faster and competitive with XGBoost on tabular data
- [ ] Probability calibration — better-calibrated probabilities → better MAP@3
- [ ] Ensemble — average softmax probabilities from RF + XGBoost before ranking top-3
- [ ] Cross-validation — use OOF predictions instead of training on the full set blindly

---

## License

MIT © 2025 — free to use and adapt with attribution.

---

> **⬆️ Found this useful? An upvote on Kaggle goes a long way! Questions and suggestions welcome in the comments.**
