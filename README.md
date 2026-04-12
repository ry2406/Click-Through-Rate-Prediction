# Click-Through Rate Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9558B2)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

Predicting mobile ad click-through rates using machine learning on the Avazu dataset.

## Project Overview

This project builds an end-to-end CTR prediction pipeline on the [Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) dataset from Kaggle. The goal is to predict whether a user will click on a mobile ad (`click = 1`) given features about the ad impression, device, site/app, and time of day.

The pipeline progresses through five modeling layers — from a Logistic Regression baseline to tuned GBDTs, a SMOTE experiment, a PyTorch DeepFM, and finally ensemble methods — with each decision driven by EDA findings and evaluation metrics.

## Dataset

- **Source:** [Kaggle — Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
- **Full training set:** ~40M rows, 24 columns
- **Sample used:** 10% stratified sample (~4M rows), preserving the original click rate (~17%)
- **Test set:** ~4.6M rows (Kaggle competition test set)

> The dataset files (`train.csv`, `test.csv`) are not included in this repository due to size. Download them from the Kaggle link above.

## Methodology

### EDA (Part 2)
- Class imbalance analysis → informed SMOTE vs. native class weighting comparison
- Hourly CTR patterns → kept raw `hour_of_day` for GBDT, added `is_weekend` binary feature
- Site vs. app traffic behavioral differences → motivated cross features
- High-cardinality feature distributions → justified target encoding over label encoding
- Cramér's V association analysis → guided feature selection decisions

### Feature Engineering (Part 3)
- **Target encoding** (5-fold to prevent leakage) for high-cardinality categoricals: `site_id`, `site_domain`, `app_id`, `app_domain`, `device_id`, `device_ip`, `device_model`
- **Cross features:** `site_category × hour_of_day`, `app_category × hour_of_day`, `site_category × device_type`
- **Count features:** `device_ip_count`, `device_id_count`, `site_id_count`, `app_id_count`
- Consistent train/test feature pipeline with proper handling of unseen categories

### Data Split (Part 4)
- Time-based train/validation/test split (not random) to respect temporal ordering
- StandardScaler fitted on training set only
- TimeSeriesSplit for cross-validation

### Modeling (Part 5)

| Layer | Model | Description |
|-------|-------|-------------|
| 1 | Logistic Regression | Linear baseline |
| 2 | LightGBM / XGBoost | Tuned via RandomizedSearchCV with `neg_log_loss` scoring |
| 3 | GBDT + SMOTE | Resampling experiment to test if SMOTE helps |
| 4 | DeepFM (PyTorch) | FM + DNN for second-order and higher-order feature interactions |
| 5 | Weighted Blend / Stacking | Ensemble of all base models |

### Analysis & Interpretation (Part 6)
- SHAP feature importance (TreeExplainer on best GBDT)
- Calibration curves + Platt Scaling / Isotonic Regression
- Error analysis on worst-predicted samples
- Segment analysis: site vs. app, time-of-day bands, device types

## Results

| Model | LogLoss | AUC-ROC | PR-AUC |
|-------|---------|---------|--------|
| **Weighted Blend** | **0.3921** | **0.7567** | **0.3907** |
| LightGBM | 0.3923 | 0.7563 | 0.3900 |
| XGBoost | 0.3924 | 0.7560 | 0.3892 |
| Logistic Regression | 0.4038 | 0.7344 | 0.3619 |
| DeepFM | 0.4083 | 0.7324 | 0.3573 |

*All metrics evaluated on the held-out test set.*

## Key Findings

1. **Feature engineering drove the largest gains.** Target encoding high-cardinality features was the single most impactful decision, confirmed by SHAP analysis.
2. **GBDTs outperformed both linear and deep approaches** on this tabular dataset with ~20 engineered features.
3. **SMOTE did not help.** It degraded LogLoss and AUC-ROC — GBDT native class weighting was more effective.
4. **Ensemble provided marginal but consistent improvement** over the best single model.
5. **GBDT models were already well-calibrated** — post-hoc calibration adjustments were minor.

## Repository Structure

```
├── CTR_Prediction.ipynb    # Main notebook (EDA → Feature Engineering → Modeling → Analysis)
├── clip_submission.py      # Post-processing script to clip predictions to [0, 1]
├── requirements.txt        # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run

1. Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data) and place them in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `CTR_Prediction.ipynb` from top to bottom.

> **Note:** The notebook uses a 10% stratified sample of the training data. A GPU is recommended for the DeepFM section (Part 5, Layer 4).

## Tech Stack

- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Deep Learning:** PyTorch (DeepFM implementation)
- **Interpretation:** SHAP

## Author

**Ruide Yin** — [GitHub](https://github.com/yinruide)
