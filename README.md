# Predict2Protect: Corporate Bankruptcy Prediction

My submission to the **Predict2Protect hackathon (IIT (ISM) Dhanbad)**: predicting which U.S. public companies will go bankrupt from up to 20 years of financial ratios each.

**Tools:** Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · TensorFlow/Keras · Matplotlib · Seaborn

---

## Problem

Each company has one row of 18 financial ratios per fiscal year, from 1999 to 2018. The task is to classify each company as **alive** or **failed**. The competition scored on **Macro-F1**, with **recall on the bankrupt class** as the secondary metric, because missing a company that is about to fail is the costly mistake.

## Dataset at a Glance

| | |
|---|---|
| Training rows (company-years) | 62,789 |
| Unique companies | 7,176 (test set: 1,795 companies, 15,893 rows) |
| Features | 18 financial ratios (profitability, leverage, liquidity, turnover) + industry group |
| Fiscal years | 1999–2018 |
| Missing values | None |

## 1. Exploratory Data Analysis: Key Findings

- **Severe class imbalance:** 6,689 companies are alive and only **487 failed (6.7%)**, a ratio of 13.7 : 1. The imbalance was measured per company, not per row, because each company has many yearly rows.
- **Histories vary in length:** companies have between **1 and 20 years** of data (mean 8.75, median 7). The most common length is the full 20 years (929 companies). This shaped both the sequence models and the feature engineering.
- **Extreme differences in scale:** feature ranges run from about 4M up to **274M** (e.g. Sales to Receivables), with heavy outliers, so standardisation was needed.
- **Readable features:** the anonymous `X1`–`X18` columns were mapped to their financial meaning using the competition documentation (e.g. *Net Income / Total Assets*, *Total Liabilities / Total Assets*, *EBIT / Total Assets*).

## 2. Preprocessing and Feature Engineering

1. **Class balancing:** randomly undersampled the alive companies to match the 487 failed ones, at company level, giving 974 companies for training.
2. **Scaling:** fit `StandardScaler` on the training data only and reused it on the test set to avoid leakage.
3. **Sequence building:** grouped rows by company into `(years × 18)` arrays and padded them to 20 years for the LSTM and GRU models.
4. **Temporal feature extraction for tree models:** turned each company's history into **181 features**: mean, std, min, max, median, most recent value, oldest value, overall trend (last minus first), volatility (coefficient of variation) and the 3-year recent mean for every ratio.
5. **Caching:** saved the preprocessed data as compressed NumPy arrays (`preprocessed_data/`).

## 3. Modelling

I tested **30+ model configurations**:
- **Deep learning:** 3 LSTM and 3 GRU variants on the padded sequences, with masking.
- **Traditional ML:** LightGBM, XGBoost and Random Forest, each as a baseline and with hyperparameter tuning.
- **5-fold cross-validation** to check every model for overfitting.

| Model | Macro-F1 | Bankrupt recall | Notes |
|---|---|---|---|
| **Baseline LightGBM** | **≈ 0.76–0.78** | **≈ 0.78–0.79** | ✅ Chosen: best result among models that did not overfit |
| Baseline XGBoost | 0.749 | 0.773 | ✅ Stable |
| Deep LSTM (3 layers) | 0.721 | 0.680 | ✅ Stable, second-best family |
| Tuned XGBoost / LightGBM / RF | 0.74 (CV) | – | ⚠️ 99.5–100% validation accuracy but much lower CV scores, i.e. overfitting |

**Final model:** LightGBM (300 trees, max depth 6, learning rate 0.05, row and column subsampling of 0.8, L1 and L2 regularisation).

**Validation performance:** AUC-ROC **0.84–0.86**, Macro-F1 **0.76–0.78**, bankrupt recall **0.78–0.79** and specificity of about 0.73–0.77. The ranges reflect separate training runs (see `notebook.ipynb`, `submission_notebook.ipynb` and `models/FINAL_model_info.json`).

### Key Takeaways
- **Simpler beat more complex:** the tuned models looked perfect on validation but fell apart under cross-validation. A conservatively regularised baseline generalised best.
- **Engineered features beat raw sequences:** on a small dataset (974 companies after balancing), summary features of each company's history beat LSTMs trained on the full sequences.
- **Recall comes first:** the model was chosen to catch about 79% of bankruptcies, rather than to maximise accuracy.

## Repository Structure

| Path | Contents |
|---|---|
| `submission_notebook.ipynb` | **Start here:** a clean, end-to-end version with EDA, observations, preprocessing, the final model and submission |
| `notebook.ipynb` | Full experiment log: EDA, padding strategies, LSTM and GRU variants, hyperparameter tuning, CV analysis |
| `models/` | Saved LightGBM and LSTM models, feature names, final model metadata |
| `preprocessed_data/` | Cached preprocessed train and test arrays (`.npz`) |
| `submission.csv` | Final competition predictions |

## How to Run

```bash
git clone https://github.com/c0d3l0v3r-HeHe/Predict2Protect-Hackathon-ISMDhanbad.git
cd Predict2Protect-Hackathon-ISMDhanbad
pip install pandas numpy scikit-learn lightgbm xgboost tensorflow matplotlib seaborn jupyter
# place the competition files at Data/train.csv and Data/test.csv
jupyter notebook submission_notebook.ipynb
```

The raw competition data is not included in this repository.

## Limitations and Next Steps

- Validation was done on the **balanced** sample. Performance at the real 6.7% bankruptcy rate needs a separate held-out set that keeps the original imbalance.
- Use more of the data instead of undersampling (class weights or SMOTE), and tune the decision threshold for the best trade-off between recall and precision.
- Add SHAP explanations to show which financial ratios drive each prediction.
