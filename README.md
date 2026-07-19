# 🏦 Bank Marketing — Term Deposit Predictor (Streamlit App)

A polished, banking-themed Streamlit app for the **Bank Marketing** ML
project. Enter a customer's profile and campaign details in the sidebar and
get an instant, color-coded prediction of whether they'll subscribe to a
term deposit — plus a confidence gauge, model performance dashboard, and
feature-importance breakdown.

## ✨ Features

- **Bank-themed UI** — navy & gold color palette, card-based layout
- **Sidebar input form** — grouped into logical sections (Customer Profile,
  Financial Details, Campaign Contact, Previous Campaign History)
- **Color-coded prediction box** — green for "Likely to Subscribe", red for
  "Likely to Decline", with a live confidence gauge
- **Model Insights tab** — accuracy/precision/recall/F1/ROC-AUC, feature
  importance chart, ROC curve, and confusion matrix
- **About tab** — explains the pipeline and modeling choices

## 📂 Project Files

```
├── app.py              # Main Streamlit application (UI)
├── model_utils.py       # Data loading, preprocessing & model training logic
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🚀 Getting Started

1. **Install dependencies** (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. Open the URL Streamlit prints (usually `http://localhost:8501`).

The first launch trains the Gradient Boosting model in the background
(cached with `st.cache_resource`, so it only happens once) — this typically
takes just a few seconds.

## 🧠 Modeling Pipeline

This mirrors the approach used in the project notebook:

- **Categorical encoding:** One-hot encoding (`drop_first=True`) for `job`,
  `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`,
  `poutcome`
- **Numeric features:** `age`, `balance`, `day`, `duration`, `campaign`,
  `pdays`, `previous` (used as-is)
- **Models compared in the notebook:** Logistic Regression, Random Forest,
  and **Gradient Boosting Classifier** — Gradient Boosting was selected as
  the final model for its best balance of accuracy/precision/recall/F1
- **Top predictive features:** `duration`, `poutcome`, `campaign`, `age`,
  `balance` (as identified via feature importance in the notebook)

## 📊 About the Training Data

The notebook this project is based on loads a local file
(`bank (1).csv`, from the UCI *Bank Marketing* dataset) that wasn't included
in the exported project. To keep this app **fully self-contained and
runnable out-of-the-box**, `model_utils.py` includes a synthetic data
generator that:

- Uses the **same columns and category values** as the real dataset
- Reproduces the **same directional relationships** found in the notebook's
  EDA (e.g., longer call `duration` and a `poutcome` of "success" increase
  subscription likelihood; more `campaign` contacts and existing loans
  decrease it)

### 🔁 To use your real dataset instead

1. Place your UCI-format `bank.csv` file at `data/bank.csv` (create the
   `data/` folder if needed) — it should have the columns: `age`, `job`,
   `marital`, `education`, `default`, `balance`, `housing`, `loan`,
   `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`,
   `poutcome`, `deposit`.
2. Restart the app — `model_utils.load_data()` automatically detects and
   loads it instead of generating synthetic data.
3. Clear the Streamlit cache if the app was already running: press **C**
   in the running app, or use the "⋮" menu → "Clear cache", then rerun.

## 🛠️ Customization Ideas

- Swap in your own trained model by editing `train_model()` in
  `model_utils.py`
- Add authentication or role-based views for internal bank use
- Log predictions to a database/CSV for campaign tracking
- Add a batch-prediction tab (upload a CSV of customers, download scored
  results)

## 📦 Requirements

See `requirements.txt`:
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `plotly`
