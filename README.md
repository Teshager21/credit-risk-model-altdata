# 📊 Credit Risk Model with Alternative Data

**An end-to-end credit scoring system using e-commerce transaction data, RFM segmentation, machine learning, and MLOps.**

---

## 🧠 Credit Scoring Business Understanding


### 1️⃣ How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The **Basel II Capital Accord** mandates that financial institutions align their capital reserves with the actual level of **credit risk** they face. This requires banks to **quantify risk exposures** reliably and defendably. Consequently, models must not only be accurate but also **interpretable, transparent, and auditable**.

In this context, simple models like **logistic regression with Weight of Evidence (WoE) encoding** are preferred because:
- They offer **clear explanations** for decision-making.
- They support **regulatory reporting** and internal governance.
- They reduce the risk of **model uncertainty** and **bias claims** in audits.

Without interpretability, complex models (even if more accurate) may **fail regulatory scrutiny** or be non-actionable for internal risk teams.

---

### 2️⃣ Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks?

In traditional credit modeling, the **target variable** is a binary label indicating whether a borrower defaulted. However, our dataset lacks this label. To address this, we engineer a **proxy variable** using behavioral signals such as **Recency, Frequency, and Monetary (RFM)** activity to infer disengaged (potentially high-risk) customers.

This is necessary to:
- Enable **supervised learning** in the absence of ground truth.
- Leverage **alternative data** from e-commerce platforms for financial modeling.

However, the use of a proxy introduces **business risks**:
- **Label Noise**: Proxy labels may **misclassify customers**, leading to biased predictions.
- **Overfitting to behavior patterns**, not actual financial defaults.
- **Missed revenue** from falsely labeled good customers (false negatives).
- **Increased exposure** from wrongly approved risky customers (false positives).

Thus, clear **documentation, validation**, and **human-in-the-loop verification** are essential before using the model in production.

---

### 3️⃣ What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect                     | Logistic Regression + WoE              | Gradient Boosting / Complex Models      |
|---------------------------|----------------------------------------|-----------------------------------------|
| **Interpretability**      | ✅ High — easy to explain               | ❌ Low — requires tools like SHAP/LIME   |
| **Regulatory Acceptance** | ✅ Strong — aligns with Basel II        | ⚠️ Requires rigorous documentation       |
| **Performance**           | ⚠️ Moderate — linear assumptions        | ✅ High — captures non-linear patterns   |
| **Speed & Simplicity**    | ✅ Fast to train, easy to deploy        | ⚠️ Slower training and deployment        |
| **Use Case Fit**          | ✅ Ideal for regulated environments     | ✅ Ideal for maximizing accuracy         |

In regulated settings like banking, it’s often better to **start with interpretable models**, and justify any use of complex models with **robust validation and explainability frameworks**.

---

📌 _Summary_: The Basel II Accord compels us to strike a balance between **accuracy** and **interpretability**. Given the absence of default labels, proxy modeling is necessary but must be approached cautiously. In regulated contexts, **simple models offer clarity and compliance**, while complex models offer accuracy at the cost of oversight burdens.


### 📌 Why Interpretable Models Matter
Under the **Basel II Capital Accord**, financial institutions must quantify credit risk and maintain adequate capital. This creates a need for **interpretable, transparent, and auditable models**. Regulatory compliance prefers models where decisions can be explained (e.g., using Weight of Evidence or logistic regression), rather than opaque black-box models.

### 📌 Why We Need a Proxy Default Variable
The dataset lacks a direct **loan default indicator**. We create a **proxy target** using customer disengagement signals via **RFM (Recency, Frequency, Monetary) analysis**. However, this introduces potential **label noise**, and predictions may suffer from **false positives** or **missed defaults**, impacting lending decisions.

---

## 🗂 Project Structure

```
credit-risk-model-altdata/
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── data/
│   ├── raw/                       # Raw Xente dataset
│   └── processed/                 # Cleaned and transformed data
├── notebooks/
│   └── 1.0-eda.ipynb              # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training & tracking
│   ├── predict.py                 # Inference script
│   └── api/
│       ├── main.py                # FastAPI app
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Key Features

- 📦 **Alternative data modeling** using e-commerce transactions
- 🧮 **RFM Segmentation** to create a proxy default label
- 🏗️ **Feature Engineering Pipeline** using `sklearn.pipeline`
- 🤖 **ML Models**: Logistic Regression, Random Forest, Gradient Boosting
- 🧪 **MLFlow tracking** & experiment versioning
- 🧪 **Unit testing** with Pytest
- 🚀 **API Deployment** using FastAPI + Docker
- 🔁 **CI/CD**: GitHub Actions with Linting + Testing

---

## 🚧 Tasks Overview

| Task | Description |
|------|-------------|
| ✅ Task 1 | Business understanding and Basel II interpretation |
| ✅ Task 2 | EDA on transactional features and customer behavior |
| ✅ Task 3 | Feature engineering (RFM, encoding, scaling) |
| ✅ Task 4 | Proxy target creation using K-Means clustering |
| ✅ Task 5 | Model training, tuning, MLFlow logging, testing |
| ✅ Task 6 | API deployment, Docker, CI/CD with GitHub Actions |

---

## 🔍 Tech Stack

- Python 3.10+
- Pandas, NumPy, Scikit-learn
- XGBoost / LightGBM
- MLFlow, FastAPI
- Docker, Pytest, Flake8
- GitHub Actions for CI

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/credit-risk-model-altdata.git
cd credit-risk-model-altdata

# Create virtual env and install requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run model training
python src/train.py

# Run FastAPI app
uvicorn src.api.main:app --reload

# Run tests
pytest tests/
```

---

## 📎 References & Learning

- [Basel II Capital Accord Summary](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [RFM Customer Segmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860138/)
- [WoE & IV Scoring Guide](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [MLFlow Model Tracking](https://mlflow.org/)
- [Dataset](https://www.kaggle.com/datasets/atwine/xente-challenge)

---

## 👥 Team

> Project by Analytics Engineer at Bati Bank – in collaboration with Xente eCommerce.

Tutors: Mahlet | Rediet | Kerod | Rehmet
Challenge Week: B5W5 | Final Submission Due: July 1, 2025

---

## 📄 License

This project is for academic and educational purposes only. Commercial use of the models or insights should comply with regulatory standards and institutional approval.

---
