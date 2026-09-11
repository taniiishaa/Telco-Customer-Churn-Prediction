# 📉 Telco Churn Intelligence

### Predicting which customers may leave — and turning those predictions into retention decisions.

> **Can customer behavior reveal who is likely to churn before they actually leave?**

This project explores that question using the **Telco Customer Churn dataset** and a complete machine-learning workflow.

Rather than stopping at *“Can we predict churn?”*, the project goes one step further:

**Which customer characteristics are associated with churn, how well can different models identify at-risk customers, and how can those insights support retention strategies?**

---

## 🎯 The Business Problem

Customer churn is more than a classification problem.

For a telecom company, losing a customer can mean:

```text
Customer signs up
       │
       ▼
Customer uses the service
       │
       ▼
Customer becomes dissatisfied / disengaged
       │
       ▼
       ┌───────────────┐
       │    CHURN?     │
       └───────┬───────┘
               │
        ┌──────┴──────┐
        ▼             ▼
       YES            NO
        │             │
        ▼             ▼
Revenue lost      Customer retained
```

The goal of this project is therefore to identify patterns associated with the **YES** branch early enough to support targeted retention efforts.

---

# 🔍 From Customer Data → Churn Risk

The project follows an end-to-end machine-learning pipeline:

```text
                    TELCO CUSTOMER DATA
                            │
                            ▼
                   ┌─────────────────┐
                   │ Data Preparation│
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      EDA        │
                   │ Explore Churn   │
                   │ & Relationships │
                   └────────┬────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Preparation │
                 │ Scale + Encode      │
                 └──────────┬──────────┘
                            │
                            ▼
                    Train / Test Split
                       80% / 20%
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        Logistic Regression      Random Forest
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Model Evaluation
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
              ROC AUC            Classification
                                   Report
                            │
                            ▼
                    Feature Importance
                            │
                            ▼
                  RETENTION INSIGHTS
```

The result is a workflow that connects **machine learning with business interpretation**.

---

# 🧹 Step 1 — Preparing the Data

The raw dataset contains a mixture of:

* numerical variables
* categorical variables
* customer identifiers
* the target variable `Churn`

One important cleaning step involves `TotalCharges`.

It is converted into a numeric column:

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

The irrelevant `customerID` column is then removed.

The target is converted into a machine-learning-friendly binary representation:

```text
Churn = Yes  →  1
Churn = No   →  0
```

---

# 🧩 Step 2 — Understanding the Features

The dataset contains different kinds of customer information.

### Numerical features

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

### Categorical features

Examples include:

```text
gender
Partner
Dependents
PhoneService
InternetService
Contract
PaymentMethod
...
```

That mixture creates an important preprocessing challenge:

> **Numerical and categorical variables cannot simply be treated the same way.**

---

# ⚙️ Step 3 — The Preprocessing Pipeline

Instead of manually transforming columns one by one, the project uses Scikit-learn's **ColumnTransformer**.

```text
                    Customer Features
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Numerical Data              Categorical Data
             │                           │
             ▼                           ▼
      StandardScaler              OneHotEncoder
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Model-Ready Data
```

### Numerical transformation

Numerical features are standardized using:

```python
StandardScaler()
```

### Categorical transformation

Categorical features are converted into machine-readable variables using:

```python
OneHotEncoder(handle_unknown='ignore')
```

The entire transformation is placed inside the model pipeline.

That means preprocessing and modeling travel together instead of becoming disconnected notebook steps.

---

# 🧪 Step 4 — Keeping the Test Data Honest

The dataset is split into:

```text
80% ───────────── Training
20% ───────────── Testing
```

The split uses:

```python
stratify=Y
```

This is particularly useful because churn datasets contain an imbalance between customers who stay and customers who leave.

```text
                 COMPLETE DATASET
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        80% Training         20% Testing
              │                   │
              │              Unseen Data
              │                   │
              └─────────┬─────────┘
                        ▼
                   Evaluate Model
```

A fixed `random_state=42` is also used to keep the experiment reproducible.

---

# 🤖 Two Models. One Question.

Instead of relying on a single algorithm, this project compares two fundamentally different approaches.

## 01 — Logistic Regression

A classic classification model that provides a strong baseline for binary prediction.

```text
Customer Features
       │
       ▼
Preprocessing
       │
       ▼
Logistic Regression
       │
       ▼
Churn Probability
       │
       ▼
Churn / No Churn
```

It is relatively simple and interpretable, making it useful as a baseline.

---

## 02 — Random Forest 🌲

Random Forest combines multiple decision trees to make a stronger prediction.

Conceptually:

```text
              Customer
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     Tree 1    Tree 2    Tree 3
       │         │         │
       ▼         ▼         ▼
     Churn?    Churn?    Churn?
       │         │         │
       └─────────┼─────────┘
                 ▼
           Forest Decision
                 │
                 ▼
            Churn Risk
```

The project uses:

```python
RandomForestClassifier(
    random_state=42,
    n_estimators=100
)
```

---

# 📊 How Are the Models Compared?

The project doesn't judge the models using accuracy alone.

It generates:

* Classification Report
* ROC AUC Score
* ROC Curve

The ROC curve provides a visual comparison of how effectively the models distinguish between churn and non-churn customers across different classification thresholds.

```text
True Positive Rate
       │
   1.0 │                 ╭──────
       │              ╭──╯
       │           ╭──╯
       │        ╭──╯
       │     ╭──╯
       │  ╭──╯
   0.0 └──────────────────────────
       0.0                    1.0
             False Positive Rate
```

The notebook compares the ROC curves for:

```text
Logistic Regression
        VS
Random Forest
```

The existing project analysis reports **Random Forest as the stronger model by ROC AUC**, with an AUC around **0.84**.

---

# 🌲 Looking Beyond Prediction

A prediction tells us **who may churn**.

But a business also needs to know:

> **Why might these customers be at risk?**

That's where feature importance becomes useful.

The Random Forest model's transformed feature importances are extracted and ranked.

```text
                 Random Forest
                       │
                       ▼
              Feature Importance
                       │
              ┌────────┴────────┐
              ▼                 ▼
          High Impact        Lower Impact
              │
              ▼
       Investigate Drivers
              │
              ▼
        Retention Actions
```

The notebook visualizes the **top 10 features** contributing to the model's predictions.

---

# 💡 From Features to Business Decisions

The analysis highlights several important churn-related patterns.

### 📅 Month-to-Month Contracts

Customers without long-term contracts can represent a higher churn-risk segment.

**Potential strategy:**

```text
Month-to-Month Customer
          │
          ▼
Identify At-Risk Segment
          │
          ▼
Offer Contract Upgrade
          │
          ▼
Loyalty / Pricing Incentive
          │
          ▼
Improve Retention
```

---

### 🌐 Fiber Optic Customers

The analysis identifies fiber-optic service as an important churn-related feature.

This doesn't automatically mean that fiber service *causes* churn.

Instead, it signals a segment worth investigating.

Possible business questions include:

* Are pricing expectations higher?
* Are customers experiencing service issues?
* Is the perceived value matching the price?
* Are competitors offering better packages?

---

### 🆕 New Customers

Low-tenure customers are another important risk signal.

This suggests the early customer journey may be particularly important.

```text
New Customer
     │
     ▼
Onboarding
     │
     ▼
Early Experience
     │
     ├──── Positive ────► Stay
     │
     └──── Negative ────► Churn Risk
```

A targeted onboarding strategy could therefore be more effective than waiting until a customer is already considering cancellation.

---

# 🧠 Prediction → Action

The central idea of the project can be summarized as:

```text
                CUSTOMER DATA
                     │
                     ▼
              MACHINE LEARNING
                     │
                     ▼
               CHURN RISK
                     │
                     ▼
              "WHY AT RISK?"
                     │
                     ▼
              FEATURE ANALYSIS
                     │
                     ▼
             BUSINESS INSIGHT
                     │
                     ▼
             RETENTION ACTION
```

That's the difference between simply building a classifier and using machine learning for a business problem.

---

# 🗂️ Repository Structure

```text
Telco-Customer-Churn-Prediction-and-Retention-System/
│
├── 📂 Customer_Churn_Prediction/
│   │
│   ├── 📓 Customer_Churn_Prediction.ipynb
│   │
│   ├── 📊 WA_Fn-UseC_-Telco-Customer-Churn (1).csv
│   │
│   └── 📄 requirements.txt
│
└── 📄 README.md
```

### `Customer_Churn_Prediction.ipynb`

The complete analysis and modeling workflow.

### Dataset CSV

The Telco Customer Churn dataset used for the experiment.

### `requirements.txt`

Project dependencies.

---

# 🛠️ Technology Stack

| Area              | Technology                                   |
| ----------------- | -------------------------------------------- |
| Language          | 🐍 Python                                    |
| Data Manipulation | 🐼 Pandas, NumPy                             |
| Visualization     | 📊 Matplotlib, Seaborn                       |
| Preprocessing     | ⚙️ Scikit-learn                              |
| Baseline Model    | 📈 Logistic Regression                       |
| Ensemble Model    | 🌲 Random Forest                             |
| Evaluation        | 🎯 Classification Report, ROC AUC, ROC Curve |
| Environment       | 📓 Jupyter Notebook / Google Colab           |

---

# ▶️ Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/taniiishaa/Telco-Customer-Churn-Prediction-and-Retention-System.git
cd Telco-Customer-Churn-Prediction-and-Retention-System
```

### 2. Install dependencies

```bash
pip install -r Customer_Churn_Prediction/requirements.txt
```

### 3. Open the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Customer_Churn_Prediction/
└── Customer_Churn_Prediction.ipynb
```

Alternatively, the notebook can be opened in **Google Colab**.

### 4. Run the cells

Execute the notebook sequentially to reproduce the preprocessing, training, evaluation, ROC comparison, and feature-importance analysis.

---

# 📌 Key Takeaways

### From the machine-learning side

* Built a complete binary classification workflow.
* Compared **Logistic Regression** with **Random Forest**.
* Used `ColumnTransformer` for mixed feature types.
* Applied `StandardScaler` to numerical variables.
* Applied `OneHotEncoder` to categorical variables.
* Used stratified train-test splitting.
* Evaluated models using ROC AUC and classification metrics.
* Extracted Random Forest feature importance.

### From the business side

* Churn prediction is only the beginning.
* Customer segments can reveal where retention efforts should focus.
* Contract type, internet service, and tenure emerged as important areas of investigation.
* Model outputs can be translated into targeted retention strategies.

---

# ⚠️ Important Interpretation Note

Feature importance shows that a variable is useful to the model's predictions.

It does **not** prove that the variable causes customers to churn.

For example:

```text
Feature is important
        ≠
Feature causes churn
```

A production retention system would require additional validation, monitoring, customer-level risk analysis, cost/benefit analysis, and careful evaluation before automated business decisions were made.

---

# 🚀 What Could Come Next?

The current notebook establishes the prediction and analysis foundation.

A production-oriented version could evolve into:

```text
             CHURN MODEL
                  │
                  ▼
          Customer Risk Score
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   High Risk             Low Risk
        │                   │
        ▼                   ▼
Retention Action       Normal Journey
        │
        ▼
  Track Response
        │
        ▼
   Measure Impact
        │
        ▼
 Retrain / Improve
```

Possible extensions:

* 🎯 customer-level churn probability dashboard
* 📊 interactive Plotly dashboard
* 🔎 individual customer risk explanations
* 💰 retention-cost vs. customer-value analysis
* 📬 targeted retention recommendations
* ⚖️ threshold optimization based on business cost
* 🔄 model monitoring and retraining
* 🌐 Flask/FastAPI prediction API
* ☁️ cloud deployment
* 📈 integration with a BI dashboard

---

# 🧭 The Bigger Picture

This project isn't just about predicting a column called `Churn`.

It's about following a much more useful path:

```text
RAW CUSTOMER DATA
       ↓
UNDERSTAND THE DATA
       ↓
PREPARE THE FEATURES
       ↓
TRAIN MULTIPLE MODELS
       ↓
COMPARE PERFORMANCE
       ↓
UNDERSTAND MODEL SIGNALS
       ↓
IDENTIFY HIGH-RISK PATTERNS
       ↓
DESIGN RETENTION STRATEGIES
```

The final goal of machine learning in a business setting isn't simply:

> **“My model achieved a good score.”**

It's:

> **“My model found a useful signal, and I understand how that signal could support a better decision.”**

---

<p align="center">
  <b>🐍 Python · 📊 Data Science · 🤖 Machine Learning · 🌲 Random Forest · 📈 Customer Analytics</b>
</p>

<p align="center">
  <i>Predict the risk. Understand the signal. Improve the decision.</i>
</p>
