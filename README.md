# 📉 Telco Customer Churn Prediction and High-Impact Retention Strategy

## Project Overview

This is an end-to-end Data Science project focused on predicting customer churn for a fictional telecom company. The primary goal was not just model prediction, but to **identify the key, actionable drivers of churn** and translate those insights into a clear, high-ROI retention strategy for the business.

The project follows a rigorous, professional workflow encompassing Data Acquisition, EDA, Preprocessing, Comparative Modeling (Logistic Regression vs. Random Forest), and deep feature importance analysis.

## Key Outcomes and Impact

- **Model Performance:** Built and compared two classification models (Logistic Regression and Random Forest), with **Random Forest achieving the highest ROC AUC** (e.g., 0.84) for superior customer risk ranking.
- **Actionable Insights:** Identified **Month-to-Month contracts, Fiber Optic internet service, and low tenure (new customers)** as the three most critical risk factors.
- **Strategic Recommendations:** Developed targeted, high-value retention strategies for each driver, including loyalty incentives for contract upgrades and focused onboarding programs for new subscribers.

---

## Technical Stack & Libraries

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Core scripting and analysis. |
| **Data Handling** | Pandas, NumPy | Data manipulation, cleaning, and numerical operations. |
| **Modeling** | Scikit-learn | Used for `Pipeline`, `ColumnTransformer`, `LogisticRegression`, and `RandomForestClassifier`. |
| **Visualization** | Matplotlib, Seaborn | Explored variable distributions, calculated churn rates, and generated the final ROC Curve. |
| **Environment** | Jupyter Notebook / Google Colab | Iterative development and presentation of results. |

---

## Project Workflow (6-Step Process)

This project strictly followed a standard ML pipeline, showcasing clean and reproducible code:

1.  **Data Acquisition & Setup:** Loaded the Telco Customer Churn Dataset and handled initial data type inconsistencies (specifically coercing `TotalCharges`).
2.  **Exploratory Data Analysis (EDA):** Visualized the imbalanced target variable (Churn) and used `seaborn` bar plots to show that churn rates are significantly higher for month-to-month contracts and fiber optic users.
3.  **Data Preprocessing:** Defined a powerful `ColumnTransformer` to handle various data types:
    * **Numerical:** Scaled features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.
    * **Categorical:** Encoded all non-numeric features using `OneHotEncoder`.
4.  **Data Splitting:** Applied an 80/20 Train-Test split, using `stratify=Y` to maintain the original churn ratio in both sets, crucial for imbalanced data.
5.  **Comparative Modeling:** Built robust `Pipelines` for both Logistic Regression and Random Forest, trained on the preprocessed data, and evaluated their performance using the `classification_report` and **ROC AUC score**.
6.  **Interpretation & Recommendations:** Extracted and visualized the **top 10 feature importances** from the Random Forest model to drive evidence-based business strategies (see details in the notebook).

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/taniiishaa/Telco-Customer-Churn-Prediction-and-Retention-System.git
    ```
2.  **Download the Data:** Obtain the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file from Kaggle or the source linked in the notebook.
3.  **Launch the Notebook:** Open the primary `.ipynb` file in your preferred environment (Jupyter Lab/Notebook or Google Colab).
4.  **Execute Cells:** Run the cells sequentially to reproduce all findings and model results.

---
