# 🛡️ FraudShield AI

### Detect Fraud. Protect Every Transaction.

FraudShield AI is an end-to-end machine learning application for **credit-card transaction fraud detection**.

The project addresses a highly imbalanced binary classification problem where fraudulent transactions represent only a very small portion of all transactions. Instead of relying on accuracy alone, the project evaluates the model using **precision, recall, F1-score, ROC-AUC, PR-AUC, false positives, false negatives, and business-cost analysis**.

The final model is integrated into an interactive **Streamlit dashboard** that estimates fraud probability and provides a risk level and suggested action.

> ⚠️ **Disclaimer:** FraudShield AI is an educational and portfolio project. It is not a production banking fraud-prevention system and should not be used to make real financial decisions without appropriate validation, security, monitoring, governance, and regulatory controls.

---

## 🚀 Demo

The application allows a user to:

* Enter transaction information
* Enter anonymized transaction features
* Generate a fraud probability
* View the transaction risk level
* Receive a suggested action
* Understand the model's decision threshold
* Explore the project and model information

### Application Flow

```text
Transaction Data
       │
       ▼
Feature Preparation
       │
       ▼
XGBoost Model
       │
       ▼
Fraud Probability
       │
       ▼
Risk Assessment
       │
       ├── Low
       ├── Medium
       ├── High
       └── Critical
       │
       ▼
Suggested Action
```

---

# 🎯 Project Objectives

The main objectives of this project were to:

1. Understand the challenges of fraud detection with highly imbalanced data.
2. Perform exploratory data analysis.
3. Prepare transaction data for machine learning.
4. Establish a Logistic Regression baseline.
5. Experiment with class weighting and SMOTE.
6. Train an XGBoost fraud detection model.
7. Evaluate models using appropriate fraud-detection metrics.
8. Analyze the precision-recall tradeoff.
9. Tune the classification threshold using a business-cost framework.
10. Build a user-facing Streamlit application.
11. Save and deploy the trained model in a portable format.
12. Structure the project as a professional machine-learning portfolio project.

---

# 📊 Dataset

This project uses the **Credit Card Fraud Detection dataset** containing European cardholder transactions.

Dataset source:

**Kaggle — Credit Card Fraud Detection**

The original dataset contains:

| Property                |   Value |
| ----------------------- | ------: |
| Total transactions      | 284,807 |
| Features                |      30 |
| Target column           | `Class` |
| Legitimate transactions | 284,315 |
| Fraudulent transactions |     492 |
| Legitimate percentage   | ~99.83% |
| Fraud percentage        |  ~0.17% |

The dataset contains:

```text
Time
V1
V2
...
V28
Amount
Class
```

Where:

* `Time` represents elapsed seconds from the beginning of the dataset.
* `Amount` represents the transaction amount.
* `V1–V28` are anonymized numerical features.
* `Class = 0` represents a legitimate transaction.
* `Class = 1` represents a fraudulent transaction.

### Important Dataset Limitation

The meanings of `V1–V28` are not provided in the public dataset.

Therefore, this project does **not** claim that a particular `V` feature represents a specific customer behavior or financial characteristic.

---

# 🔍 Exploratory Data Analysis

Several aspects of the dataset were investigated before modeling.

## Class Distribution

Fraud detection is highly imbalanced:

```text
Legitimate:  █████████████████████████████████████████████  ~99.83%

Fraud:      ▏                                             ~0.17%
```

This imbalance means that a model could achieve very high accuracy while still failing to detect many fraudulent transactions.

Therefore:

> Accuracy alone is not an appropriate primary metric for this problem.

---

## Transaction Amount Analysis

The transaction amount was analyzed separately for legitimate and fraudulent transactions.

### Legitimate Transactions

* Mean: approximately **88.29**
* Median: approximately **22.00**
* Maximum: approximately **25,691.16**

### Fraudulent Transactions

* Mean: approximately **122.21**
* Median: approximately **9.25**
* Maximum: approximately **2,125.87**

The distributions are highly skewed, which motivated the use of a logarithmic transformation during exploratory analysis.

---

## Time Analysis

Transaction timing was also explored.

The dataset covers approximately **48 hours**.

The average transaction time differed between legitimate and fraudulent transactions, but time alone is not sufficient to identify fraud.

---

# 🧹 Data Preparation

The target variable was separated from the input features:

```python
X = df.drop(columns=["Class"])
y = df["Class"]
```

The dataset was divided using a stratified train/test split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Dataset Split

| Set      | Transactions |
| -------- | -----------: |
| Training |      227,845 |
| Testing  |       56,962 |

The fraud class was preserved in both sets using stratification.

---

# 🤖 Models Tested

Several approaches were evaluated.

## 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

Results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9991 |
| Precision | 0.8267 |
| Recall    | 0.6327 |
| F1-score  | 0.7168 |
| ROC-AUC   | 0.9605 |
| PR-AUC    | 0.7414 |

The model achieved high accuracy, but its recall showed that a significant portion of fraud cases could still be missed.

---

## 2. Class-Weighted Logistic Regression

Class weighting was tested to give more importance to fraudulent transactions.

Results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9755 |
| Precision | 0.0610 |
| Recall    | 0.9184 |
| F1-score  | 0.1144 |
| ROC-AUC   | 0.9721 |
| PR-AUC    | 0.7190 |

This demonstrates an important fraud-detection tradeoff:

> Increasing fraud recall can substantially increase false positives.

---

## 3. SMOTE + Logistic Regression

SMOTE was also evaluated to address class imbalance by generating synthetic minority-class samples.

Training data after SMOTE:

```text
Before:
Legitimate = 227,451
Fraud      = 394

After:
Legitimate = 227,451
Fraud      = 227,451
```

Results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9741 |
| Precision | 0.0578 |
| Recall    | 0.9184 |
| F1-score  | 0.1088 |
| ROC-AUC   | 0.9708 |
| PR-AUC    | 0.7245 |

---

# 🌳 Final Model: XGBoost

The final model uses **XGBoost**, a gradient-boosted decision-tree algorithm.

The model was selected because it provided a strong balance between fraud detection performance and false-positive control in the experiments.

### Model Configuration

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
```

### XGBoost Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9995 |
| Precision | 0.8817 |
| Recall    | 0.8367 |
| F1-score  | 0.8586 |
| ROC-AUC   | 0.9770 |
| PR-AUC    | 0.8780 |

### Why These Metrics Matter

For fraud detection:

**Precision**

> Of the transactions flagged as fraud, how many were actually fraud?

**Recall**

> Of all fraudulent transactions, how many did the model detect?

**F1-score**

> Balance between precision and recall.

**ROC-AUC**

> Measures ranking/discrimination ability across classification thresholds.

**PR-AUC**

> Particularly useful for evaluating performance on highly imbalanced classification problems.

---

# ⚖️ Threshold Optimization

A probability threshold determines when a transaction is classified as potentially fraudulent.

Instead of automatically using:

```text
threshold = 0.50
```

different thresholds were evaluated.

The project used a simple business-cost framework:

```text
False Negative Cost = 500
False Positive Cost = 25
```

The total cost was calculated as:

```python
Total Cost =
(False Negatives × 500)
+
(False Positives × 25)
```

The threshold analysis was used to identify a threshold that minimized the assumed cost on the evaluation data.

### Important

The selected threshold is **not universally optimal**.

A real financial institution would determine thresholds based on:

* fraud losses
* customer friction
* investigation capacity
* transaction value
* operational costs
* regulatory requirements
* business risk tolerance

---

# 🧠 Risk Assessment

FraudShield AI converts the model's probability into a simple risk category.

| Probability | Risk Level |
| ----------: | ---------- |
|     `< 30%` | Low        |
|    `30–70%` | Medium     |
|    `70–90%` | High       |
|     `≥ 90%` | Critical   |

The application also provides a suggested action:

|             Model Probability | Suggested Action  |
| ----------------------------: | ----------------- |
|      Below decision threshold | Approve           |
| Above threshold and below 90% | Review Manually   |
|                         ≥ 90% | Temporarily Block |

These actions are **demonstration logic**, not production financial rules.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard.

### Main Pages

```text
Home
│
├── Model Overview
├── Key Metrics
├── How FraudShield AI Works
└── Key Capabilities

Fraud Detection
│
├── Transaction Information
├── V1–V28 Inputs
├── Fraud Probability
├── Risk Level
└── Suggested Action

About
│
├── Project Description
├── Dataset Information
├── Model Information
├── Limitations
└── Production Considerations
```

---

# 🛠️ Technology Stack

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| Python           | Programming language                         |
| pandas           | Data manipulation                            |
| NumPy            | Numerical computation                        |
| scikit-learn     | Data splitting, preprocessing and evaluation |
| imbalanced-learn | SMOTE                                        |
| XGBoost          | Final classification model                   |
| Streamlit        | Interactive web application                  |
| Plotly           | Data visualization                           |
| joblib           | Saving feature metadata                      |
| Git/GitHub       | Version control and portfolio                |

---

# 📁 Project Structure

```text
fraudshield-ai/
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
├── models/
│   ├── fraud_model.json
│   ├── feature_names.joblib
│   └── model_metadata.joblib
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── images/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_analysis.png
│   └── dashboard.png
│
└── data/
    └── fraudshield_test_transactions.csv
```

> The original full credit-card dataset should not be committed to the repository.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fraudshield-ai.git
```

Move into the project:

```bash
cd fraudshield-ai
```

---

## 2. Create a virtual environment

Windows:

```bash
py -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 🧪 Testing the Model

The application can be tested using real transactions from the held-out test dataset.

For example:

```python
fraud_test_row = X_test[y_test == 1].iloc[0]

legitimate_test_row = X_test[y_test == 0].iloc[0]
```

The model can then generate fraud probabilities:

```python
fraud_probability = model.predict_proba(
    fraud_test_row.to_frame().T
)[0][1]
```

Testing should include both:

* legitimate transactions
* fraudulent transactions

This allows the application behavior to be checked against known labels.

---

# 📈 Key Machine Learning Lessons

This project provided practical experience with:

### 1. Imbalanced Classification

Fraud datasets can contain extremely few positive examples.

### 2. Why Accuracy Can Be Misleading

A model can achieve very high accuracy while missing fraudulent transactions.

### 3. Precision vs Recall

Increasing fraud detection can also increase false positives.

### 4. SMOTE

Synthetic oversampling can help address minority-class imbalance, but it does not automatically produce a better model.

### 5. Class Weighting

Class weights can increase attention to the minority class.

### 6. XGBoost

Gradient-boosted trees can model complex nonlinear relationships.

### 7. Threshold Tuning

The classification threshold can be adjusted according to business costs.

### 8. PR-AUC

Precision-Recall AUC is especially informative for highly imbalanced problems.

### 9. Model Deployment

A trained model needs to be integrated into an application before it becomes a usable ML product.

### 10. Responsible AI

Fraud predictions are risk signals rather than definitive proof of criminal or fraudulent behavior.

---

# ⚠️ Limitations

This project has several important limitations.

### Dataset Limitation

The dataset is a public historical dataset and does not represent all modern payment environments.

### Anonymized Features

The meanings of `V1–V28` are not available.

### No Real-Time Data

The application currently processes manually entered transaction information.

### No Production API

The project does not currently integrate with a real payment-processing system.

### No Continuous Monitoring

Production systems require monitoring for:

* data drift
* concept drift
* model degradation
* false-positive rates
* false-negative rates

### No Authentication

The current Streamlit application is a demonstration and does not implement enterprise authentication.

### No Transaction History

The current application evaluates individual transactions rather than maintaining a production transaction-monitoring database.

---

# 🔐 Production Considerations

A real fraud-detection platform would require additional components such as:

```text
Payment System
      │
      ▼
Real-Time API
      │
      ▼
Feature Engineering
      │
      ▼
Fraud Model
      │
      ├──────────────┐
      ▼              ▼
Risk Score      Rule Engine
      │              │
      └──────┬───────┘
             ▼
       Decision Engine
             │
       ┌─────┴─────┐
       ▼           ▼
   Approve      Investigation
```

Additional requirements would include:

* authentication and authorization
* secure data pipelines
* encryption
* audit logging
* model monitoring
* data drift detection
* model retraining
* human investigation workflows
* privacy protection
* regulatory compliance
* incident response

---

# 🔮 Future Improvements

Planned improvements include:

* [ ] Batch CSV transaction prediction
* [ ] Downloadable prediction reports
* [ ] Interactive Plotly risk gauge
* [ ] Confusion matrix dashboard
* [ ] ROC curve visualization
* [ ] Precision-Recall curve visualization
* [ ] Feature importance visualization
* [ ] Model comparison dashboard
* [ ] Threshold and business-cost simulator
* [ ] Transaction history
* [ ] Authentication
* [ ] FastAPI prediction service
* [ ] Docker deployment
* [ ] Model monitoring
* [ ] Data drift detection
* [ ] Automated model retraining

---

# 👨‍💻 Author

**Miruk Yilikal**

Electrical & Computer Engineering Student
Addis Ababa University

Areas of interest:

* Machine Learning
* Data Analytics
* Artificial Intelligence
* Software Engineering
* AI-powered Applications

---

# ⭐ Project Purpose

FraudShield AI was built as a practical demonstration of the complete machine-learning workflow:

```text
Problem Definition
       ↓
Data Understanding
       ↓
Exploratory Data Analysis
       ↓
Data Preparation
       ↓
Baseline Model
       ↓
Imbalance Handling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Threshold Optimization
       ↓
Model Saving
       ↓
Streamlit Deployment
       ↓
Portfolio / Production Readiness
```

The project demonstrates not only how to train a machine-learning model, but also how to transform that model into an interactive application while considering **evaluation, business costs, limitations, and responsible deployment**.
