# 🏥 Patient Readmission Risk Prediction

A machine learning project that predicts whether a patient is at **high risk of hospital readmission** based on clinical and demographic information.

The goal is to help healthcare providers identify high-risk patients early and support better follow-up and care planning.

## 📌 Project Overview

Hospital readmissions can increase healthcare costs and indicate that a patient may need additional monitoring or post-discharge care.

This project uses **Python and Machine Learning** to analyze patient data and build a predictive model for estimating the likelihood of readmission.

### 🎯 Objective

* Analyze patient demographic and clinical data
* Perform data cleaning and preprocessing
* Identify factors associated with readmission
* Train machine learning classification models
* Evaluate model performance
* Predict readmission risk for new patients

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib & Seaborn** – Data visualization
* **Scikit-learn** – Machine learning
* **Jupyter Notebook** – Development and analysis

## 📊 Dataset

The dataset contains patient-related information such as:

* Age
* Gender
* Medical conditions
* Number of previous visits
* Hospital stay duration
* Number of medications
* Lab/test results
* Previous admissions
* Discharge information
* Readmission status

> **Note:** The project should use de-identified or publicly available data. It is intended for educational and research purposes and should not be used for clinical decisions.

## 🔄 Project Workflow

```text
Patient Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Readmission Risk Prediction
```

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Removing duplicate records
* Detecting and handling outliers
* Encoding categorical variables
* Feature scaling where required
* Splitting data into training and testing sets

Example:

```python
import pandas as pd

df = pd.read_csv("patient_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Fill numerical missing values
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)
```

## 📈 Exploratory Data Analysis

Important questions explored during EDA:

* What percentage of patients are readmitted?
* Which age groups have higher readmission rates?
* Does previous hospitalization affect readmission?
* Does length of stay correlate with readmission?
* Which medical conditions are associated with higher risk?
* How does medication count relate to readmission?

Example:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x="age", bins=10)

plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()
```

## 🤖 Machine Learning Models

The project can compare multiple classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

## 📏 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, predictions))

print(
    classification_report(
        y_test,
        predictions
    )
)

print(confusion_matrix(y_test, predictions))
```

### ⚠️ Why Recall Matters

For readmission-risk prediction, **recall can be particularly important** because missing a genuinely high-risk patient can be more problematic than flagging some lower-risk patients for additional follow-up.

Therefore, model selection should not rely on accuracy alone.

## 🔍 Feature Importance

For tree-based models, feature importance can help identify which variables contribute most to predictions.

```python
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(10))
```

## 📁 Project Structure

```text
patient-readmission-risk/
│
├── data/
│   └── patient_data.csv
│
├── notebooks/
│   └── readmission_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── readmission_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repository-url>

cd patient-readmission-risk
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Run the training script:

```bash
python src/train.py
```

Make predictions:

```bash
python src/predict.py
```

Or open the Jupyter Notebook:

```bash
jupyter notebook
```

## 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## 🔮 Future Improvements

* Handle class imbalance using SMOTE or class weights
* Perform hyperparameter tuning
* Add explainable AI using SHAP
* Build a Streamlit prediction dashboard
* Add model monitoring
* Deploy the model using Docker
* Create an API using FastAPI
* Add patient-level risk explanations

## ⚠️ Disclaimer

This project is developed for **educational purposes only**. It is not a medical diagnostic system and should not be used to make real-world clinical decisions.

## 👨‍💻 Author

**Jiya Thakur**

If you found this project useful, consider ⭐ starring the repository.
