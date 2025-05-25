# Employee_Attrition_Prediction

This project aims to predict whether an employee is likely to leave a company (attrition) based on various workplace and personal features using machine learning techniques.

---

## 📌 Features

- Predicts employee attrition (Yes/No) based on input features.
- Uses structured HR data like job role, satisfaction, income, overtime, etc.
- Data preprocessing and model training included.
- Interactive API using Flask (optional).
- Model and scaler are saved for reuse.

---

## 🛠 Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)
- **Jupyter Notebook** (Model training and EDA)
- **Flask** (for optional API endpoint)
- **Joblib** or **Pickle** (for model serialization)

---
## 📊 Dataset Overview

The dataset contains HR-related features such as:

- Age, Department, JobRole
- MonthlyIncome, Overtime
- JobSatisfaction, WorkLifeBalance
- YearsAtCompany, and more

**Target Variable:** `Attrition` (Yes/No)

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the flask API locally 
```bash
python run app_improved.py
```

Let me know if:
- You’re using a specific model (e.g., neural network vs. logistic regression)?
- You’d like to include visuals like plots or a confusion matrix?
- You want a deployment version (e.g., for Streamlit or Render)?

I’ll tailor it accordingly.
