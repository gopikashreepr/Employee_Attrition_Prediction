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

## 📂 Project Structure

<pre> ``` ├── dataset/ # Dataset folder │ └── employee_data.csv # HR analytics dataset ├── models/ # Saved model and scaler │ ├── attrition_model.pkl │ └── scaler.pkl ├── app.py # Flask app (optional for API) ├── notebook.ipynb # EDA and model training ├── requirements.txt # Python dependencies └── README.md # Project documentation ``` </pre>
