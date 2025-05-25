import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
data = pd.read_csv('employee_attrition.csv')

# Target variable encoding
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Features to use
features = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears',
            'YearsAtCompany', 'WorkLifeBalance', 'OverTime', 'EnvironmentSatisfaction',
            'StockOptionLevel', 'NumCompaniesWorked']

X = data[features]
y = data['Attrition']

# Encode categorical variable 'OverTime'
X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

# Scale numerical features
num_cols = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears',
            'YearsAtCompany', 'WorkLifeBalance', 'EnvironmentSatisfaction',
            'StockOptionLevel', 'NumCompaniesWorked']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)

# Save model and scaler
joblib.dump(model, 'attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved!")
