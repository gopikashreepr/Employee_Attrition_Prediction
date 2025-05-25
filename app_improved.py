import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load model and scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("🧑‍💼 Employee Attrition Predictor")

st.markdown("""
Predict whether an employee is likely to leave the company based on their profile.
Fill in the details below and click **Predict**.
""")

def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', 18, 60, 30)
        job_satisfaction = st.slider('Job Satisfaction (1-4)', 1, 4, 3)
        monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000, step=500)
        total_working_years = st.slider('Total Working Years', 0, 40, 5)
        years_at_company = st.slider('Years at Company', 0, 40, 3)
    with col2:
        work_life_balance = st.slider('Work Life Balance (1-4)', 1, 4, 3)
        overtime = st.selectbox('Overtime', ['No', 'Yes'])
        environment_satisfaction = st.slider('Environment Satisfaction (1-4)', 1, 4, 3)
        stock_option_level = st.slider('Stock Option Level (0-3)', 0, 3, 0)
        num_companies_worked = st.slider('Number of Companies Worked', 0, 10, 1)
    
    data = {
        'Age': age,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'WorkLifeBalance': work_life_balance,
        'OverTime': 1 if overtime == 'Yes' else 0,
        'EnvironmentSatisfaction': environment_satisfaction,
        'StockOptionLevel': stock_option_level,
        'NumCompaniesWorked': num_companies_worked
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale numerical columns used in training model
num_cols = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears',
            'YearsAtCompany', 'WorkLifeBalance', 'EnvironmentSatisfaction',
            'StockOptionLevel', 'NumCompaniesWorked']

input_df[num_cols] = scaler.transform(input_df[num_cols])

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader('Prediction Result')
attrition_status = "❗️ Likely to Leave" if prediction == 1 else "✅ Likely to Stay"
st.markdown(f"### {attrition_status}")

st.subheader('Prediction Probability')
st.progress(int(prediction_proba * 100))
st.write(f"Probability of leaving the company: **{prediction_proba:.2%}**")

# Feature importance plot
st.subheader("Feature Importance")
importance = model.feature_importances_
features = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears',
            'YearsAtCompany', 'WorkLifeBalance', 'OverTime', 'EnvironmentSatisfaction',
            'StockOptionLevel', 'NumCompaniesWorked']

feat_imp = pd.Series(importance, index=features).sort_values(ascending=True)

fig, ax = plt.subplots()
feat_imp.plot(kind='barh', ax=ax, color='teal')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance from Random Forest')
st.pyplot(fig)

# Show input features summary
st.subheader("Employee Profile Summary")
st.table(input_df[num_cols].applymap(lambda x: f"{x:.2f}"))

# Option to download prediction report
def generate_report():
    report = f"""
    Employee Attrition Prediction Report
    ------------------------------------
    Prediction: {attrition_status}
    Probability of Leaving: {prediction_proba:.2%}
    
    Input Features:
    {input_df.to_string(index=False)}
    
    Feature Importance (Top 3):
    {feat_imp.tail(3).to_string()}
    """
    return report

if st.button("Download Prediction Report"):
    report_text = generate_report()
    st.download_button("Download as TXT", report_text, file_name="attrition_prediction_report.txt")

st.markdown("---")
st.markdown("**Note:** This prediction is based on a machine learning model trained on IBM HR Analytics data. Use as guidance, not final HR decisions.")
