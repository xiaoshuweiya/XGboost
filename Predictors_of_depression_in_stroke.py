import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('XGBoost.pkl')

feature_names = ["Gender", "Age", "PIR", "Drinking", "Sleep_disorder", "Moderate_physical_activity", "Total_cholesterol"]

# Streamlit user interface
st.title("Predictors of depression in stroke patients")

# sex: categorical selection
Gender = st.selectbox("Gender (0=Male, 1=Female):", options=[0, 1], format_func=lambda x: 'Male (0)' if x == 0 else 'Female (1)')

# age: numerical input
Age = st.number_input("Age:", min_value=20, max_value=85, value=50)

# PIR: numerical input
PIR = st.number_input("PIR:", min_value=0, max_value=5, value=3)

# Dringking: categorical selection
Drinking = st.selectbox("Drinking (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Sleep disorder: categorical selection
Sleep_disorder = st.selectbox("Sleep_disorder (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# sex: categorical selection
Moderate_physical_activity = st.selectbox("Moderate_physical_activity (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# PIR: numerical input
Total_cholesterol = st.number_input("Total_cholesterol:", min_value=2.07, max_value=9.98, value=6.0)

# Process inputs and make predictions
feature_values = [Gender, Age, PIR, Drinking, Sleep_disorder, Moderate_physical_activity, Total_cholesterol]
features = np.array([feature_values])

if st.button("Predict"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, you have a high risk of Depression. "
                f"The model predicts that your probability of having Depression is {probability:.1f}%. "
                "While this is just an estimate, it suggests that you may be at significant risk. "
                "I recommend that you consult a dactor as soon as possible for further evaluation and "
                "to ensure you receive an accurate diagnosis and necessary treatment."
            )
        else:
            advice = (
                f"According to our model, you have a low risk of Depression. "
                f"The model predicts that your probability of not having Depression is {probability:.1f}%. "
                "However, maintaining a healthy lifestyle is still very important. "
                "I recommend regular check-ups to monitor your heart health, "
                "and to seek medical advice promptly if you experience any symptoms."
            )

        st.write(advice)

        # Calculate SHAP values and display force plot
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

            shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

        # Display the SHAP force plot image
        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")