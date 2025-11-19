# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# from PIL import Image

# st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")
# @st.cache_resource
# def load_model():
#     return joblib.load("basic_student_depression_model.pkl")  # Load your trained model
# model = load_model()

# with st.sidebar:
#     st.title(" Student Depression Predictor")
#     st.markdown("Predict the likelihood of depression in students.")
#     st.markdown("---")
#     st.markdown("Developed by: **Ayush Khare**")
#     st.markdown("🔗 GitHub: [Ayush Khare](https://github.com/LordoftheRings9)")
#     st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/ayush-khare-226a79339?utm_source=share_via&utm_content=profile&utm_medium=member_android)")
#     st.markdown("---")
#     st.markdown("**Model accuracy will depend upon the accuracy of the data provided**")
#     st.markdown("#### Please provide the student details here →")
    
# st.markdown("<h1 style='text-align: center; color: #6DA7EE;'>Student Depression Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #6B7280;'>Enter the student's lifestyle and habits below</h3>", unsafe_allow_html=True)
    
# image = Image.open('9152887.jpg')
# st.image(image, use_container_width=True)
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("#### Please provide the student details below:")

# col1, col2 = st.columns(2)
# with col1:
#     id_val = st.number_input("Student ID (any numeric value)", min_value=0, step=1)
#     gender = st.radio("Gender", ["Male", "Female"])
#     age = st.number_input("Age", min_value=1, max_value=120, step=1)
#     city = st.text_input("City")
#     profession = st.text_input("Profession")
#     academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0)
#     study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
#     sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

# with col2:
#     dietary_habits = st.radio("Dietary Habits", ["Healthy", "Unhealthy"])
#     suicidal_thoughts = st.radio("Ever had suicidal thoughts?", ["Yes", "No"])
#     work_pressure = st.slider("Work Pressure (1-5)", 0.0, 5.0, 0.0)
#     financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
#     family_history = st.radio("Family history of mental illness?", ["Yes", "No"])
#     job_satisfaction = st.slider("Job Satisfaction (1-5)", 0.0, 5.0, 0.0)
#     study_pressure_hours = st.number_input("Work/Study Hours per week", min_value=0, max_value=24, step=1)
#     cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
#     degree = st.text_input("Degree")

# gender = 1 if gender == 'Male' else 0
# dietary_habits = 1 if dietary_habits == 'Healthy' else 0
# suicidal_thoughts = 1 if suicidal_thoughts == 'Yes' else 0
# family_history = 1 if family_history == 'Yes' else 0

# sleep_mapping = {
#     'Less than 5 hours': 4,
#     '5-6 hours': 5.5,
#     '7-8 hours': 7.5,
#     'More than 8 hours': 9
# }
# sleep_duration = sleep_mapping.get(sleep_duration, 7.5)

# columns = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
#            'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
#            'Sleep Duration', 'Dietary Habits', 'Degree',
#            'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
#            'Financial Stress', 'Family History of Mental Illness']

# input_df = pd.DataFrame([[id_val, gender, age, city, profession, academic_pressure,
#                           work_pressure, cgpa, study_satisfaction, job_satisfaction,
#                           sleep_duration, dietary_habits, degree, suicidal_thoughts,
#                           study_pressure_hours, financial_stress, family_history]],
#                         columns=columns)

# button_style = """
# <style>
# .stButton>button {
#     background-color: #2E8B57;
#     color: white;
#     font-size: 16px;
#     height: 3em;
#     width: 100%;
#     border-radius: 12px;
#     transition: background-color 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #3CB371;
# }
# </style>
# """
# st.markdown(button_style, unsafe_allow_html=True)
# if st.button("Predict"):
#     with st.spinner("Predicting..."):
#         try:
#             prediction_proba = model.predict_proba(input_df)
#             depression_prob = prediction_proba[0][1]  # probability of depression

#             # Conditional messages based on probability
#             if depression_prob < 0.2:
#                 st.markdown(f"<h3 style='color:green;'>Very unlikely to have depression.</h3>", unsafe_allow_html=True)
#             elif 0.2 <= depression_prob < 0.4:
#                 st.markdown(f"<h3 style='color:green;'>Unlikely to have depression.</h3>", unsafe_allow_html=True)
#             elif 0.4 <= depression_prob < 0.6:
#                 st.markdown(f"<h3 style='color:orange;'>May have depression.</h3>", unsafe_allow_html=True)
#             elif 0.6 <= depression_prob < 0.8:
#                 st.markdown(f"<h3 style='color:orange;'>Likely to have depression.</h3>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<h3 style='color:red;'>Highly likely to have depression.</h3>", unsafe_allow_html=True)

#             st.write(f"Depression Probability: {depression_prob*100:.2f}%")

#         except Exception as e:
#             st.error(f"Error during prediction: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load("basic_student_depression_model.pkl")

model = load_model()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🧠 Student Depression Predictor")
    st.markdown("Predict the likelihood of depression in students based on lifestyle and academic factors.")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by: **Ayush Khare**")
    st.markdown("🔗 [GitHub](https://github.com/LordoftheRings9)")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/ayush-khare-226a79339)")
    st.markdown("---")
    st.info("⚠️ Model accuracy depends on the accuracy of the data provided.")
    st.markdown("#### Please provide the student details below:")

# -------------------- Header --------------------
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Student Depression Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6B7280;'>Enter the student's lifestyle and academic details</h4>", unsafe_allow_html=True)

image = Image.open("9152887.jpg")
st.image(image, use_container_width=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------- Input Form --------------------
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    id_val = st.number_input("🆔 Student ID", min_value=0, step=1)
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    city = st.text_input("City")
    profession = st.text_input("Profession")
    academic_pressure = st.slider("📚 Academic Pressure (1-5)", 1.0, 5.0, 3.0)
    study_satisfaction = st.slider("📖 Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
    sleep_duration = st.selectbox("🛌 Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

with col2:
    dietary_habits = st.radio("🍎 Dietary Habits", ["Healthy", "Unhealthy"])
    suicidal_thoughts = st.radio("💭 Ever had suicidal thoughts?", ["Yes", "No"])
    work_pressure = st.slider("💼 Work Pressure (1-5)", 0.0, 5.0, 0.0)
    financial_stress = st.slider("💸 Financial Stress (1-5)", 1, 5, 3)
    family_history = st.radio("🧬 Family history of mental illness?", ["Yes", "No"])
    job_satisfaction = st.slider("🏢 Job Satisfaction (1-5)", 0.0, 5.0, 0.0)
    study_hours = st.number_input("⏱️ Work/Study Hours per week", min_value=0, max_value=24, step=1)
    cgpa = st.number_input("🎓 CGPA", min_value=0.0, max_value=10.0, step=0.01)
    degree = st.text_input("Degree")

# -------------------- Data Preprocessing --------------------
gender = 1 if gender == "Male" else 0
dietary_habits = 1 if dietary_habits == "Healthy" else 0
suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

sleep_mapping = {
    "Less than 5 hours": 4,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 9
}
sleep_duration = sleep_mapping.get(sleep_duration, 7.5)

# Only include numeric features expected by the model
input_data = pd.DataFrame([[
    id_val, gender, age, academic_pressure, work_pressure, cgpa,
    study_satisfaction, job_satisfaction, sleep_duration, dietary_habits,
    suicidal_thoughts, study_hours, financial_stress, family_history
]], columns=[
    'id', 'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
    'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits',
    'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress',
    'Family History of Mental Illness'
])

# -------------------- Prediction Button --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<style>.stButton>button {background-color:#2E8B57;color:white;font-size:16px;border-radius:10px;}</style>", unsafe_allow_html=True)

if st.button("🔍 Predict"):
    with st.spinner("Analyzing..."):
        try:
            prob = model.predict_proba(input_data)[0][1]
            st.markdown(f"### 🧠 Depression Probability: **{prob*100:.2f}%**")

            if prob < 0.2:
                st.success("✅ Very unlikely to have depression.")
            elif prob < 0.4:
                st.success("🟢 Unlikely to have depression.")
            elif prob < 0.6:
                st.warning("🟠 May have depression.")
            elif prob < 0.8:
                st.warning("🟠 Likely to have depression.")
            else:
                st.error("🔴 Highly likely to have depression.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
