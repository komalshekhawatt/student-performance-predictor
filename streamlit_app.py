import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page configuration
st.set_page_config(
    page_title="GradeIQ - Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 GradeIQ Performance Analysis")
st.write("Enter the student parameters below to generate a predicted math score via our Ridge Regression model.")

# Create form
with st.form("prediction_form"):
    st.subheader("Student Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["male", "female"])
        ethnicity = st.selectbox("Race / Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
    
    st.subheader("Background & Preparation")
    parental_level_of_education = st.selectbox(
        "Parental Education Background", 
        options=["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
    )
    
    col3, col4 = st.columns(2)
    with col3:
        lunch = st.selectbox("Lunch Program", options=["free/reduced", "standard"])
    with col4:
        test_preparation_course = st.selectbox("Test Prep Course", options=["none", "completed"])

    st.subheader("Prior Scores")
    col5, col6 = st.columns(2)
    with col5:
        writing_score = st.number_input("Writing Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    with col6:
        reading_score = st.number_input("Reading Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        
    submitted = st.form_submit_button("Generate Prediction")

# Handle prediction
if submitted:
    try:
        # 1. Map to CustomData
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )
        
        # 2. Get DataFrame
        pred_df = data.get_data_as_data_frame()
        
        with st.spinner('Analyzing patterns...'):
            # 3. Predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
        st.success("Prediction Generated Successfully!")
        
        st.metric(label="Predicted Math Score", value=f"{results[0]:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
