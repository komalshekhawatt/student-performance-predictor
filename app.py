import sys
from flask import Flask, request, render_template, session, redirect, url_for

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)
application = app
app.secret_key = "gradeiq_super_secret_key"

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

# In-memory mock database for users mapping email -> Full Name
users_db = {}

# Dummy Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').lower()
        
        # Check if the user exists in our mock database
        if email in users_db:
            # Retrieve the exact name they registered with
            session['user_name'] = users_db[email]
        else:
            # Fallback for users logging in without registering first
            name_part = email.split('@')[0] if email else ""
            clean_name = ''.join(filter(str.isalpha, name_part))
            if not clean_name:
                clean_name = "User"
            session['user_name'] = clean_name.capitalize()
            
        return redirect(url_for('predict_datapoint'))
    return render_template('login.html')

# Dummy Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').lower()
        name = request.form.get('name', 'User').strip()
        
        if not name:
            name = "User"
            
        # Capitalize and store user securely in the mock DB
        formatted_name = name.title()
        if email:
            users_db[email] = formatted_name
            
        # Set session and redirect to login page!
        session['user_name'] = formatted_name
        return redirect(url_for('login'))
        
    return render_template('register.html')

# Route to accept prediction requests
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    user_name = session.get('user_name', '')
    
    if request.method == 'GET':
        return render_template('home.html', user_name=user_name)
    else:
        try:
            logging.info("Received prediction request from frontend.")
            
            # Step A: Instantiate the CustomData object with form inputs
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Step B: Get the Pandas DataFrame representation from CustomData
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Incoming Prediction Frame:\n{pred_df.to_string()}")

            # Step C: Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()
            
            logging.info("Making Prediction...")
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction successful: {results[0]}")

            return render_template('home.html', results=results[0], user_name=user_name)

        except Exception as e:
            # Capture the error via our CustomException utility
            custom_error = CustomException(e, sys)
            logging.error(f"Prediction Pipeline Error: {custom_error}")
            
            # Return safely to frontend to avoid crashing the server
            error_message = "An error occurred generating your prediction. Please verify your inputs."
            return render_template('home.html', error_message=error_message, user_name=user_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
