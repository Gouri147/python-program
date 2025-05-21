from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset and train models
df = pd.read_csv(r'C:\fertilizer app\data_core_with_yield_rainfall.csv')

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Soil Type', 'Crop Type', 'Fertilizer']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature columns
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Phosphorous', 'Potassium', 'Rainfall']

# Targets
target_yield = 'Yield'
target_fertilizer = 'Fertilizer'

# Split and train yield model
X_yield = df[features]
y_yield = df[target_yield]
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)
yield_model = RandomForestRegressor()
yield_model.fit(X_train_y, y_train_y)

# Split and train fertilizer model
y_fertilizer = df[target_fertilizer]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_yield, y_fertilizer, test_size=0.2, random_state=42)
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_train_f, y_train_f)

# ➤ WELCOME PAGE ROUTE
@app.route('/')
def welcome():
    return render_template('welcome.html')

# ➤ INDEX PAGE ROUTE
@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html',
                           soil_types=label_encoders['Soil Type'].classes_,
                           crop_types=label_encoders['Crop Type'].classes_)

# ➤ PREDICTION ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])

        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']

        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])
        rainfall = float(request.form['rainfall'])

        encoded_soil = label_encoders['Soil Type'].transform([soil_type])[0]
        encoded_crop = label_encoders['Crop Type'].transform([crop_type])[0]

        user_input = [[
            temperature, humidity, moisture,
            encoded_soil, encoded_crop,
            nitrogen, phosphorous, potassium, rainfall
        ]]

        predicted_yield = yield_model.predict(user_input)[0]
        fertilizer_encoded = fertilizer_model.predict(user_input)[0]
        fertilizer_label = label_encoders['Fertilizer'].inverse_transform([fertilizer_encoded])[0]

        return render_template('result.html',
                               predicted_yield=f"{predicted_yield:.2f}",
                               fertilizer=fertilizer_label)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
