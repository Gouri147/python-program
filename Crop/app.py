import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'C:\WhatsApp Downloads\data_core_with_yield_rainfall.csv')

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

# Target columns
target_yield = 'Yield'
target_fertilizer = 'Fertilizer'

# Split for yield prediction
X_yield = df[features]
y_yield = df[target_yield]
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

# Train yield prediction model
yield_model = RandomForestRegressor()
yield_model.fit(X_train_y, y_train_y)
yield_preds = yield_model.predict(X_test_y)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test_y, yield_preds))
print("Yield Prediction RMSE:", rmse)

# Split for fertilizer recommendation
y_fertilizer = df[target_fertilizer]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_yield, y_fertilizer, test_size=0.2, random_state=42)

# Train fertilizer recommendation model
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_train_f, y_train_f)
fertilizer_preds = fertilizer_model.predict(X_test_f)

# Compute accuracy
print("Fertilizer Recommendation Accuracy:", accuracy_score(y_test_f, fertilizer_preds))

# -------- Take user input --------
print("\nEnter input values for prediction:")

temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
moisture = float(input("Moisture (%): "))

# Validate Soil Type input
valid_soil_types = list(label_encoders['Soil Type'].classes_)
while True:
    soil_type = input(f"Soil Type {valid_soil_types}: ")
    if soil_type in valid_soil_types:
        encoded_soil = label_encoders['Soil Type'].transform([soil_type])[0]
        break
    else:
        print("Invalid Soil Type! Please enter a valid option.")

# Validate Crop Type input
valid_crop_types = list(label_encoders['Crop Type'].classes_)
while True:
    crop_type = input(f"Crop Type {valid_crop_types}: ")
    if crop_type in valid_crop_types:
        encoded_crop = label_encoders['Crop Type'].transform([crop_type])[0]
        break
    else:
        print("Invalid Crop Type! Please enter a valid option.")

nitrogen = float(input("Nitrogen (N): "))
phosphorous = float(input("Phosphorous (P): "))
potassium = float(input("Potassium (K): "))
rainfall = float(input("Rainfall (mm): "))

# Create input array
user_input = [[
    temperature, humidity, moisture,
    encoded_soil, encoded_crop,
    nitrogen, phosphorous, potassium, rainfall
]]

# Predict yield and fertilizer
predicted_yield = yield_model.predict(user_input)[0]
fertilizer_encoded = fertilizer_model.predict(user_input)[0]
fertilizer_label = label_encoders['Fertilizer'].inverse_transform([fertilizer_encoded])[0]

# Display results
print(f"\n✅ Predicted Yield: {predicted_yield:.2f} q/ha")
print(f"✅ Recommended Fertilizer: {fertilizer_label}")