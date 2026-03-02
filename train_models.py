import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# 1. Load the dataset we just created
df = pd.read_csv('crop_data.csv')

print("Starting training...")

# ==========================================
# MODEL 1: CROP RECOMMENDATION (Random Forest)
# ==========================================
# Input features: N, P, K, Temperature, Rainfall
X_rec = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall']]
# Target we want to predict: The Crop name
y_rec = df['Crop']

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_rec, y_rec)

# Save the trained model as a file
joblib.dump(rf_model, 'crop_recommender.joblib')
print("✅ Crop Recommendation Model trained and saved!")


# ==========================================
# MODEL 2: YIELD PREDICTION (XGBoost)
# ==========================================
# Computers understand numbers better than text. 
# We use LabelEncoder to turn crop names (Wheat, Rice) into numbers (0, 1).
encoder = LabelEncoder()
df['Crop_Numeric'] = encoder.fit_transform(df['Crop'])

# Input features: N, P, K, Temp, Rainfall, AND the specific Crop_Numeric
X_yield = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall', 'Crop_Numeric']]
# Target we want to predict: The Yield in kg
y_yield = df['Yield_kg_per_acre']

# Create and train the XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_yield, y_yield)

# Save the model and the encoder tool
joblib.dump(xgb_model, 'yield_predictor.joblib')
joblib.dump(encoder, 'crop_encoder.joblib')
print("✅ Yield Prediction Model trained and saved!")