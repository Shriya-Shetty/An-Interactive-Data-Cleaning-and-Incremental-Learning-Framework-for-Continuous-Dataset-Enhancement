import joblib
import pandas as pd

# Load trained model
model = joblib.load("freelancer_earnings_model.pkl")

# New freelancer input (must match training columns)
new_data = pd.DataFrame([{
    "category": "AI/ML Engineering",
    "primary_skills": "Python, TensorFlow",
    "years_experience": 5,
    "experience_level": "Mid",
    "region": "Asia",
    "country": "India",
    "education": "Bachelor",
    "hourly_rate_usd": 35,
    "primary_platform": "Upwork"
}])

# Predict earnings
prediction = model.predict(new_data)
print("Predicted Annual Income (USD):", prediction[0])
