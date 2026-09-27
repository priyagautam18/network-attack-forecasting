import pandas as pd
import joblib
import os

model_file = "models/network_attack_model.pkl"

model = joblib.load(model_file)

new_data = pd.DataFrame([
    {
        "packet_count": 650,
        "packet_size": 1600,
        "connection_count": 140,
        "failed_connections": 70,
        "port_scan": 1
    }
])

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

if prediction[0] == 1:
    result = "Potential Network Attack"
else:
    result = "Normal Network Traffic"

confidence = max(probability[0]) * 100

print("Network Attack Forecast")
print("-----------------------")
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}%")

os.makedirs("results", exist_ok=True)

with open("results/forecast.txt", "w") as file:
    file.write("Network Attack Forecast\n")
    file.write("-----------------------\n")
    file.write(f"Prediction: {result}\n")
    file.write(f"Confidence: {confidence:.2f}%\n")