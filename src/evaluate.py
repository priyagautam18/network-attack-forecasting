import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data_file = "data/processed/network_data_processed.csv"
model_file = "models/network_attack_model.pkl"

df = pd.read_csv(data_file)

X = df.drop("attack", axis=1)
y = df["attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(model_file)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Evaluation")
print("----------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs("results", exist_ok=True)

with open("results/evaluation.txt", "w") as file:
    file.write("Network Attack Forecasting Model Evaluation\n")
    file.write("--------------------------------------------\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(y_test, predictions))

print("Evaluation results saved.")