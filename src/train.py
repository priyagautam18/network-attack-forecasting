import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_file = "data/processed/network_data_processed.csv"

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

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/network_attack_model.pkl")

print("Model training completed.")
print("Model saved successfully.")