import pandas as pd
import os

input_file = "data/network_data.csv"
output_file = "data/processed/network_data_processed.csv"

df = pd.read_csv(input_file)

df = df.dropna()

os.makedirs("data/processed", exist_ok=True)

df.to_csv(output_file, index=False)

print("Data preprocessing completed.")
print("Rows:", len(df))
print("Columns:", list(df.columns))