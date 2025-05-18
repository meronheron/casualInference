import pandas as pd

# Load local Lalonde dataset
data = pd.read_csv("lalonde.csv")

# Save to CSV (to ensure consistent naming for Streamlit)
data.to_csv("lalonde.csv", index=False)

# Verify
print("Saved lalonde.csv with", len(data), "rows. Columns:", data.columns.tolist())