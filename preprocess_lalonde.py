import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading  dataset
data = pd.read_csv("lalonde.csv")

# Clean: Drop 'Unnamed: 0' and check for missing values
data = data.drop(columns=["Unnamed: 0"], errors="ignore")
print("Missing values:\n", data.isnull().sum())

# Normalization: Scale continuous covariates
covariates = ["age", "educ", "re74", "re75"]
scaler = StandardScaler()
data[covariates] = scaler.fit_transform(data[covariates])

# Saving preprocessed dataset
data.to_csv("lalonde_preprocessed.csv", index=False)

# Verify
print("Saved lalonde_preprocessed.csv with", len(data), "rows. Columns:", data.columns.tolist())