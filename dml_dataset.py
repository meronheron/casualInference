import pandas as pd
import numpy as np
# setting random seed for reproducibility
np.random.seed(42)
# read the excel dataset
df = pd.read_excel("online_retail_II.xlsx")
# clean data: remove missing Customer ID, cancellations
print("Original rows:", len(df))
df = df.dropna(subset=["Customer ID"])
print("Rows after dropping missing Customer ID:", len(df))
df = df[~df["Invoice"].str.startswith("C", na=False)]
# sample 10,000 rows
df = df.sample(n=10000, random_state=42)
# create synthetic treatment (T: 1 if Country is UK, 0 otherwise)
df["T"] = (df["Country"] == "United Kingdom").astype(int)

# create outcome (Y: total purchase amount = Quantity * Price)
df["Y"] = df["Quantity"] * df["Price"]

# Select covariates (X) and controls (W)
covariates = ["Quantity", "Price"]
controls = ["Customer ID"]
df = df[covariates + controls + ["T", "Y"]]
# save processed dataset
df.to_csv("dml_kaggle.csv", index=False)

# verify
print("Saved dml_kaggle.csv with", len(df), "rows. Columns:", df.columns.tolist())