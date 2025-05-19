import pandas as pd
import numpy as np
# setting random seed for reproducibility
np.random.seed(42)
# read the excel dataset
df = pd.read_excel("online_retail_II.xlsx")
# Clean data: remove missing Customer ID, cancellations
print("Original rows:", len(df))
df = df.dropna(subset=["Customer ID"])
print("Rows after dropping missing Customer ID:", len(df))
df = df[~df["Invoice"].str.startswith("C", na=False)]
