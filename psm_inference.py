import pandas as pd
import numpy as np
from causalml.match import NearestNeighborMatch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# loading preprocessed dataset
data = pd.read_csv("lalonde_preprocessed.csv")

# define treatment, outcome, and covariates
treatment = "treat"
outcome = "re78"
covariates = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]

# PSM
psm = NearestNeighborMatch(replace=True, random_state=42)
matched_data = psm.match(data=data, treatment_col=treatment, score_cols=covariates)

# calculate ATE
treated = matched_data[matched_data[treatment] == 1]
control = matched_data[matched_data[treatment] == 0]
ate = treated[outcome].mean() - control[outcome].mean()
print("estimated ATE:", ate)

# calculate propensity scores manually
lr = LogisticRegression(random_state=42)
lr.fit(data[covariates], data[treatment])
data["propensity_score"] = lr.predict_proba(data[covariates])[:, 1]  # Before matching
matched_data["propensity_score"] = lr.predict_proba(matched_data[covariates])[:, 1]  # After matching

# Plot 
plt.figure(figsize=(10, 6))
# before matching
sns.kdeplot(data=data[data[treatment] == 1]["propensity_score"], label="Treated (Before)", color="blue")
sns.kdeplot(data=data[data[treatment] == 0]["propensity_score"], label="Control (Before)", color="orange")
# bfter matching
sns.kdeplot(data=matched_data[matched_data[treatment] == 1]["propensity_score"], label="Treated (After)", color="blue", linestyle="--")
sns.kdeplot(data=matched_data[matched_data[treatment] == 0]["propensity_score"], label="Control (After)", color="orange", linestyle="--")
plt.title("propensity score distribution before and after matching")
plt.xlabel("propensity score")
plt.ylabel("density")
plt.legend()
plt.show()

# Save matched data
matched_data.to_csv("lalonde_matched.csv", index=False)

# Verify
print("Saved lalonde_matched.csv with", len(matched_data), "rows. Columns:", matched_data.columns.tolist())