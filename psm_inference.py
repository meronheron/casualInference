import pandas as pd
from causalml.match import NearestNeighborMatch

# Load preprocessed dataset
data = pd.read_csv("lalonde_preprocessed.csv")

# Define treatment, outcome, and covariates
treatment = "treat"
outcome = "re78"
covariates = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]

# Perform PSM
psm = NearestNeighborMatch(replace=True, random_state=42)
matched_data = psm.match(data=data, treatment_col=treatment, score_cols=covariates)

# Calculate Average Treatment Effect (ATE) manually
treated = matched_data[matched_data[treatment] == 1]
control = matched_data[matched_data[treatment] == 0]
ate = treated[outcome].mean() - control[outcome].mean()
print("Estimated ATE:", ate)

# Save matched data
matched_data.to_csv("lalonde_matched.csv", index=False)

# Verify
print("Saved lalonde_matched.csv with", len(matched_data), "rows. Columns:", matched_data.columns.tolist())