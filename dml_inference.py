import pandas as pd
import numpy as np
from causalml.inference.meta import BaseTLearner
from sklearn.ensemble import RandomForestRegressor

# setting random seed for reproducibility
np.random.seed(42)

# loading the dataset
df = pd.read_csv("dml_kaggle.csv")

# define features, treatment, and outcome
X = df[["Quantity", "Price", "Customer ID"]]  # Covariates + control
T = df["T"]  # Treatment (1: UK, 0: non-UK)
y = df["Y"]  # Outcome (total purchase amount)

# initialize BaseTLearner
t_learner = BaseTLearner(
    learner=RandomForestRegressor(n_estimators=100, random_state=42)
)

# fiting  BaseTLearner
t_learner.fit(X, T, y)

# estimate ATE
ate = t_learner.estimate_ate(X, T, y)
print(f"Average Treatment Effect (ATE): {ate[0].item():.2f}")
print(f"ATE 95% Confidence Interval: [{ate[1].item():.2f}, {ate[2].item():.2f}]")