import pandas as pd
import numpy as np
from causalml.inference.meta import XLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#Setting random seed for reproducibility
np.random.seed(42)
# loading dataset
df = pd.read_csv("dml_kaggle.csv")
#define features, treatment, and outcome
X = df[["Quantity", "Price", "Customer ID"]]  # Covariates + control
T = df["T"]  # Treatment (1: UK, 0: non-UK)
y = df["Y"]  # Outcome (total purchase amount)
# initialize XLearner
x_learner = XLearner(
    learner=RandomForestRegressor(n_estimators=100, random_state=42),
    control_learner=RandomForestRegressor(n_estimators=100, random_state=42),
    treatment_learner=RandomForestRegressor(n_estimators=100, random_state=42),
    propensity_model=RandomForestClassifier(n_estimators=100, random_state=42)
)

# fit XLearner
x_learner.fit(y, T, X=X)