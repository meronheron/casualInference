import pandas as pd
import numpy as np
from causalml.inference.meta import BaseTLearner
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
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

# CATE estimate individual treatment effects) for each observation
cate = t_learner.predict(X, T, y)
#creates a DataFrame to organize CATE estimates and treatment status (T) for plotting.
cate_df = pd.DataFrame({
    'CATE': cate.flatten(),
    'Treatment': T
})
# Plotting CATE estimates
plt.figure(figsize=(10, 6))
sns.kdeplot(data=cate_df[cate_df['Treatment'] == 1]['CATE'], label='Treated (UK)', color='blue')
sns.kdeplot(data=cate_df[cate_df['Treatment'] == 0]['CATE'], label='Control (non-UK)', color='orange')
plt.axvline(x=ate[0].item(), color='red', linestyle='--', label=f'ATE = {ate[0].item():.2f}')

