import pandas as pd
import numpy as np
from causalml.match import NearestNeighborMatch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def run_psm_inference():
    # setting random seed for reproducibility
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

    # calculate propensity scores
    lr = LogisticRegression(random_state=42)
    lr.fit(data[covariates], data[treatment])
    data["propensity_score"] = lr.predict_proba(data[covariates])[:, 1]
    matched_data["propensity_score"] = lr.predict_proba(matched_data[covariates])[:, 1]

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=data[data[treatment] == 1]["propensity_score"], label="Treated (Before)", color="blue", ax=ax)
    sns.kdeplot(data=data[data[treatment] == 0]["propensity_score"], label="Control (Before)", color="orange", ax=ax)
    sns.kdeplot(data=matched_data[matched_data[treatment] == 1]["propensity_score"], label="Treated (After)", color="blue", linestyle="--", ax=ax)
    sns.kdeplot(data=matched_data[matched_data[treatment] == 0]["propensity_score"], label="Control (After)", color="orange", linestyle="--", ax=ax)
    plt.title("Propensity score distribution before and after matching")
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend()

    # prepare output
    output = f"Estimated ATE: {ate:.2f}\nMatched data has {len(matched_data)} rows. Columns: {matched_data.columns.tolist()}"
    
    return fig, output

if __name__ == "__main__":
    fig, output = run_psm_inference()
    print(output)
    plt.show()