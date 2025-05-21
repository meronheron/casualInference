import pandas as pd
import numpy as np
from causalml.inference.meta import BaseTLearner
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

def run_dml_inference():
    # setting random seed for reproducibility
    np.random.seed(42)

    # loading the csv dataset
    df = pd.read_csv("dml_kaggle.csv")

    # define covariates, treatment, and outcome
    X = df[["Quantity", "Price", "Customer ID"]]
    T = df["T"]
    y = df["Y"]

    # initialize BaseTLearner
    t_learner = BaseTLearner(
        learner=RandomForestRegressor(n_estimators=100, random_state=42)
    )

    # fitting BaseTLearner
    t_learner.fit(X, T, y)

    # estimate ATE
    ate = t_learner.estimate_ate(X, T, y)

    # estimate CATE
    cate = t_learner.predict(X, T, y)

    # create DataFrame for CATE estimates
    cate_df = pd.DataFrame({
        'CATE': cate.flatten(),#stores individual treatment effects in a column named CATE.
        'Treatment': T
    })

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=cate_df[cate_df['Treatment'] == 1]['CATE'], label='Treated (UK)', color='green', linewidth=1, ax=ax)
    sns.kdeplot(data=cate_df[cate_df['Treatment'] == 0]['CATE'], label='Control (non-UK)', color='blue', linewidth=1, ax=ax)
    plt.axvline(x=ate[0].item(), color='red', linestyle='--', linewidth=0.5, label=f'ATE = {ate[0].item():.2f}')
    plt.title('CATE Distribution for Treated and Control Groups')
    plt.xlabel('CATE (Effect of UK vs. non-UK on Purchase Amount)')
    plt.ylabel('Density')
    plt.legend()

    # prepare output
    output = f"Average Treatment Effect (ATE): {ate[0].item():.2f}\nATE 95% Confidence Interval: [{ate[1].item():.2f}, {ate[2].item():.2f}]"
    
    return fig, output

if __name__ == "__main__":
    fig, output = run_dml_inference()
    print(output)
    plt.show()