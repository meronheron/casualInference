import pandas as pd
import numpy as np
from causalml.inference.meta import XLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#Setting random seed for reproducibility
np.random.seed(42)
# loading dataset
df = pd.read_csv("dml_kaggle.csv")