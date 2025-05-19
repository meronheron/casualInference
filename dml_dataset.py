import pandas as pd
import numpy as np
from econml.data.dgps import linear_dgp
# Setting  random seed for reproducibility
np.random.seed(42)
#make synthethic data by using linear_dgp
n_samples = 3000
n_features = 5
data_dict = linear_dgp(n_samples=n_samples, n_x=n_features, n_w=2)
#extract data components
W = data_dict['W']  # Controls
X = data_dict['X']  # Covariates
T = data_dict['T']  # Oreatment (binary)
Y = data_dict['Y']  # 0utcome
# Create a DataFrame
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
df['W0'] = W[:, 0]
df['W1'] = W[:, 1]
df['T'] = T
df['Y'] = Y
# Save the DataFrame to a CSV file
df.to_csv('synthetic_data.csv', index=False)
# Verify
print("Saved synthetic_data.csv with", len(df), "rows. Columns:", df.columns.tolist())

