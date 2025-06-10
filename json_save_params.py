import json

import numpy as np

# Example: this was params will be stored:

# position weight matrix:
tmp = np.array([[3 / 8, 1 / 8, 2 / 8, 2 / 8], [1 / 10, 2 / 10, 3 / 10, 4 / 10], [1 / 7, 2 / 7, 1 / 7, 3 / 7]])
Theta = tmp.T

# background distribution
ThetaB = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

params = {
    "w": 3,
    "alpha": 0.5,
    "k": 10,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

# Note: matrices cannot be provided above — they are converted to lists here
# (later, after loading, just convert them back to a matrix using e.g. np.asarray(.))

with open('data/params_set1.json', 'w') as outfile:
    json.dump(params, outfile)