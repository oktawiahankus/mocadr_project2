# EM algorithm

import numpy as np
from PIL.features import check

from modules.generate_data import generate_data

tmp = np.array([[3/8,1/8,2/8,2/8],[1/10,2/10,3/10,4/10],[1/7,2/7,1/7,3/7]])
Theta = tmp.T

# background distribution
ThetaB=np.array([1/4,1/4,1/4,1/4])

params = {
    "w" : 3,
    "alpha" : 0.5,
    "k" : 10,
    "Theta" : Theta.tolist(),
    "ThetaB" : ThetaB.tolist()
    }

# trzeba będzie dbać o format X, jeżeli wczytywany z pliku!!!
some_data = generate_data(params)

def initialize_params(data):
    X = data["data"].copy()
    # na wszelki wypadek sobie zmieniamy
    X = np.asarray(X)
    k, w = X.shape

    ThetaB = np.empty(4)
    Theta = np.empty([4, w])

    for i in [1, 2, 3, 4]:
        mask = X == i
        row = i - 1

        ThetaB[row] = np.sum(mask) / (k * w)
        Theta[row, :] = np.sum(mask, axis=0) / k

    data["Theta"] = Theta
    data["ThetaB"] = ThetaB
    return data

check = initialize_params(some_data)
print(check)