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

def initialize_params(X):
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

    return Theta, ThetaB

check = initialize_params(some_data["data"])
print(check)

def EM(data, est_alpha = False, max_iter = 20, err = 1e-4):
    alpha, X = data.values()
    Theta, ThetaB = initialize_params(X)

    if est_alpha:
        alpha = 0.5 # zapominamy, że znamy alpha i je estymujemy, a tu inicjalizacja

    dist = 1
    iter = 0
    while dist > err or iter < max_iter:
        if est_alpha:
            print("Trzeba wyestymować też alpha")

        # te, Z co są równe 1
        # latent = ...