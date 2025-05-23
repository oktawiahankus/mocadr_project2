# EM algorithm

import numpy as np
from PIL.features import check
from numpy.ma.core import indices

from modules.generate_data import generate_data

np.random.seed(42)

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
    # to też pewnie można jakoś wektorowo, ale tutaj widać, co się dzieje
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
    k, w = X.shape

    row_indices = np.intp(X - 1)
    col_indices = np.tile(np.arange(w), (k, 1))

    if est_alpha:
        alpha = 0.5 # zapominamy, że znamy alpha i je estymujemy, a tu inicjalizacja

    dist = 1
    iter = 0
    while dist > err and iter < max_iter:
        iter += 1

        # wartości Theta, dostosowane do obliczeń
        Theta_c = Theta[row_indices, col_indices]
        ThetaB_c = ThetaB[row_indices]

        f = np.prod(Theta_c, axis = 1)
        fB = np.prod(ThetaB_c, axis = 1)

        # te, Z co są równe 1
        latent = f * alpha / ((f * alpha) + (fB * (1 - alpha)))
        # żeby umożliwić działania wektorowe
        latent = latent[:, np.newaxis]
        Theta_lam = np.sum(latent)
        ThetaB_lam = np.sum(1 - latent) * w

        alpha_est = alpha

        if est_alpha:
            alpha_est = np.mean(latent)

        Theta_est = Theta.copy()
        ThetaB_est = ThetaB.copy()

        for i in [1, 2, 3, 4]:
            mask = X == i
            row = i - 1

            ThetaB_est[row] = np.sum(mask * (1 -  latent)) / ThetaB_lam
            Theta_est[row, :] = np.sum(mask * latent, axis=0) / Theta_lam

        # na razie takie sprawdzenie
        # może trzeba inną odległość ??
        dist = np.sum((Theta - Theta_est)**2) + np.sum((ThetaB - ThetaB_est)**2) + (alpha - alpha_est)**2

        alpha = alpha_est
        Theta = Theta_est
        ThetaB = ThetaB_est

    return {"alpha": alpha,
            "Theta": Theta,
            "ThetaB": ThetaB}

print(EM(some_data))