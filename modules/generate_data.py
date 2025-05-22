import numpy as np

# params --> jakiś specjalny plik json
# od razu funkcja taka, żeby z tym działało,
# ale wczytanie danych do formatu pythona przed
# tą funkcją

# czy my checemy jakąś osobną funkcję do zamiany
# danych na json??? (chyba)

# wartości, z których chcemy robić ciągi
values = np.array([1, 2, 3, 4])

# position weight matrix:
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

# setup to są te słowniki params
def generate_data(setup):
    w, alpha, k, Theta, ThetaB = setup.values()
    Theta = np.asarray(Theta)
    ThetaB = np.asarray(ThetaB)

    X = np.random.rand(k, w)
    mask = X <= alpha

    for i in range(w):
        col_mask = mask[:,i]
        m = np.sum(col_mask)
        m_val = np.random.choice(values, m, p = Theta[:,i])

        X[:,i][col_mask] = m_val

    b = np.sum(~mask)
    b_val = np.random.choice(values, b, p = ThetaB)
    X[~mask] = b_val

    return {"alpha": alpha,
            "data": X}

print(generate_data(params))