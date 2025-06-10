import json
import numpy as np
import argparse

# Read-in params

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='File with input data (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='File where the estimated parameters will be saved (default: %(default)s)')
    parser.add_argument('--init-method', default="data", required=False,
                        help='Type of initialization of the distributions (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Should alpha be estimated or not? (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.init_method, args.estimate_alpha


input_file, output_file, init_method, estimate_alpha = ParseArguments()

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

def tv_dist(origin, est):
    return np.sum(np.sum(np.abs(origin - est), axis = 0) / 2)

def initialize_params(X, method = "data"):
    # na wszelki wypadek sobie zmieniamy
    # teraz już musimy, bo generate tak generuje
    X = np.asarray(X)
    k, w = X.shape

    if method == "data":
        ThetaB = np.empty(4)
        Theta = np.empty([4, w])
        for i in [1, 2, 3, 4]:
            mask = X == i
            row = i - 1
            ThetaB[row] = np.sum(mask) / (k * w)
            Theta[row, :] = np.sum(mask, axis=0) / k
        return Theta, ThetaB

    elif method == "uniform":
        # Jednolity rozkład
        Theta = np.ones([4, w]) / 4
        ThetaB = np.ones(4) / 4
        return Theta, ThetaB

    elif method == "random":
        Theta = np.array([np.random.dirichlet(np.ones(4)) for _ in range(w)]).T  # shape (4, w)
        ThetaB = np.random.dirichlet(np.ones(4))
        return Theta, ThetaB

    else:
        raise ValueError("Invalid argument for function initialize_params")


def EM(data, est_alpha=False, max_iter=1000, err=1e-5, init_method="data"):
    alpha, X = data.values()
    Theta, ThetaB = initialize_params(X, method=init_method)
    X = np.asarray(X)
    k, w = X.shape

    row_indices = np.intp(X - 1)
    col_indices = np.tile(np.arange(w), (k, 1))

    if est_alpha:
        alpha = 0.5  # zapominamy, że znamy alpha i je estymujemy, a tu inicjalizacja

    dist = 1
    iter = 0
    history = []

    while dist > err and iter < max_iter:
        iter += 1

        # wartości Theta, dostosowane do obliczeń
        Theta_c = Theta[row_indices, col_indices]
        ThetaB_c = ThetaB[row_indices]

        f = np.prod(Theta_c, axis=1)
        fB = np.prod(ThetaB_c, axis=1)

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

            ThetaB_est[row] = np.sum(mask * (1 - latent)) / ThetaB_lam
            Theta_est[row, :] = np.sum(mask * latent, axis=0) / Theta_lam

        if est_alpha:
            dist = (tv_dist(Theta, Theta_est) + tv_dist(ThetaB, ThetaB_est) + np.abs(alpha - alpha_est) / 2) / (w + 2)
            alpha = alpha_est
        else:
            dist = (tv_dist(Theta, Theta_est) + tv_dist(ThetaB, ThetaB_est)) / (w + 1)

        history.append(dist)
        Theta = Theta_est
        ThetaB = ThetaB_est

    return {"alpha": alpha,
            "Theta": Theta.tolist(),
            "ThetaB": ThetaB.tolist(),
            "history": history}

est_alpha = False

if estimate_alpha == "yes":
    est_alpha = True

estimated_params = EM(data, est_alpha = est_alpha, init_method = init_method)

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)



