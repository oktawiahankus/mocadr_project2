import json
import numpy as np
import argparse

# Read-in params:

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False,
                        help='File with parameters (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False,
                        help='File to save generated data (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output


param_file, output_file = ParseArguments()

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)

def generate_data(setup):
    # wartości, z których chcemy robić ciągi
    values = np.array([1, 2, 3, 4])

    w, alpha, k, Theta, ThetaB = setup.values()
    Theta = np.asarray(Theta)
    ThetaB = np.asarray(ThetaB)

    X = np.empty((k, w))
    mask = np.full((k, w), False)
    # losujemy, z jakiego rozkładu ma pochodzić rząd (próba)
    probs = np.random.rand(k)
    chosen_rows = probs <= alpha
    mask[chosen_rows] = True

    for i in range(w):
        col_mask = mask[:,i]
        m = np.sum(col_mask)
        m_val = np.random.choice(values, m, p = Theta[:,i])

        X[:,i][col_mask] = m_val

    b = np.sum(~mask)
    b_val = np.random.choice(values, b, p = ThetaB)
    X[~mask] = b_val

    return {"alpha": alpha,
            "data": X.tolist()}

gen_data = generate_data(params)

with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)

