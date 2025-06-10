import numpy as np
import json

with open("data/params_set1.json", 'r') as inputfile:
    params = json.load(inputfile)

w = params['w']
k = params['k']
alpha = params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB = np.asarray(params['ThetaB'])

print(params)

