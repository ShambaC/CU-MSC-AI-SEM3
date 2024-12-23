import numpy as np
import pandas as pd

from backend import Perceptron

X = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
])

y = np.array([[1], [-1], [-1], [-1]])

weights = np.array([0, 0, 0, 0])

percept = Perceptron(X.shape[1], weights)
percept.fit(X, y, 20, 1, 1)

res = percept.predict([-1, 1, -1])
print(res)