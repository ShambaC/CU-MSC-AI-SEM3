import numpy as np
import pandas as pd

from backend import Perceptron
from sklearn.model_selection import train_test_split

X = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
])

y = np.array([
    [-1],
    [1],
    [1],
    [-1],
    [1],
    [-1],
    [-1],
    [1]
])

weights = np.array([0, 0, 0, 0])

percept = Perceptron(X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

percept.fit(X_train, y_train, 50, 0.5, 1)
percept.evaluate(X_test, y_test)