import numpy as np
import pandas as pd

from backend import Perceptron
from sklearn.model_selection import train_test_split

X = pd.read_csv('3_NAND.csv')
# X = pd.read_csv('3_NOR.csv')
# X = pd.read_csv('3_XOR.csv')
# X = pd.read_csv('3_XNOR.csv')

y = X.pop('output')

percept = Perceptron(X.shape[1])
percept.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

percept.fit(X_train, y_train, 20, 1, 1)
percept.evaluate(X_test, y_test)