import numpy as np
import pandas as pd

from backend import create_madaline
from sklearn.model_selection import train_test_split

X = pd.read_csv('3_NAND.csv')
# X = pd.read_csv('3_NOR.csv')
# X = pd.read_csv('3_XOR.csv')
# X = pd.read_csv('3_XNOR.csv')

y = X.pop('output')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MADALINE = create_madaline("3-2-1")

MADALINE.fit(X_train, y_train, 20, 0.5)
MADALINE.evaluate(X_test, y_test)