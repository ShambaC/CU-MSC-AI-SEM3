"""
Backend for perceptron
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union

class Perceptron() :
    """Perceptron class"""
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.weights = np.random.uniform(-0.5, 0.5, self.num_inputs+1)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], epochs: int=20, lr: float=2e-2, thresh: float=1) -> None :
        """Method to fit the perceptron to a dataset"""
        ...

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> float :
        """Method to predict test data"""
        ...

    def __call__(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwds):
        return self.predict(X)