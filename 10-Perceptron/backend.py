"""
Backend for perceptron
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union

class Perceptron() :
    """Perceptron class"""
    def __init__(self, num_inputs: int, weights: np.ndarray=None):
        self.num_inputs = num_inputs
        if weights == None :
            self.weights = np.random.uniform(-0.5, 0.5, self.num_inputs+1)
        else :
            if len(weights) != num_inputs + 1 :
                self.weights = np.random.uniform(-0.5, 0.5, self.num_inputs+1)
            else :
                self.weights = weights.ravel()


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], epochs: int=20, lr: float=2e-2, thresh: float=1) -> None :
        """Method to fit the perceptron to a dataset
        
        Args
        ----
            X:
                Dataset to fit
            y:
                output of the dataset
            epochs:
                Number of times the learning is repeated before stopping
            lr:
                Learning rate
            thresh:
                THreshold for the activation function
        """

        # Checks
        if isinstance(X, pd.DataFrame) :
            if 'object' in X.dtypes.values :
                raise ValueError("Object type data in dataset. Please only use numerical data.")
            X = X.to_numpy()
        if isinstance(X, np.ndarray) :
            if X.dtype == 'O' :
                raise ValueError("Object type data in dataset. Please only use numerical data.")
        else :
            X = np.array(X)
            if X.dtype == 'O' :
                raise ValueError("Object type data in dataset. Please only use numerical data.")
        
        if isinstance(y, pd.DataFrame) :
            if 'object' in y.dtypes.values :
                raise ValueError("Object type data in dataset. Please only use numerical data.")
            y = y.to_numpy()
        if isinstance(y, np.ndarray) :
            if y.dtype == 'O' :
                raise ValueError("Object type data in dataset. Please only use numerical data.")
        else :
            y = np.array(y)
            if y.dtype == 'O' :
                raise ValueError("Object type data in dataset. Please only use numerical data.")


        is_fitting = False
        epoch_ctr = 0

        self.thresh = thresh

        while epoch_ctr < epochs and is_fitting :
            epoch_ctr += 1
            with tqdm(total=X.shape[0]) as pbar :

                has_error = []

                def row_iter(x: np.ndarray, idx: int) :
                    res_bool = False

                    a = np.insert(x, 0, 1)
                    y_in = np.sum(np.multiply(a, self.weights))

                    y_out = 0
                    if y_in > thresh : y_out = 1
                    elif y_in >= -thresh and y_in <= thresh : y_out = 0
                    elif y_in < -thresh : y_out = -1

                    if y_out != y[idx] :
                        self.weights = np.add(self.weights, np.multiply(x, (lr * y[idx])))
                        res_bool = True

                    pbar.set_description(f'Epoch {epoch_ctr} ')
                    pbar.set_postfix_str("")
                    pbar.update(1)

                    return res_bool

                for idx, row_x in enumerate(X) :
                    res = row_iter(row_x, idx)
                    has_error.append(res)
                
                if all(has_error) :
                    is_fitting = False
                    print("Perceptron fitted, stopping training!")
                    return
                
        print("Max number of epochs reached. Stopping training!")
        return
        

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> float :
        """Method to predict test data"""
        ...

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None :
        ...

    def __call__(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwds):
        return self.predict(X)