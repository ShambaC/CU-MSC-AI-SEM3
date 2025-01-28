import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from typing import Union

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def MSE(pred_val : np.ndarray, obs_val : np.ndarray) -> float :
    
    sq_err = np.power(np.subtract(obs_val, pred_val), 2)
    mse_val = (1 / 2) * np.sum(sq_err)

    return mse_val.item()

class MLP :
    """Simple MLP with only sigmoid activations and SGD"""
    def __init__(self, topology: str, seed: int = 2025) -> None :
        np.random.seed(seed)

        print(f"Creating MLP for topology {topology}\n")
        topology = topology.strip()

        if topology.startswith("-") or topology.endswith("-") :
            raise ValueError("Incorrect formatting of topology(Starts or ends with '-')")
        
        self.topology : list[int] = [int(token) for token in topology.split("-")]

    def compile(self, learning_rate: float = 0.9) -> None :
        self.num_layers = len(self.topology)
        self.learning_rate = learning_rate

        print("Building layers")
        self.layers_data : list[np.ndarray] = []
        self.layer_err_data : list[np.ndarray] = []
        for i in range(self.num_layers) :
            layer = np.zeros((1, self.topology[i]))
            self.layers_data.append(layer)
            self.layer_err_data.append(layer)

        print("Generating weight matrices\n")

        self.weights : list[np.ndarray] = []
        self.biases : list[np.ndarray] = []
        self.parameters = 0

        for i in trange(0, self.num_layers-1, 1, ascii=' --', colour='green') :
            weight_mat = np.random.uniform(-1, 1, (self.topology[i], self.topology[i+1]))
            self.weights.append(weight_mat)
            self.parameters += self.topology[i] * self.topology[i+1]

            bias = np.random.uniform(-0.5, 0.5, (1, self.topology[i+1]))
            self.biases.append(bias)
            self.parameters += self.topology[i+1]
        
        print("Done Generating weight matrices")
        print("Done compiling model")
        print(f"Parameters: {self.parameters}")
        print('#' * 60)
        print('\n')

    def fit(self, X : Union[np.ndarray, pd.DataFrame], y : Union[np.ndarray, pd.DataFrame, pd.Series], epochs: int) -> None :
        
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
            
        
        loss = 1.0
        for epoch in range(1, epochs + 1, 1) :
            for row in trange(X.shape[0], ascii=" --", colour='green', desc=f'Epoch: {epoch}', postfix=f'Loss: {loss}') :
                self.layers_data[0] = X[row].reshape((1, X.shape[1]))

                # Forward propagation
                for i in range(self.num_layers - 1) :
                    z_data = np.matmul(self.layers_data[i], self.weights[i])
                    z_data_biased = np.add(z_data, self.biases[i])
                    z_out = sigmoid(z_data_biased)
                    self.layers_data[i+1] = z_out

                loss = MSE(self.layers_data[-1], y[row])

                # Error calculation
                # Output layer
                self.layer_err_data[-1] = self.layers_data[-1] * (1 - self.layers_data[-1]) * (y[row] - self.layers_data[-1])
                # Hidden layers
                for i in range(self.num_layers - 2, 1) :
                    a = self.layers_data[i].T
                    b = self.weights[i].T
                    c = self.layer_err_data[i+1].T

                    res_err = a * (1 - a) * np.matmul(c, b)
                    self.layer_err_data[i] = res_err.T

                # backpropagation
                for i in range(self.num_layers - 1) :
                    # Weights
                    a = [[u * v for u in self.layer_err_data[i+1][0]] for v in self.layers_data[i][0]]
                    del_weight = np.array(a) * self.learning_rate

                    self.weights[i] += del_weight

                    # biases
                    self.biases[i] += self.learning_rate * self.layer_err_data[i+1]

    def predict(self, X : Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[float, np.ndarray] :

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
            

        self.layers_data[0] = X.reshape((1, X.shape[0]))

        for i in range(self.num_layers - 1) :            
            z_data = np.matmul(self.layers_data[i], self.weights[i])
            z_data_biased = np.add(z_data, self.biases[i])
            z_out = sigmoid(z_data_biased)
            self.layers_data[i+1] = z_out

        res = self.layers_data[-1]
        return res
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None :
        
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
            
        print("WARNING! ACCURACY WILL ALWAYS BE 0 IN THIS METHOD AS THE ACTIVATION FUNCTION IS SIGMOID")

        loss = 0
        for rowX, rowY in tqdm(zip(X, y), ascii=' --', colour='green') :
            res = self.predict(rowX)
            loss += MSE(res, rowY)

        loss = loss / len(y)
        print(f"Loss: {loss}")