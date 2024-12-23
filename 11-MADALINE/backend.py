"""
Backend for MADALINE network
"""

import numpy as np
from tqdm import tqdm

class MADALINE :
    def __init__(self, num_layers : int, layer_data : list) -> None:
        self.num_layers = num_layers
        self.layer_data = layer_data

        # Initialize layers
        self.layer_in = []
        self.layer_out = []
        self.bias_weight = []
        for i in range(num_layers) :
            layer = np.zeros((layer_data[i], 1))
            layer_b = np.zeros((layer_data[i], 1))
            layer_c = np.random.uniform(-0.5, 0.5, (layer_data[i], 1))
            self.layer_in.append(layer)
            self.layer_out.append(layer_b)
            self.bias_weight.append(layer_c)

        # Initialize weight matrices
        self.weight_mats = []
        for i in range(num_layers-1) :
            layer = np.random.uniform(-0.5, 0.5, (layer_data[i+1], layer_data[i]))
            self.weight_mats.append(layer)
        
        # TODO: implement parameter calculation

    def compute(self, x_data : np.ndarray, y_data : np.ndarray, epochs : int) :
        """
        Method to fit data to the network
        """
        
        data_count = x_data.shape[0]
        current_epoch = 0
        fitting = True

        while current_epoch - 1 < epochs and fitting :
            current_epoch += 1
            with tqdm(total=data_count) as pbar :

                def row_iter(x : np.ndarray) :

                    for i in range(self.num_layers) :
                        if i == 0 :
                            self.layer_in[i] = x.T
                            self.layer_out[i] = x.T
                            continue

                        intermediate_array = np.matmul(self.weight_mats[i-1], self.layer_out[i-1])
                        self.layer_in[i] = np.add(intermediate_array, self.bias_weight[i])

                        for j in range(self.layer_in[i].shape[0]) :
                            self.layer_out[i][j] = 1 if self.layer_in[i][j] >= 0 else -1

                    if np.array_equal(y_data[current_epoch - 1], self.layer_out[self.num_layers - 1]) :
                        ...

                    pbar.set_description(f"Epoch {current_epoch}")
                    pbar.set_postfix_str("")
                    pbar.update(1)
                
                np.apply_along_axis(row_iter, 1, x_data)


def create_madaline(topology : str) :
    """
    Method to parse topology of a network and return the madaline object

    pass topology in the following format:\n
    `num_nodes-num_nodes-...-num_nodes`
    """
    topology = topology.strip()

    if topology.startswith("-") or topology.endswith("-") :
        raise ValueError("Incorrect formatting of topology(Starts or ends with '-')")
    
    layer_data = [int(token) for token in topology.split("-")]

    return MADALINE(len(layer_data), layer_data)