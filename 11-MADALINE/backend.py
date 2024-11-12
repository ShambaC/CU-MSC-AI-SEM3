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
        layer_in = []
        layer_out = []
        bias_weight = []
        for i in range(num_layers) :
            layer = np.zeros((layer_data[i], 1))
            layer_b = np.zeros((layer_data[i], 1))
            layer_c = np.ones((layer_data[i], 1))
            layer_in.append(layer)
            layer_out.append(layer_b)
            bias_weight.append(layer_c)

        # Initialize weight matrices
        weight_mats = []
        for i in range(num_layers-1) :
            layer = np.random.uniform(-0.5, 0.5, (layer_data[i+1], layer_data[i]))
            weight_mats.append(layer)
        
        # TODO: implement parameter calculation

    def compute(x_data : np.ndarray, y_data : np.ndarray, epochs : int) :
        """
        Method to fit data to the network
        """
        
        data_count = x_data.shape[0]
        current_epoch = 0

        with tqdm(total=data_count) as pbar :

            def row_iter(x) :
                current_epoch += 1

                pbar.set_description(f"Epoch {current_epoch}")
            
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