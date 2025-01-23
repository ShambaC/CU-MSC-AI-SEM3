import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pathlib
from tqdm import tqdm

def diana(data: np.ndarray, clusters : int=3) :
    ...