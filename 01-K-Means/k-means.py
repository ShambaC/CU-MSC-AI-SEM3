import pathlib
import pandas as pd
import numpy as np

DATASET_URI = f"{pathlib.Path().resolve()}\\..\\Dataset\\IRIS\\iris.data"
DATASET_HEADERS = ['sepal length', 'sepal width', 'petal length', 'petal width', 'type']

datasetFile = open(DATASET_URI)
dataFrame = pd.read_csv(datasetFile, header=None, names=DATASET_HEADERS)

print(dataFrame)