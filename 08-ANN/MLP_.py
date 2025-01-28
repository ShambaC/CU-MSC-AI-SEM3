import numpy as np
import pandas as pd
import pathlib

from backend_MLP import MLP
from sklearn.model_selection import train_test_split

DATASET_URI = f"{pathlib.Path().resolve()}\\..\\Dataset\\IRIS\\iris.data"
DATASET_HEADERS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
FLOWER_TYPES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Open dataset file and read it using pandas
datasetFile = open(DATASET_URI)
df = pd.read_csv(datasetFile, header=None, names=DATASET_HEADERS)
datasetFile.close()
del datasetFile

X = pd.get_dummies(df, columns=['type'], dtype=np.uint8)
y = X[['type_Iris-setosa', 'type_Iris-versicolor', 'type_Iris-virginica']]
X = X.drop(columns=['type_Iris-setosa', 'type_Iris-versicolor', 'type_Iris-virginica'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

lr = 0.9
epochs = 100

model = MLP('4-5-3')
model.compile(lr)
model.fit(X_train, y_train, epochs)


print("Predictions: ")
for i in range(5) :
    a = X_test.iloc[i]
    res = model.predict(a)

    res_str = FLOWER_TYPES[np.argmax(res)]
    og_str = FLOWER_TYPES[np.argmax(y_test.iloc[i])]

    print(f"Count: {i+1} | Actual value: {og_str}, Predicted value: {res_str}")