def euclid_distance(pointA, pointB, size: int = 2) -> float :
    """A method to calculate the euclidean distance between two points

    Parameters
    ----------
    pointA : Any
        The first point
    pointB : Any
        The second point
    size : int, optional
        The dimension of the points or the number of elements in the coordinates. Default value is 2.

    Returns
    -------
    float
        The euclidean distance between the points
    """

    sqDistance = 0.0

    for i in range(size) :
        sqDistance += pow(pointA[i] - pointB[i], size)

    distance = float(pow(sqDistance, 1.0 / size))
    return distance


if __name__ == "__main__" :
    k = int(input("Enter the number of clusters: "))

    import pathlib
    import pandas as pd
    import numpy as np

    DATASET_URI = f"{pathlib.Path().resolve()}\\..\\Dataset\\IRIS\\iris.data"
    DATASET_HEADERS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']

    datasetFile = open(DATASET_URI)
    dataFrame = pd.read_csv(datasetFile, header=None, names=DATASET_HEADERS)
    datasetFile.close()

    data_amount = dataFrame.shape[0]

    import random
    cluster_indices = random.sample(range(data_amount), k)
    cluster_rows = [dataFrame.loc[x] for x in cluster_indices]
    cluster_dict = dict.fromkeys(cluster_rows, [])

    for index, row in dataFrame.iterrows() :
        cluster_distance = np.array([
            euclid_distance(
                [row['sepal_length'], row['sepal_width'], row['petal_length'], row['petal_width']],
                [cluster['sepal_length'], cluster['sepal_width'], cluster['petal_length'], cluster['petal_width']],
                4)
                for cluster in cluster_rows
            ])
        
        chosen_cluster = cluster_rows[np.argmin(cluster_distance)]
        cluster_dict[chosen_cluster].append(row)