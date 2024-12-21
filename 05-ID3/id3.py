import numpy as np
import pandas as pd
import click

from tqdm import tqdm

def dataset_entropy(df: pd.DataFrame, label: str, class_list: set | list) -> float :
    """Method to calculate the entropy for the whole dataset

    Args
        df:
            The training dataframe
        label:
            The class label
    Returns
        entropy for the dataset
    """

    total_data = df.shape[0]
    entropy = 0.0

    for c in class_list :
        total_class_count = df[label].value_counts()[c].item()
        class_prob = total_class_count / total_data
        class_entropy = - (class_prob) * np.log2(class_prob)
        entropy += class_entropy

    return entropy

def calc_feature_entropy(df: pd.DataFrame, output_label: str, feature_label: str, class_list: set | list) -> float :
    """Method to calculate entropy for a specific feature"""

    total_data = df.shape[0]
    feature_classes = df[feature_label].unique().tolist()
    feature_entropy = 0

    for label_class in feature_classes :
        total_count = df[feature_label].value_counts()[label_class].item()
        local_entropy = 0

        for c in class_list :
            class_count = df[df[feature_label] == label_class][output_label].value_counts()[c].item()
            prob = class_count / total_count
            entropy = - (prob) * np.log2(prob)
            local_entropy += entropy

        entropy = (total_count / total_data) * local_entropy
        feature_entropy += local_entropy

    return feature_entropy

@click.command()
@click.option('--file', '-F', help='Absolute location of the dataset')
@click.option('--label', '-L', help='The output feature label of the dataset')
def main(file, label) :
    data = pd.read_csv(file)
    train_df = data.copy()
    class_list = set(data[label].to_list())

    df_list = []
    df_list.append(train_df)

    epoch_ctr = 0
    while True :
        total = len(df_list)
        with tqdm(total=total) as pbar:
            epoch_ctr += 1
            for df in df_list: 
                df_entropy = dataset_entropy(df, label, class_list)
                for feature in tqdm(df.columns) :
                    if feature == label :
                        continue

                    feat_ent = calc_feature_entropy(df, label, feature, class_list)

                pbar.set_description(f"Epoch {epoch_ctr}")
                pbar.update(1)