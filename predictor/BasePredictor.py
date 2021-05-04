import random
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from dataset.dataprovider import DataProvider


def seed_all(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)


def draw_distribution_plot(labels):
    plt.figure()
    labels.value_counts().plot.barh()
    plt.show()


def draw_plot(title_dict: {}, is_train:bool, x_label:str, y_label: str, data):
    phase = 'Train' if is_train is True else 'Validation'
    plt.title(f"{phase} accuracy for {title_dict['count']} data for {title_dict['model_type']}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(data)
    plt.show()


class BasePredictor(ABC):
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.X = None
        self.Y = None
        self.model = None
        self.history = None
        self.print = False

    def process_data(self):
        dataset = self.dataset
        if self.print:
            print(dataset.head)
            print(dataset.shape)

        # Encode the labels
        self.Y = pd.get_dummies(dataset.label).to_numpy()
        if self.print:
            print(self.Y[10])

        # Scaling the data
        scaler = StandardScaler()
        self.X = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))
        if self.print:
            print(self.X[10])

    def train(self, *args):
        pass

    def predict(self, X):
        pass

    def visualize(self, *args):
        pass


dataset_provider = DataProvider(
    input_dir='/Users/tomek/Workspace/music-genre-recognizer/input',
    genres='blues classical country disco hiphop jazz metal pop reggae rock'.split(),
    features='chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'.split(),
    visualize=True,
    split_songs=True,
    splitted_song_duration_in_ms=3000
)

# draw_distribution_plot(Y_test)
# draw_distribution_plot(Y_train)
