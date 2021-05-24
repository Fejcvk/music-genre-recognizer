import random
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from dataset_provider.DatasetProvider import DataProvider


def seed_all(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)


def draw_distribution_plot(labels):
    plt.figure()
    labels.value_counts().plot.barh()
    plt.show()


def draw_plot(title_dict: {}, is_train:bool, x_label:str, y_label: str, data, metric:str):
    phase = 'Train' if is_train is True else 'Validation'
    plt.title(f"{phase} {metric} for {title_dict['count']} data for {title_dict['model_type']}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(data)
    plt.show()


class BasePredictor(ABC):
    def __init__(self, dataset_provider:DataProvider):
        self.dataset_provider = dataset_provider
        self.dataset = None
        self.X = None
        self.Y = None
        self.model = None
        self.history = None
        self.print = False
        self.prepare()

    def prepare(self):
        self.dataset = self.dataset_provider.get_or_create_dataset()
        self.process_data()

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

    def visualize(self, model_name: str):
        title_dict = {'count': self.dataset.shape[0], 'model_type': model_name}
        x_label = 'Iterations'
        for metric in ['accuracy', 'loss']:
            draw_plot(title_dict, is_train=True, x_label=x_label, y_label=metric, data=self.history.history[metric],
                      metric=metric)
            draw_plot(title_dict, is_train=False, x_label=x_label, y_label=metric,
                      data=self.history.history['val_' + metric], metric=metric)

    def save(self, *args):
        pass

    def load(self, *args):
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
