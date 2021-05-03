import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from dataset.dataprovider import DataProvider


class Predictor:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.X = None
        self.Y = None
        self.model = None
        self.history = None

    def process_data(self):
        dataset = self.dataset
        print(dataset.head)
        print(dataset.shape)

        # Encode the labels
        self.Y = pd.get_dummies(dataset.label).to_numpy()
        print(self.Y[10])

        # Scaling the data
        scaler = StandardScaler()
        self.X = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))
        print(self.X[10])

    def train_ffnn(self, epochs: int, batch_count: int = 200, dropout: bool = False):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=.2)
        print(X_train.shape)
        print(Y_train.shape)
        batch_size = int(X_train.shape[0] / batch_count)

        # Simple FFNN
        # 3 hidden layers with relu activation + output layer with softmax activation
        ffnn = keras.models.Sequential()
        if dropout:
            ffnn.add(layers.Dropout(.3, input_shape=(X_train.shape[1],)))
        ffnn.add(layers.Dense(256, activation='relu'))
        ffnn.add(layers.Dense(128, activation='relu'))
        ffnn.add(layers.Dense(128, activation='relu'))
        ffnn.add(layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=0.001)

        ffnn.compile(optimizer=opt,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.history = ffnn.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
        self.model = ffnn

    def predict(self, X, Y):
        return self.model.evaluate(X, Y)

    def visualize(self, model_name: str):
        plt.title(f"Train accuracy for {self.dataset.shape[0]} data for {model_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.plot(self.history.history['accuracy'])
        plt.show()

        plt.title(f"Train loss for {self.dataset.shape[0]} data for {model_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.plot(self.history.history['loss'])
        plt.show()

        plt.title(f"Validation Accuracy for {self.dataset.shape[0]} data for {model_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.plot(self.history.history['val_accuracy'])
        plt.show()

        plt.title(f"Validation loss for {self.dataset.shape[0]} data for {model_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.plot(self.history.history['val_loss'])
        plt.show()


dataset_provider = DataProvider(
    input_dir='/Users/tomek/Workspace/music-genre-recognizer/input',
    genres='blues classical country disco hiphop jazz metal pop reggae rock'.split(),
    features='chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'.split(),
    visualize=True,
    split_songs=True,
    splitted_song_duration_in_ms=3000
)

predictor = Predictor(dataset=dataset_provider.get_or_create_dataset())
predictor.process_data()
predictor.train_ffnn(epochs=300, dropout=True)
predictor.visualize(model_name='FFNN'
                    )
