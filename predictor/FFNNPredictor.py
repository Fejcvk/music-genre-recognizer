from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from predictor.BasePredictor import BasePredictor, draw_plot, dataset_provider, seed_all
from datetime import datetime


class FFNNPredictor(BasePredictor):
    def train(self,
              epochs: int,
              x_train: np.ndarray,
              y_train: np.ndarray,
              batch_size: int = 64,
              dropout: bool = False,
              x_test: np.ndarray = None,
              y_test: np.ndarray = None,
              lr: float = 0.001):

        print(x_train.shape)
        print(y_train.shape)

        # Simple FFNN
        # 3 hidden layers with relu activation + output layer with softmax activation
        ffnn = keras.models.Sequential()

        ffnn.add(layers.Dense(256, activation='relu'))
        ffnn.add(layers.Dropout(.3, input_shape=(x_train.shape[1],)))
        ffnn.add(layers.Dense(128, activation='relu'))
        if dropout:
            ffnn.add(layers.Dropout(.3, input_shape=(x_train.shape[1],)))
        ffnn.add(layers.Dense(64, activation='relu'))
        ffnn.add(layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=lr)

        ffnn.compile(optimizer=opt,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        ffnn.build(input_shape=x_train.shape)
        ffnn.summary()
        self.history = ffnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_test, y_test))
        self.model = ffnn

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path_to_save: str):
        self.model.save(path_to_save)

    def load(self, path):
        self.model = keras.models.load_model(path)


SEED = 35
seed_all(SEED=SEED)

predictor = FFNNPredictor(dataset_provider)

X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.Y, test_size=.2)

predictor.train(epochs=200, batch_size=32, x_train=X_train, y_train=Y_train, dropout=True, x_test=X_test, y_test=Y_test,
                lr=0.0001)
predictor.visualize(model_name='FFNN')
# predictor.save(path_to_save=f'../models/ffnn/model-{datetime.now()}-90%')

# predictor.load('../models/ffnn/model-2021-05-06 20:28:31.848718')
preds = predictor.model.predict_classes(X_test)
y_true = np.argmax(Y_test, axis=1)
print(confusion_matrix(y_true=y_true, y_pred=preds))
print(classification_report(y_true, preds))
