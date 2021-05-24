from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from predictor.BasePredictor import BasePredictor, dataset_provider, seed_all

from tensorflow import keras
import numpy as np


class CNNPredictor(BasePredictor):

    def process_data(self):
        super().process_data()
        self.X = self.X.reshape(1000, 10, 26)
        self.X = self.X[..., np.newaxis]
        self.Y = self.Y[0:10000:10]

    def save(self, path_to_save:str):
        self.model.save(path_to_save)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int,
              x_test: np.ndarray = False,
              y_test: np.ndarray = False,
              dropout=False,
              batch_size: int = 64,
              lr=0.0001):
        model = keras.Sequential()

        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
        model.add(keras.layers.BatchNormalization())

        # model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
        # model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
        # model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        if dropout:
            model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.build(input_shape=input_shape)
        model.summary()

        self.history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(x_test, y_test))
        self.model = model


predictor = CNNPredictor(dataset_provider=dataset_provider)
SEED = 35
seed_all(SEED=SEED)
X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.Y, test_size=.2)

predictor.train(x_train=X_train,
                y_train=Y_train,
                x_test=X_test,
                y_test=Y_test,
                batch_size=16,
                epochs=50,
                dropout=True,
                lr=0.0003)

# 88% 50 epok bs 16 lr 0.0003

predictor.visualize(model_name='CNN')

preds = predictor.model.predict_classes(X_test)
y_true = np.argmax(Y_test, axis=1)
print(confusion_matrix(y_true=y_true, y_pred=preds))
print(classification_report(y_true, preds))

predictor.save(path_to_save=f'../models/cnn/model-{datetime.now()}-88%')

