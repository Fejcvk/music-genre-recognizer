from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from predictor.BasePredictor import BasePredictor, draw_plot, dataset_provider


class FFNNPredictor(BasePredictor):
    def train(self, epochs: int, x_train: np.ndarray, y_train: np.ndarray, batch_size: int = 64,
              dropout: bool = False, x_test=None, y_test=None):
        print(x_train.shape)
        print(y_train.shape)

        # Simple FFNN
        # 3 hidden layers with relu activation + output layer with softmax activation
        ffnn = keras.models.Sequential()
        if dropout:
            ffnn.add(layers.Dropout(.3, input_shape=(x_train.shape[1],)))
        ffnn.add(layers.Dense(256, activation='relu'))
        ffnn.add(layers.Dense(128, activation='relu'))
        ffnn.add(layers.Dense(128, activation='relu'))
        ffnn.add(layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=0.001)

        ffnn.compile(optimizer=opt,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.history = ffnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_test, y_test))
        self.model = ffnn

    def predict(self, X:np.ndarray) -> np.ndarray:
        self.model.predict(X)

    def visualize(self, model_name: str):
        title_dict = {'count': self.dataset.shape[0], 'model_type': model_name}
        x_label = 'Iterations'
        for metric in ['accuracy', 'loss']:
            draw_plot(title_dict, is_train=True, x_label=x_label, y_label=metric, data=self.history.history[metric])
            draw_plot(title_dict, is_train=False, x_label=x_label, y_label=metric,
                      data=self.history.history['val_' + metric])


predictor = FFNNPredictor(dataset=dataset_provider.get_or_create_dataset())
X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.Y, test_size=.2)
predictor.train_ffnn(epochs=300, x_train=X_train, y_train=Y_train, dropout=True, x_test=X_test, y_test=Y_test)
predictor.visualize(model_name='FFNN')
print(predictor.model.evaluate(X_test, Y_test))
