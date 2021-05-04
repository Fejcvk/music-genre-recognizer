import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import random
from dataset.dataprovider import DataProvider
from sklearn.model_selection import GridSearchCV


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


class Predictor:
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

    def train_ffnn(self, epochs: int, x_train: np.ndarray, y_train: np.ndarray, batch_size: int = 64,
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

    def train_svm(self, gamma: float, C: float, x_train: np.ndarray, y_train: np.ndarray, random_state:int, kernel:str):
        model = svm.SVC(C=C,gamma=gamma, kernel=kernel, random_state=random_state)
        model.fit(x_train, y_train)
        # scores = cross_validate(model,x_train,y_train,scoring='f1_macro',cv=10)
        # print(scores)
        self.model = model

    def predict(self, X, Y):
        return self.model.evaluate(X, Y)

    def grid_search_for_svm(self, x_train, y_train):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(x_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(X_test)
        print(confusion_matrix(y_true=Y_test, y_pred=grid_predictions))
        print(classification_report(Y_test, grid_predictions))
        self.model = grid

    def visualize(self, model_name: str):
        title_dict = {'count':self.dataset.shape[0], 'model_type':model_name}
        x_label = 'Iterations'
        for metric in ['accuracy', 'loss']:
            draw_plot(title_dict, is_train=True, x_label=x_label, y_label=metric, data=self.history.history[metric])
            draw_plot(title_dict, is_train=False, x_label=x_label, y_label=metric,data=self.history.history['val_'+ metric])


dataset_provider = DataProvider(
    input_dir='/Users/tomek/Workspace/music-genre-recognizer/input',
    genres='blues classical country disco hiphop jazz metal pop reggae rock'.split(),
    features='chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'.split(),
    visualize=True,
    split_songs=True,
    splitted_song_duration_in_ms=3000
)

SEED = 51
seed_all(SEED=SEED)

predictor = Predictor(dataset=dataset_provider.get_or_create_dataset())
predictor.process_data()

# X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.Y, test_size=.2)
# predictor.train_ffnn(epochs=300, x_train=X_train, y_train=Y_train, dropout=True, x_test=X_test, y_test=Y_test)
# predictor.visualize(model_name='FFNN')
# print(predictor.model.evaluate(X_test, Y_test))

# draw_distribution_plot(Y_train)
# draw_distribution_plot(Y_test)

#Optimal hyperparameters C=10 gamma=0.1 found by gridsearch
X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.dataset.label, test_size=.2)
predictor.train_svm(gamma=0.1,C=10, kernel="rbf",random_state=SEED,x_train=X_train,y_train=Y_train)
print(f"SVN accuracy = {predictor.model.score(X_test, Y_test)}")
print(classification_report(y_pred=predictor.model.predict(X_test), y_true=Y_test))