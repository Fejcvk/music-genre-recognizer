from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from predictor.BasePredictor import BasePredictor, seed_all, dataset_provider


class SVMPredictor(BasePredictor):
    def train(self, gamma: float, C: float, x_train: np.ndarray, y_train: np.ndarray, random_state: int,
                  kernel: str):
        model = svm.SVC(C=C, gamma=gamma, kernel=kernel, random_state=random_state)
        model.fit(x_train, y_train)
        # scores = cross_validate(model,x_train,y_train,scoring='f1_macro',cv=10)
        # print(scores)
        self.model = model

    def grid_search_for_svm(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(x_train, y_train)
        print(grid.best_estimator_)
        grid_predictions = grid.predict(x_test)
        print(confusion_matrix(y_true=y_test, y_pred=grid_predictions))
        print(classification_report(y_test, grid_predictions))
        self.model = grid

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def visualize(self, *args):
        pass


SEED = 51
seed_all(SEED=SEED)

predictor = SVMPredictor(dataset=dataset_provider.get_or_create_dataset())
predictor.process_data()

# Optimal hyperparameters C=10 gamma=0.1 found by gridsearch
X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.dataset.label, test_size=.2)
predictor.train_svm(gamma=0.1, C=10, kernel="rbf", random_state=SEED, x_train=X_train, y_train=Y_train)
print(f"SVN accuracy = {predictor.model.score(X_test, Y_test)}")
print(classification_report(y_pred=predictor.model.predict(X_test), y_true=Y_test))
