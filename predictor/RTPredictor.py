from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from predictor.BasePredictor import BasePredictor, dataset_provider, seed_all


class RTPredictor(BasePredictor):
    def train(self, n_estimators:int, X:np.ndarray, Y:np.ndarray):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        self.model = clf
        self.model.fit(X, Y)


seed = 10
seed_all(SEED = seed)

predictor = RTPredictor(dataset_provider)
X_train, X_test, Y_train, Y_test = train_test_split(predictor.X, predictor.dataset.label.to_numpy(), test_size=.2)
predictor.train(n_estimators=40, X=X_train, Y=Y_train)
print(predictor.model.score(X_test,Y_test))
print(confusion_matrix(Y_test,predictor.model.predict(X_test)))
print(classification_report(y_pred=predictor.model.predict(X_test), y_true = Y_test))
