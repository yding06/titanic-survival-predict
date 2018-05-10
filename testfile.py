import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier

class titanicClassifier:
    def __init__(self,k=90):
        self.model = None
        self.k = k
        self.data = pd.read_csv('titanic_filled.csv')

    def train(self):
        y = self.data.iloc[0::, 0].values
        X = self.data.iloc[0::, 1::].values
        X = np.delete(X, 5, 1)
        ada = AdaBoostClassifier(n_estimators=self.k)
        self.model = ada.fit(X, y)

    def score(self, x_test, y_test):
        X_test = np.delete(x_test,5,1)
        train_prediction = self.model.predict(X_test)
        acc = accuracy_score(y_test, train_prediction)
        return acc

test_df = pd.read_csv('titanic_filled.csv')
x_test = test_df.iloc[:,1:].values
y_test = test_df.iloc[:,0].values

tc = titanicClassifier()
tc.train()
accuracy = tc.score(x_test,y_test)
print(accuracy)

