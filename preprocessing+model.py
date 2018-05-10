import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing
dataset = pd.read_csv('titanic_filled.csv')

print (dataset.info())

print (dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print (dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
print (dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 1
print (dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

print (dataset[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4)
print (dataset[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)
print (dataset[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

print (dataset[['magic_feature', 'Survived']].groupby(['magic_feature'], as_index=False).mean())

dataset.loc[(dataset['Age'] > 0) & (dataset['Age'] <= 16.536), 'Age'] = 0
dataset.loc[(dataset['Age'] > 16.536) & (dataset['Age'] <= 32.402), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32.402) & (dataset['Age'] <= 48.268), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48.268) & (dataset['Age'] <= 64.134), 'Age'] = 3
dataset.loc[(dataset['Age'] > 64.134) & (dataset['Age'] <= 80), 'Age'] = 4
dataset['Age']=dataset['Age'].astype(int)

dataset.loc[(dataset['Fare'] > -0.001) & (dataset['Fare'] <= 7.896), 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.896) & (dataset['Fare'] <= 13.862), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 13.862) & (dataset['Fare'] <= 30.5), 'Fare'] = 2
dataset.loc[(dataset['Fare'] > 30.5) & (dataset['Fare'] <= 512.329), 'Fare'] = 3
dataset['Fare']=dataset['Fare'].astype(int)

features_drop = ['Ticket','CategoricalFare', 'CategoricalAge','IsAlone','FamilySize']
dataset = dataset.drop(features_drop, axis=1)

dataset = dataset.values
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
y = dataset[0::,0]
X = dataset[0::,1::]

# KNN classifier
acc_list = {}
parameter = range(1,10)

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    n_neighbor = [1,2,3,4,5,6,7,8,9,10]
    for k in n_neighbor:
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                                   n_jobs=1, n_neighbors=k, p=2, weights='uniform')
        knn.fit(X_train, y_train)
        train_prediction = knn.predict(X_test)
        acc = accuracy_score(y_test, train_prediction)
        if k in acc_list:
            acc_list[k] += acc
        else:
            acc_list[k] = acc
print(acc_list)

for i in acc_list:
    a = acc_list[i]/10
    print("KNN: When k = ",i, "the accuracy is",a)

lists = sorted(acc_list.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.xlabel('hyper-parameter')
plt.ylabel('accuracy')
plt.show()

# SVM
acc_list = {}

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    C = [1,2,3,4,5,6,7,8,9,10]
    for k in C:
        clf = svm.SVC(C=k, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        clf.fit(X_train, y_train)
        train_prediction = clf.predict(X_test)
        acc = accuracy_score(y_test, train_prediction)
        if k in acc_list:
            acc_list[k] += acc
        else:
            acc_list[k] = acc
print(acc_list)
for i in acc_list:
    a = acc_list[i]/10
    print("SVM: When k = ",i, "the accuracy is",a)
lists = sorted(acc_list.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.xlabel('hyper-parameter')
plt.ylabel('accuracy')
plt.show()


# AdaBoost
acc_list = {}
score = 0

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    C = [50,60,70,80,90,100,200,300,500]
    for k in C:
        # tree = DecisionTreeClassifier(max_depth=k, min_samples_leaf=1)
        # tree.fit(X_train, y_train)
        # ada = AdaBoostClassifier(base_estimator=tree, n_estimators=400, learning_rate=1.0, algorithm="SAMME.R",
        #                          random_state=None)
        ada = AdaBoostClassifier(n_estimators=k)
        ada.fit(X_train, y_train)
        train_prediction = ada.predict(X_test)
        acc = accuracy_score(y_test, train_prediction)
        if k in acc_list:
            acc_list[k] += acc
        else:
            acc_list[k] = acc
print(acc_list)
for i in acc_list:
    a = acc_list[i]/10
    print("AdaBoost: When k = ",i, "the accuracy is",a)

lists = sorted(acc_list.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.xlabel('hyper-parameter')
plt.ylabel('accuracy')
plt.show()

