import numpy as math
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.tree import DecisionTreeClassifier as DecTree

X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
Y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
Z = [[8], [9], [10], [11], [12]]

ada = AdaBoost()
ada.fit(X, Y)

pr = ada.predict(Z)
print(pr)