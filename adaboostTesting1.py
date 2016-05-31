import numpy as math
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.tree import DecisionTreeClassifier as DecTree
from numpy.random import randint
tmp = []
X = []
Y = []
f = open("ionosphere.data")
for line in f:
    i = 0
    tmp = []
    for word in (line.strip()).split(','):
        i = i+1
        if(i==35):
            tmp.append(word)
        else:
            tmp.append(float(word))
    Y.append(tmp.pop())
    X.append(tmp[:])
#print("X: ")
#print(X)
#print("Y: ")
#print(Y)
f.close
Z = []
odp = []
for k in range(0, 9):
    i = randint(0, 300)
    Z.append(X.pop(i))
    odp.append(Y.pop(i))
#X = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [7, 9, 11, 13, 15], [6, 8, 10, 12, 14]]
#Y = [1, 0, 0, 1]
#Z = [[8, 9, 10, 14, 15], [4, 5, 7, 11, 13], [4, 6, 8, 10, 12],  [3, 5, 7, 9, 11]]

ada = AdaBoost(DecTree(max_depth=1))
ada.fit(X, Y)

print("Predict: ")
print(ada.predict(Z))
print("Probabil: ")
print(ada.predict_proba(Z))
print("Score: ")
print(ada.score(Z, odp))
