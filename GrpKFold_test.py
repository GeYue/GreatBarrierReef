#coding=UTF-8

import numpy as np
from sklearn.model_selection import GroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
groups = np.array([110, 223, 333, 457])
#groups = np.array(['a', 'b', 'c', 'd'])
group_kfold = GroupKFold(n_splits=3)
print (group_kfold.get_n_splits(X, y, groups))

print(group_kfold)

for train_index, test_index in group_kfold.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X-trains=", X_train)
    print("X-tests=", X_test)
    print("y-trains=", y_train)
    print("y-tests=", y_test)
    print ("===========================")
