from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()
#stringing metadata code
"""try:
    print(iris.feature_names)
    print(iris.target_names)
    print(iris.data[0])
except (RuntimeError,AttributeError,KeyError):
    print("Oops Error found,check attributes")"""
test_idx = [0, 50, 100]
#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.target, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))
