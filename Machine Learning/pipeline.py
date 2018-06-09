#creating great features
import numpy as np
import matplotlib.pyplot as plt

#dog population for two species
dogA = 500
dogB = 500
#generate random heights
dogA_height =16+2.5*np.random.rand(dogA)
dogB_height =14+2.5*np.random.rand(dogB)
count=0
"""while count < 500:
    print(dogA_height)
    count+=1"""

#visualize in histogram
#plt.hist([dogA_height, dogB_height], stacked=False, color=['r','b'])
# plt.show()

#pipelining : Partitioning
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target
#divide data into 2 sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

#create a Tree classifier
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)

predictions = clf1.predict(x_test)
#print(predictions)

#create Accurancy
from sklearn.metrics import accuracy_score
print("Tree_classiier: %f " % (accuracy_score(y_test, predictions)))

#create a K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier()
clf2.fit(x_train, y_train)

predictions = clf2.predict(x_test)
#print(predictions)

#create Accurancy
print("KN_classiier: %f " % (accuracy_score(y_test, predictions)))