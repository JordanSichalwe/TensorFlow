import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class MyClassifier:
    #trainer|function f(y)
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    #Evaluator|Result
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            #pick a random label for test
            #label = random.choice(row)
            label = self.closest(row)
            #add to predictions
            predictions.append(label)
        return predictions
    def closest(self, row):
        #tracking distance and index of value giving best distance
        best_distance = euc(row, self.x_train[0])
        best_index = 0
        #checking result in x_train array to find cosest result
        for i in range(1,len(x_train)):
            dist = euc(row, x_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
        return self.y_train[best_index]

#pipelining : Partitioning
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target
#divide data into 2 sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

#create a classifier
#from sklearn.neighbors import KNeighborsClassifier
try:
    clf2 = MyClassifier()
    clf2.fit(x_train, y_train)

    predictions = clf2.predict(x_test)
    # print(predictions)
    from sklearn.metrics import accuracy_score
    # create Accurancy
    print("My_classiier: %f " % (accuracy_score(y_test, predictions)))
except(TypeError):
    print("%s found" % TypeError)