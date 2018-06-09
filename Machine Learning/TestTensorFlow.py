import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics,cross_validation

def main(unused_argv):
    #Load dataset
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
                                                                         test_size=20, random_state=42)

    #Build 3 layer Deep Neural Network with 10, 20, 30 units respectively
    classifier = learn.DNNClassifier(hidden_units=[10, 20, 30], n_classes=3)


    #Fit and predict
    classifier.fit(x_train, y_train, steps=500)
    score = metrics.accurancy_score(y_test, classifier.predict(x_test))
    print("Accuracy:{0:f}".format(score))
if __name__ == '__main__':
    tf.app.run()

