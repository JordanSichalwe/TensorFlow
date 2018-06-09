import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
mnist.test.cls = np.array([label.argmax() for label in mnist.test.labels])

img_size = 28
img_size_flat = 28 * 28
img_shape = (img_size, img_size)
num_classes = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, img_size_flat])
Y_true = tf.placeholder(tf.float32, [None, num_classes])
Y_true_cls = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(X, weights) + biases
Y_pred = tf.nn.softmax(logits)
Y_pred_cls = tf.argmax(Y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_pred)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.01).minimize(cost)
correct_prediction = tf.equal(Y_pred_cls, Y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {X: x_batch, Y_true: y_true_batch}

        sess.run(optimizer, feed_dict=feed_dict_train)


feed_dict_test = {X: mnist.test.images,
                  Y_true: mnist.test.labels,
                  Y_true_cls: mnist.test.cls}


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = sess.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


if __name__ == "__main__":
    optimize(num_iterations=10)
    print_accuracy()
